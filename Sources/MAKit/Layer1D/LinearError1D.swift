//
// LinearError1D.swift
// MAKit
//
// Created by Jean-François Reboud on 10/10/2022.
//

import MetalKit

/// Last layer with a 1D shape neural structure and a loss function that depends linearly on its inputs.
public class LinearError1D: LayerOutput1D
{
    ///
    /// Create a layer with same values as this.
    ///
    /// - Parameters:
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///     - inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new layer. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    public override func copy(
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = mapping[idPrev] as! Layer1D
        
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
        
        let newLayer = LinearError1D(layerPrev: layerPrev, params: params)
        newLayer.coeff = self.coeff
        
        return newLayer
    }
    
    ///
    /// Estimate the gradients of weights thanks to Gradient Checking.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter groundTruth: The ground truth.
    /// - Returns: The estimated gradients of weights.
    ///
    public func collectGradientsApprox(
        _ groundTruth: [[Double]]) throws -> [Double]
    {
        var gradients = [Double]()
        let nbGradients = neurons.get(0)!.nbGC / 2
        for elem in 0..<nbGradients
        {
            let loss1 = try getLossGC(groundTruth, elem: 2 * elem)
            let loss2 = try getLossGC(groundTruth, elem: 2 * elem + 1)
            
            let gradient = (loss1 - loss2) / Double(2 * Ɛ)
            gradients.append(gradient)
        }
        return gradients
    }
    
    ///
    /// Get the loss consecutive of a modified weights during the Gradient Checking process.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - elem: The modified weight for which we collect the resulting loss.
    /// - Returns: The loss value.
    ///
    func getLossGC(_ groundTruth: [[Double]], elem: Int) throws -> Double
    {
        let batchSize = groundTruth.count
        if batchSize != self.batchSize ||
           batchSize <= 0 || batchSize > neurons.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
        
        var losses = [Double](repeating: 0.0, count: batchSize)
        for batch in 0..<batchSize
        {
            let gt = groundTruth[batch]
            if gt.count != nbNeurons
            {
                throw LayerError.DataSize
            }
            
            for depth in 0..<nbNeurons
            {
                let out = neurons.get(depth)!.gc[batch][elem].out
                let diff = out - gt[depth]
                
                losses[batch] += diff
            }
        }
        return Double(coeff) * losses.reduce(0, +) /
               Double(nbNeurons * batchSize)
    }
    
    ///
    /// Get loss in the CPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter groundTruth: The ground truth.
    /// - Returns: The loss value.
    ///
    public func getLossCPU(_ groundTruth: [[Double]]) throws -> Double
    {
        let batchSize = groundTruth.count
        if batchSize != self.batchSize ||
           batchSize <= 0 || batchSize > neurons.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
        
        var losses = [Double](repeating: 0.0, count: batchSize)
        for elem in 0..<batchSize
        {
            let gt = groundTruth[elem]
            if gt.count != nbNeurons
            {
                throw LayerError.DataSize
            }
            
            for depth in 0..<nbNeurons
            {
                let out = neurons.get(depth)!.v[elem].out
                let diff = out - gt[depth]
                
                losses[elem] += diff
            }
        }
        return Double(coeff) * losses.reduce(0, +) /
               Double(nbNeurons * batchSize)
    }
    
    ///
    /// Get loss in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter groundTruth: The ground truth.
    /// - Returns: The loss value.
    ///
    public func getLossGPU(_ groundTruth: [[Double]]) throws -> Float
    {
        let batchSize = groundTruth.count
        if self.groundTruth == nil
        {
            self.groundTruth = MetalSharedBuffer<Float>(
                batchSize * nbNeurons,
                deviceID: deviceID
            )
        }
        
        let bufferPtr = self.groundTruth.buffer
        for (i, dataI) in groundTruth.enumerated()
        {
            if dataI.count != nbNeurons
            {
                throw LayerError.DataSize
            }
            for (j, dataIJ) in dataI.enumerated()
            {
                bufferPtr[j + i * nbNeurons] = Float(dataIJ)
            }
        }
        MetalKernel.get.upload([self.groundTruth])
        
        return try getLossGPU(
            self.groundTruth,
            batchSize: groundTruth.count
        )
    }
    
    ///
    /// Get loss in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameters:
    ///     -  groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    /// - Returns: The loss value.
    ///
    public func getLossGPU(
        _ groundTruth: MetalBuffer<Float>,
        batchSize: Int) throws -> Float
    {
        if batchSize != self.batchSize
        {
            throw LayerError.BatchSize
        }
        if batchSize * nbNeurons > groundTruth.nbElems
        {
            throw LayerError.DataSize
        }
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        if loss == nil
        {
            loss = MetalSharedBuffer<Float>(batchSize, deviceID: deviceID)
        }
        if batchSize > loss.nbElems
        {
            throw LayerError.BatchSize
        }
        
        let command = MetalKernel.get.createCommand(
            "linearErrorLoss", deviceID: deviceID
        )
        command.setBuffer(outs.metal, atIndex: 0)
        command.setBuffer(groundTruth.metal, atIndex: 1)
        command.setBytes(pNbNeurons, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBuffer(loss.metal, atIndex: 4)
        
        let threads = command.maxThreadsPerThreadgroup
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: batchSize, height: 1, depth: 1)
        command.dispatchThreads(
            threadsPerGrid: threadsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        command.enqueue()
        
        MetalKernel.get.download([loss])
        var loss: Float = 0.0
        let lossPtr = self.loss.buffer
        for i in 0..<batchSize
        {
            loss += lossPtr[i]
        }
        return Float(coeff) * loss / Float(nbNeurons * batchSize)
    }
    
    ///
    /// Apply the gradient in the CPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    public func applyGradientCPU() throws
    {
        if batchSize <= 0 || batchSize > neurons.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
        
        if let layerPrev = self.layerPrev as? Layer1D
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    neuronsPrev.get(depth)!.v[elem].delta =
                        coeff / Double(nbNeurons * batchSize)
                }
            }
            propagateDirty()
        }
    }
    
    ///
    /// Apply the gradient in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    public func applyGradientGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pCoeff: [Float] = [Float(coeff)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            
            if layerPrev.delta == nil
            {
                layerPrev.delta = MetalPrivateBuffer<Float>(
                    batchSize * nbNeurons, deviceID: deviceID)
            }
            if batchSize * nbNeurons > layerPrev.delta.nbElems
            {
                throw LayerError.BatchSize
            }
            
            let command = MetalKernel.get.createCommand(
                "linearErrorApplyGradient", deviceID: deviceID
            )
            command.setBuffer(outs.metal, atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pCoeff, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBuffer(layerPrev.delta.metal, atIndex: 4)
            
            let threadsPerThreadgroup = MTLSizeMake(8, 8, 1)
            let threadsPerGrid = MTLSize(width: nbNeurons,
                                         height: batchSize,
                                         depth: 1)
            command.dispatchThreads(
                threadsPerGrid: threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
