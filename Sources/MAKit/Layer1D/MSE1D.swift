//
// MSE1D.swift
// MAKit
//
// Created by Jean-François Reboud on 10/10/2022.
//

import MetalKit

/// Last layer with a 1D shape neural structure.
public class MSE1D: LayerOutput1D
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
        
        let layer = MSE1D(layerPrev: layerPrev, params: params)
        layer.coeff = self.coeff
        
        return layer
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
        let nbGradients = neurones.get(0)!.nbGC / 2
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
        if batchSize <= 0 || batchSize > neurones.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
        
        var losses = [Double](repeating: 0.0, count: batchSize)
        for batch in 0..<batchSize
        {
            let gt = groundTruth[batch]
            if gt.count != nbNeurones
            {
                throw LayerError.DataSize
            }
            
            for depth in 0..<nbNeurones
            {
                let out = neurones.get(depth)!.gc[batch][elem].out
                let diff = out - gt[depth]
                
                losses[batch] += diff * diff
            }
        }
        return Double(coeff) * losses.reduce(0, +) /
               Double(nbNeurones * batchSize)
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
           batchSize <= 0 || batchSize > neurones.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
        
        var losses = [Double](repeating: 0.0, count: batchSize)
        for elem in 0..<batchSize
        {
            let gt = groundTruth[elem]
            if gt.count != nbNeurones
            {
                throw LayerError.DataSize
            }
            
            for depth in 0..<nbNeurones
            {
                let out = neurones.get(depth)!.v[elem].out
                let diff = out - gt[depth]
                
                losses[elem] += diff * diff
            }
        }
        return Double(coeff) * losses.reduce(0, +) /
               Double(nbNeurones * batchSize)
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
        if batchSize * nbNeurones > groundTruth.nbElems
        {
            throw LayerError.DataSize
        }
        
        let pNbNeurones: [UInt32] = [UInt32(nbNeurones)]
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
            "MSE1DLoss", deviceID: deviceID
        )
        command.setBuffer(outs.metal, atIndex: 0)
        command.setBuffer(groundTruth.metal, atIndex: 1)
        command.setBytes(pNbNeurones, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBuffer(loss.metal, atIndex: 4)
        
        let threads = command.threadExecutionWidth
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
        for i in 0..<self.loss.nbElems
        {
            loss += lossPtr[i]
        }
        return Float(coeff) * loss / Float(nbNeurones * batchSize)
    }
    
    ///
    /// Apply the gradient in the CPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter groundTruth: The ground truth.
    ///
    public func applyGradientCPU(_ groundTruth: [[Double]]) throws
    {
        let batchSize = groundTruth.count
        if batchSize != self.batchSize ||
           batchSize <= 0 || batchSize > neurones.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
        
        if let layerPrev = self.layerPrev as? Layer1D
        {
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize
            {
                let gt = groundTruth[elem]
                if gt.count != nbNeurones
                {
                    throw LayerError.DataSize
                }
                
                for depth in 0..<nbNeurones
                {
                    let out = neurones.get(depth)!.v[elem].out
                    let diff = out - gt[depth]
                    
                    neuronesPrev.get(depth)!.v[elem].delta =
                        2 * coeff * diff / Double(nbNeurones * batchSize)
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
    /// - Parameters:
    ///     -  groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///
    public func applyGradientGPU(
        _ groundTruth: MetalBuffer<Float>,
        batchSize: Int) throws
    {
        if batchSize != self.batchSize
        {
            throw LayerError.BatchSize
        }
        if batchSize * nbNeurones > groundTruth.nbElems
        {
            throw LayerError.DataSize
        }
        
        if let layerPrev = self.layerPrev as? Layer1D
        {
            let pNbNeurones: [UInt32] = [UInt32(nbNeurones)]
            let pCoeff: [Float] = [Float(coeff)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            
            if layerPrev.delta == nil
            {
                layerPrev.delta = MetalPrivateBuffer<Float>(
                    batchSize * nbNeurones, deviceID: deviceID)
            }
            if batchSize * nbNeurones > layerPrev.delta.nbElems
            {
                throw LayerError.BatchSize
            }
            
            let command = MetalKernel.get.createCommand(
                "MSE1DApplyGradient", deviceID: deviceID
            )
            command.setBuffer(outs.metal, atIndex: 0)
            command.setBuffer(groundTruth.metal, atIndex: 1)
            command.setBytes(pNbNeurones, atIndex: 2)
            command.setBytes(pCoeff, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBuffer(layerPrev.delta.metal, atIndex: 5)
            
            let threadsPerThreadgroup = MTLSizeMake(8, 8, 1)
            let threadsPerGrid = MTLSize(width: nbNeurones,
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