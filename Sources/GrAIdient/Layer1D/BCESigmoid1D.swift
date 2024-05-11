//
// BCESigmoid1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 07/07/2023.
//

import Foundation

///
/// Output layer with a 1D shape neural structure and a loss that computes binary cross entropy on top
/// of a sigmoid activation.
///
public class BCESigmoid1D: LayerOutput1D
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
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = BCESigmoid1D(layerPrev: layerPrev, params: params)
        layer.coeff = self.coeff
        
        return layer
    }
    
    ///
    /// Estimate the gradients of weights thanks to Gradient Checking.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    /// - Returns: The estimated gradients of weights.
    ///
    public func collectGradientsApprox<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]],
        batchSize: Int,
        nbNeurons: Int) throws -> [T]
    {
        var gradients = [T]()
        let nbGradients = neurons.get(0)!.nbGC / 2
        for elem in 0..<nbGradients
        {
            let loss1 = try getLossGC(
                groundTruth,
                batchSize: batchSize,
                nbNeurons: nbNeurons,
                elem: 2 * elem
            )
            let loss2 = try getLossGC(
                groundTruth,
                batchSize: batchSize,
                nbNeurons: nbNeurons,
                elem: 2 * elem + 1
            )
            
            let gradient = (loss1 - loss2) / T(2 * Ɛ)
            gradients.append(gradient)
        }
        return gradients
    }
    
    ///
    /// Get the loss consecutive of a modified weights during the Gradient Checking process.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    ///     - elem: The modified weight for which we collect the resulting loss.
    /// - Returns: The loss value.
    ///
    func getLossGC<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]],
        batchSize: Int,
        nbNeurons: Int,
        elem: Int) throws -> T
    {
        try checkGroundTruthCPU(
            groundTruth,
            batchSize: batchSize,
            nbNeurons: nbNeurons
        )
        
        var losses = [T](repeating: 0.0, count: batchSize)
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
                var value: T
                
                if out > 0
                {
                    value = T(1 - gt[depth]) * T(out)
                    value += T(log(1 + exp(-out)))
                }
                else
                {
                    value = -T(out) * T(gt[depth])
                    value += T(log(exp(out) + 1))
                }
                
                losses[batch] += value
            }
        }
        return T(coeff) * losses.reduce(0, +) /
               T(nbNeurons * batchSize)
    }
    
    ///
    /// Get loss in the CPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    /// - Returns: The loss value.
    ///
    public func getLossCPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]],
        batchSize: Int,
        nbNeurons: Int) throws -> T
    {
        try checkGroundTruthCPU(
            groundTruth,
            batchSize: batchSize,
            nbNeurons: nbNeurons
        )
        
        var losses = [T](repeating: 0.0, count: batchSize)
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
                var value: T
                
                if out > 0
                {
                    value = T(1 - gt[depth]) * T(out)
                    value += T(log(1 + exp(-out)))
                }
                else
                {
                    value = -T(out) * T(gt[depth])
                    value += T(log(exp(out) + 1))
                }
                
                losses[elem] += value
            }
        }
        return T(coeff) * losses.reduce(0, +) /
               T(nbNeurons * batchSize)
    }
    
    ///
    /// Get loss in the GPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    /// - Returns: The loss value.
    ///
    public func getLossGPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]],
        batchSize: Int,
        nbNeurons: Int) throws -> T
    {
        try checkGroundTruthGPU(
            groundTruth,
            batchSize: batchSize,
            nbNeurons: nbNeurons
        )
        return try T(getLossGPU(
            self.groundTruth,
            batchSize: groundTruth.count,
            nbNeurons: nbNeurons
        ))
    }
    
    ///
    /// Get loss in the GPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     -  groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    /// - Returns: The loss value.
    ///
    public func getLossGPU(
        _ groundTruth: FloatBuffer,
        batchSize: Int,
        nbNeurons: Int) throws -> Float
    {
        try checkGroundTruthGPU(
            groundTruth,
            batchSize: batchSize,
            nbNeurons: nbNeurons
        )
        try checkLossGPU(batchSize: batchSize)
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let command = MetalKernel.get.createCommand(
            "BCESigmoid1DLoss", deviceID: deviceID
        )
        command.setBuffer(outs.metal(), atIndex: 0)
        command.setBuffer(groundTruth.metal(), atIndex: 1)
        command.setBytes(pNbNeurons, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBuffer(loss.metal(), atIndex: 4)
        
        command.dispatchThreads(batchSize)
        command.enqueue()
        
        var loss: Float = 0.0
        let lossPtr = self.loss.download()
        for i in 0..<batchSize
        {
            loss += lossPtr[i]
        }
        return Float(coeff) * loss / Float(nbNeurons * batchSize)
    }
    
    ///
    /// Compute the derivative of the loss in the CPU execution context.
    ///
    /// This function is necessary to initialize the backward pass !
    /// In a way, it plays a similar role as the `setData` of the first layer.
    ///
    /// The `setData` API sets data to the first layer to initialize the forward pass.
    /// Here we use the `groundTruth` to initialize the backward pass.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    ///
    public func lossDerivativeCPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]],
        batchSize: Int,
        nbNeurons: Int) throws
    {
        try checkGroundTruthCPU(
            groundTruth,
            batchSize: batchSize,
            nbNeurons: nbNeurons
        )
        
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
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
                    let value: Double
                    
                    if out >= 0
                    {
                        value = 1.0 / (1.0 + exp(-out))
                    }
                    else
                    {
                        value = exp(out) / (1.0 + exp(out))
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(depth)!.v[elem].delta =
                            coeff * (value - Double(gt[depth])) /
                            Double(nbNeurons * batchSize)
                    }
                    else
                    {
                        neuronsPrev.get(depth)!.v[elem].delta +=
                            coeff * (value - Double(gt[depth])) /
                            Double(nbNeurons * batchSize)
                    }
                }
            }
            propagateDirty()
        }
    }
    
    ///
    /// Compute the derivative of the loss in the GPU execution context.
    ///
    /// This function is necessary to initialize the backward pass !
    /// In a way, it plays a similar role as the `setData` of the first layer.
    ///
    /// The `setData` API sets data to the first layer to initialize the forward pass.
    /// Here we use the `groundTruth` to initialize the backward pass.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    ///
    public func lossDerivativeGPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]],
        batchSize: Int,
        nbNeurons: Int) throws
    {
        try checkGroundTruthGPU(
            groundTruth,
            batchSize: batchSize,
            nbNeurons: nbNeurons
        )
        try lossDerivativeGPU(
            self.groundTruth,
            batchSize: groundTruth.count,
            nbNeurons: nbNeurons
        )
    }
    
    ///
    /// Compute the derivative of the loss in the GPU execution context.
    ///
    /// This function is necessary to initialize the backward pass !
    /// In a way, it plays a similar role as the `setData` of the first layer.
    ///
    /// The `setData` API sets data to the first layer to initialize the forward pass.
    /// Here we use the `groundTruth` to initialize the backward pass.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     -  groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    ///
    public func lossDerivativeGPU(
        _ groundTruth: FloatBuffer,
        batchSize: Int,
        nbNeurons: Int) throws
    {
        try checkGroundTruthGPU(
            groundTruth,
            batchSize: batchSize,
            nbNeurons: nbNeurons
        )
        
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pCoeff: [Float] = [Float(coeff)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "BCESigmoid1DLossDerivative", deviceID: deviceID
            )
            command.setBuffer(outs.metal(), atIndex: 0)
            command.setBuffer(groundTruth.metal(), atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pCoeff, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal(), atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
