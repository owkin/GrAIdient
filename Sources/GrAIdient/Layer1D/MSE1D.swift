//
// MSE1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 10/10/2022.
//

/// Output layer with a 1D shape neural structure and a loss that computes mean squared error.
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
        
        let params = GrAI.Model.Params(context: context)
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
    public func collectGradientsApprox<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]]) throws -> [T]
    {
        var gradients = [T]()
        let nbGradients = neurons.get(0)!.nbGC / 2
        for elem in 0..<nbGradients
        {
            let loss1 = try getLossGC(groundTruth, elem: 2 * elem)
            let loss2 = try getLossGC(groundTruth, elem: 2 * elem + 1)
            
            let gradient = (loss1 - loss2) / T(2 * Ɛ)
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
    func getLossGC<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]],
        elem: Int) throws -> T
    {
        let batchSize = groundTruth.count
        if batchSize != self.batchSize ||
           batchSize <= 0 || batchSize > neurons.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
        
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
                let out = T(neurons.get(depth)!.gc[batch][elem].out)
                let diff = out - gt[depth]
                
                losses[batch] += diff * diff
            }
        }
        return T(coeff) * losses.reduce(0, +) /
               T(nbNeurons * batchSize)
    }
    
    ///
    /// Get loss in the CPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter groundTruth: The ground truth.
    /// - Returns: The loss value.
    ///
    public func getLossCPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]]) throws -> T
    {
        let batchSize = groundTruth.count
        if batchSize != self.batchSize ||
           batchSize <= 0 || batchSize > neurons.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
        
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
                let out = T(neurons.get(depth)!.v[elem].out)
                let diff = out - gt[depth]
                
                losses[elem] += diff * diff
            }
        }
        return T(coeff) * losses.reduce(0, +) /
               T(nbNeurons * batchSize)
    }
    
    ///
    /// Get loss in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter groundTruth: The ground truth.
    /// - Returns: The loss value.
    ///
    public func getLossGPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]]) throws -> T
    {
        try checkGroundTruthGPU(groundTruth)
        
        return try T(getLossGPU(
            self.groundTruth,
            batchSize: groundTruth.count
        ))
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
        try checkLossGPU(batchSize: batchSize)
        if batchSize != self.batchSize
        {
            throw LayerError.BatchSize
        }
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let command = MetalKernel.get.createCommand(
            "MSE1DLoss", deviceID: deviceID
        )
        command.setBuffer(outs.metal, atIndex: 0)
        command.setBuffer(groundTruth.metal, atIndex: 1)
        command.setBytes(pNbNeurons, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBuffer(loss.metal, atIndex: 4)
        
        command.dispatchThreads(batchSize)
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
    /// Compute the derivative of the loss in the CPU execution context.
    ///
    /// This function is necessary to initialize the backward pass !
    /// In a way, it plays a similar role as the `setData` of the first layer.
    ///
    /// The `setData` API sets data to the first layer to initialize the forward pass.
    /// Here we use the `groundTruth` to initialize the backward pass.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter groundTruth: The ground truth.
    ///
    public func lossDerivativeCPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]]) throws
    {
        let batchSize = groundTruth.count
        if batchSize != self.batchSize ||
           batchSize <= 0 || batchSize > neurons.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
        
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
                    let out = T(neurons.get(depth)!.v[elem].out)
                    let diff = out - gt[depth]
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(depth)!.v[elem].delta =
                            2 * coeff * Double(diff) /
                            Double(nbNeurons * batchSize)
                    }
                    else
                    {
                        neuronsPrev.get(depth)!.v[elem].delta +=
                            2 * coeff * Double(diff) /
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
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter groundTruth: The ground truth.
    ///
    public func lossDerivativeGPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]]) throws
    {
        try checkGroundTruthGPU(groundTruth)
        
        try lossDerivativeGPU(
            self.groundTruth,
            batchSize: groundTruth.count
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
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameters:
    ///     -  groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///
    public func lossDerivativeGPU(
        _ groundTruth: MetalBuffer<Float>,
        batchSize: Int) throws
    {
        if batchSize != self.batchSize
        {
            throw LayerError.BatchSize
        }
        
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pCoeff: [Float] = [Float(coeff)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "MSE1DLossDerivative", deviceID: deviceID
            )
            command.setBuffer(outs.metal, atIndex: 0)
            command.setBuffer(groundTruth.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pCoeff, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
