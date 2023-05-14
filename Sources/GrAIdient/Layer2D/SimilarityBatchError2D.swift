//
// SimilarityBatchError2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 14/05/2023.
//

/// Output layer with a 2D shape neural structure and a loss that computes pairwise similarity within batch.
public class SimilarityBatchError2D: LayerOutput2D
{
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public override init(layerPrev: Layer2D, params: GrAI.Model.Params)
    {
        if layerPrev.nbChannels != 1
        {
            fatalError("Previous layer should have only 1 channel.")
        }
        super.init(layerPrev: layerPrev, params: params)
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    public required init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
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
        let layerPrev = mapping[idPrev] as! Layer2D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = SimilarityBatchError2D(layerPrev: layerPrev, params: params)
        layer.coeff = self.coeff
        
        return layer
    }
    
    ///
    /// Estimate the gradients of weights thanks to Gradient Checking.
    ///
    /// - Returns: The estimated gradients of weights.
    ///
    public func collectGradientsApprox<T: BinaryFloatingPoint>() -> [T]
    {
        var gradients = [T]()
        let nbGradients = neurons.first!.get(0)!.nbGC / 2
        for elem in 0..<nbGradients
        {
            let loss1: T = getLossGC(elem: 2 * elem)
            let loss2: T = getLossGC(elem: 2 * elem + 1)
            
            let gradient = (loss1 - loss2) / T(2 * Ɛ)
            gradients.append(gradient)
        }
        return gradients
    }
    
    ///
    /// Get the loss consecutive of a modified weights during the Gradient Checking process.
    ///
    /// - Parameter elem: The modified weight for which we collect the resulting loss.
    /// - Returns: The loss value.
    ///
    func getLossGC<T: BinaryFloatingPoint>(elem: Int) -> T
    {
        var loss = T(0.0)
        for batch1 in 0..<batchSize {
        for batch2 in 0..<batchSize
        {
            if batch1 == batch2
            {
                continue
            }
            
            for i in 0..<height {
            for j in 0..<width
            {
                let out1 = T(neurons[0].get(i, j)!.gc[batch1][elem].out)
                let out2 = T(neurons[0].get(i, j)!.gc[batch2][elem].out)
                
                loss += out1 * out2
            }}
        }}
        return T(coeff) * loss / T(batchSize)
    }
    
    ///
    /// Get loss in the CPU execution context.
    ///
    /// - Returns: The loss value.
    ///
    public func getLossCPU<T: BinaryFloatingPoint>() -> T
    {
        var loss = T(0.0)
        for elem1 in 0..<batchSize {
        for elem2 in 0..<batchSize
        {
            if elem1 == elem2
            {
                continue
            }
            
            for i in 0..<height {
            for j in 0..<width
            {
                let out1 = T(neurons[0].get(i, j)!.v[elem1].out)
                let out2 = T(neurons[0].get(i, j)!.v[elem2].out)
                
                loss += out1 * out2
            }}
        }}
        return T(coeff) * loss / T(batchSize)
    }
    
    ///
    /// Get loss in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    /// - Returns: The loss value.
    ///
    /*public func getLossGPU(
        _ groundTruth: MetalBuffer<Float>,
        batchSize: Int) throws -> Float
    {
        try checkLossGPU(batchSize: batchSize)
        if batchSize != self.batchSize
        {
            throw LayerError.BatchSize
        }
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let command = MetalKernel.get.createCommand(
            "MSE2DLoss", deviceID: deviceID
        )
        command.setBuffer(outs.metal, atIndex: 0)
        command.setBuffer(groundTruth.metal, atIndex: 1)
        command.setBytes(pNbChannels, atIndex: 2)
        command.setBytes(pDimensions, atIndex: 3)
        command.setBytes(pNbBatch, atIndex: 4)
        command.setBuffer(loss.metal, atIndex: 5)
        
        command.dispatchThreads(batchSize)
        command.enqueue()
        
        MetalKernel.get.download([loss])
        var loss: Float = 0.0
        let lossPtr = self.loss.buffer
        for i in 0..<batchSize
        {
            loss += lossPtr[i]
        }
        return Float(coeff) * loss /
               Float(batchSize * nbChannels * height * width)
    }*/
    
    ///
    /// Compute the derivative of the loss in the CPU execution context.
    ///
    /// This function is necessary to initialize the backward pass !
    /// In a way, it plays a similar role as the `setData` of the first layer.
    ///
    public func lossDerivativeCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                var sum = 0.0
                for elem1 in 0..<batchSize
                {
                    if elem1 == elem
                    {
                        continue
                    }
                    sum += 2 * neuronsPrev[0].get(i, j)!.v[elem1].out
                }
                neuronsPrev[0].get(i, j)!.v[elem].delta =
                    coeff / Double(batchSize) * sum
            }}}
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
    /// - Parameters:
    ///     -  groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///
    /*public func lossDerivativeGPU(
        _ groundTruth: MetalBuffer<Float>,
        batchSize: Int) throws
    {
        if batchSize != self.batchSize
        {
            throw LayerError.BatchSize
        }
        
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pCoeff: [Float] = [Float(coeff)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            
            let command = MetalKernel.get.createCommand(
                "MSE2DLossDerivative", deviceID: deviceID
            )
            command.setBuffer(outs.metal, atIndex: 0)
            command.setBuffer(groundTruth.metal, atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pDimensions, atIndex: 3)
            command.setBytes(pCoeff, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbChannels * width,
                height: batchSize * height
            )
            command.enqueue()
            
            propagateDirty()
        }
    }*/
}
