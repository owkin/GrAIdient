//
// LayerOutput1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// Loss layer of a model with a 1D shape neural structure.
open class LayerOutput1D: Layer1D
{
    /// Coefficient to be applied to the loss computation.
    public var coeff: Double = 1.0
    
    ///
    /// Ground truth buffer in the GPU execution context.
    /// Shape ~ (batch, nbNeurons).
    ///
    public internal(set) var groundTruth: MetalSharedBuffer<Float>! = nil
    
    ///
    /// Loss buffer in the GPU execution context.
    /// Shape ~ (batch,).
    ///
    public internal(set) var loss: MetalSharedBuffer<Float>! = nil
    
    private enum Keys: String, CodingKey
    {
        case coeff
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer1D, params: GrAI.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbNeurons: layerPrev.nbNeurons,
                   params: params)
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
        let container = try decoder.container(keyedBy: Keys.self)
        let coeff = try container.decode(Float.self, forKey: .coeff)
        self.coeff = Double(coeff)
        try super.init(from: decoder)
    }
    
    ///
    /// Encode to the disk.
    ///
    /// If the value fails to encode anything, `encoder` will encode an empty
    /// keyed container in its place.
    ///
    /// Throw an error if any values are invalid for the given
    /// encoder's format.
    ///
    /// - Parameter encoder: The encoder to write data to.
    ///
    public override func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        try container.encode(Float(coeff), forKey: .coeff)
        try super.encode(to: encoder)
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    open override func resetKernelGPU()
    {
        super.resetKernelGPU()
        groundTruth = nil
        loss = nil
    }
    
    ///
    /// Setup ground truth state in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter groundTruth: The ground truth.
    ///
    public func checkGroundTruthGPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]]) throws
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
    }
    
    ///
    /// Setup loss state  in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter batchSize: The batch size of data.
    ///
    public func checkLossGPU(batchSize: Int) throws
    {
        if loss == nil
        {
            loss = MetalSharedBuffer<Float>(batchSize, deviceID: deviceID)
        }
        else if batchSize > loss.nbElems
        {
            throw LayerError.BatchSize
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for depth in 0..<nbNeurons
            {
                neurons.get(depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for j in 0..<nbNeurons
            {
                neurons.get(j)!.gc[batch][elem].out =
                    neuronsPrev.get(j)!.gc[batch][elem].out
            }}}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardGCGPU() throws
    {
        try forwardGCCPU()
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for j in 0..<nbNeurons
            {
                neurons.get(j)!.v[elem].out = neuronsPrev.get(j)!.v[elem].out
            }}
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let nbElems = outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command = MetalKernel.get.createCommand(
                "sum1", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    open override func backwardCPU()
    {
        // Note that backward is not called except when it is
        // an intermediate layer.
        // Model.backward is only called on non dirty layers.
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbNeurons
            {
                let delta = neurons.get(depth)!.v[elem].delta
                if layerPrev.dirty
                {
                    neuronsPrev.get(depth)!.v[elem].delta = delta
                }
                else
                {
                    neuronsPrev.get(depth)!.v[elem].delta += delta
                }
            }}
            propagateDirty()
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func backwardGPU() throws
    {
        // Note that backward is not called except when it is
        // an intermediate layer.
        // Model.backward is only called on non dirty layers.
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command: MetalCommand
            if layerPrev.dirty
            {
                command = MetalKernel.get.createCommand(
                    "sum1", deviceID: deviceID
                )
            }
            else
            {
                command = MetalKernel.get.createCommand(
                    "sum2", deviceID: deviceID
                )
            }
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(layerPrev.delta.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
            
            propagateDirty()
        }
    }
}
