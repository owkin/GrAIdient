//
// LayerSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 20/02/2023.
//

/// Layer with a sequential shape neural structure.
open class LayerSeq: Layer
{
    /// Neural structure used in the CPU execution context.
    public internal(set) var neurons: GridNeurons! = nil
    
    ///
    /// Output buffer (result of the forward pass) used in the GPU execution context.
    /// Shape ~ (batch, seq, nbNeurons).
    ///
    public var outs: MetalPrivateBuffer<Float>! = nil
    ///
    /// Gradient buffer (result of the backward pass) used in the GPU execution context.
    /// Shape ~ (batch, seq, nbNeurons).
    ///
    public var delta: MetalPrivateBuffer<Float>! = nil
    
    /// Length of the sequence.
    public let sequence: Int
    /// Number of neurons.
    public let nbNeurons: Int
    
    /// Number of different weigths for which we are estimating the gradient during Gradient Checking.
    public override var nbGC: Int
    {
        get {
            return neurons.get(0, 0)!.nbGC
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case sequence
        case nbNeurons
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - sequence: Length of the sequence.
    ///     - nbNeurons: Number of neurons.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer?,
                sequence: Int,
                nbNeurons: Int,
                params: GrAI.Model.Params)
    {
        self.sequence = sequence
        self.nbNeurons = nbNeurons
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
        let container = try decoder.container(keyedBy: Keys.self)
        sequence = try container.decode(Int.self, forKey: .sequence)
        nbNeurons = try container.decode(Int.self, forKey: .nbNeurons)
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
    open override func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        try container.encode(sequence, forKey: .sequence)
        try container.encode(nbNeurons, forKey: .nbNeurons)
        try super.encode(to: encoder)
    }
    
    ///
    /// Clean state resources in the CPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    open override func resetKernelCPU()
    {
        super.resetKernelCPU()
        neurons = nil
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    open override func resetKernelGPU()
    {
        super.resetKernelGPU()
        outs = nil
        delta = nil
    }
    
    ///
    /// Initialize state resources in the CPU execution context.
    ///
    /// We initialize the neurons' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        if neurons == nil
        {
            neurons = GridNeurons(width: nbNeurons, height: sequence)
            for neuron in neurons.all
            {
                neuron.initBatch(batchSize)
            }
        }
        else if batchSize <= 0 || batchSize > neurons.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        if outs == nil
        {
            outs = MetalPrivateBuffer<Float>(
                batchSize * sequence * nbNeurons, deviceID: deviceID
            )
        }
        else if batchSize <= 0 || batchSize > outs.nbElems / nbNeurons
        {
            throw LayerError.BatchSize
        }
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' backward state.
    ///
    public override func checkStateBackwardGPU(batchSize: Int) throws
    {
        if computeDelta
        {
            if delta == nil
            {
                delta = MetalPrivateBuffer<Float>(
                    batchSize * sequence * nbNeurons, deviceID: deviceID
                )
            }
            else if batchSize <= 0 ||
                        batchSize > delta.nbElems / (sequence * nbNeurons)
            {
                throw LayerError.BatchSize
            }
        }
    }
    
    ///
    /// Apply a backward guided pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    /// - Parameter positive:
    ///     if positive, negative gradients are reset
    ///     if not, positive gradients are reset
    ///
    public override func backwardGuidedGPU(positive: Bool) throws
    {
        let nbElems = delta.nbElems
        if nbElems > 0
        {
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command: MetalCommand
            if positive
            {
                command = MetalKernel.get.createCommand(
                    "resetNeg", deviceID: deviceID
                )
            }
            else
            {
                command = MetalKernel.get.createCommand(
                    "resetPos", deviceID: deviceID
                )
            }
            
            command.setBytes(pNbElems, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
        }
        try backwardGPU()
    }
}
