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
    public var outs: MetalPrivateBuffer<Float16>! = nil
    ///
    /// Gradient buffer (result of the backward pass) used in the GPU execution context.
    /// Shape ~ (batch, seq, nbNeurons).
    ///
    public var delta: MetalPrivateBuffer<Float16>! = nil
    
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
            outs = MetalPrivateBuffer<Float16>(
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
                delta = MetalPrivateBuffer<Float16>(
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
    
    /// Get the outputs of this layer in the CPU execution context.
    public func getOutsCPU<T: BinaryFloatingPoint>() -> [T]
    {
        var outs = [T]()
        for elem in 0..<batchSize {
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            let out = T(neurons.get(seq, depth)!.v[elem].out)
            outs.append(out)
        }}}
        return outs
    }
    
    /// Get the outputs of this layer in the GPU execution context.
    public func getOutsGPU<T: BinaryFloatingPoint>() -> [T]
    {
        return outs.download().map
        {
            T($0)
        }
    }
    
    ///
    /// Get the delta of this layer in the CPU execution context.
    ///
    /// Throw an error when layer has not been updated through backward pass.
    ///
    public func getDeltaCPU<T: BinaryFloatingPoint>() throws -> [T]
    {
        if dirty
        {
            throw UpdateError.Dirty
        }
        
        var delta = [T]()
        for elem in 0..<batchSize {
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            let out = T(neurons.get(seq, depth)!.v[elem].delta)
            delta.append(out)
        }}}
        return delta
    }
    
    ///
    /// Get the delta of this layer in the GPU execution context.
    ///
    /// Throw an error when layer has not been updated through backward pass.
    ///
    ///
    public func getDeltaGPU<T: BinaryFloatingPoint>() throws -> [T]
    {
        if dirty
        {
            throw UpdateError.Dirty
        }
        
        return delta.download().map
        {
            T($0)
        }
    }
}
