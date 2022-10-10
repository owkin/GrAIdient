//
// Layer1D.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// Layer with a 1D shape neural structure.
open class Layer1D: Layer
{
    /// Neural structure used in the CPU execution context.
    public internal(set) var neurones: EnsembleNeurones = EnsembleNeurones(0)
    
    /// Output buffer (result of the forward pass) used in the GPU execution context.
    public internal(set) var outs: MetalPrivateBuffer<Float>! = nil
    /// Gradient buffer (result of the backward pass) used in the GPU execution context.
    public internal(set) var delta: MetalPrivateBuffer<Float>! = nil
    
    /// Number of neurons.
    public let nbNeurones: Int
    
    /// Number of different weigths for which we are estimating the gradient during Gradient Checking.
    public override var nbGC: Int
    {
        get {
            return neurones.get(0)!.nbGC
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case nbNeurones
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - nbNeurones: Number of neurons.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer?, nbNeurones: Int, params: MAKit.Model.Params)
    {
        self.nbNeurones = nbNeurones
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
        nbNeurones = try container.decode(Int.self, forKey: .nbNeurones)
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
        try container.encode(nbNeurones, forKey: .nbNeurones)
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
        neurones = EnsembleNeurones(0)
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
    public func checkStateCPU(batchSize: Int) throws
    {
        if neurones.nbElems == 0
        {
            neurones = EnsembleNeurones(nbNeurones)
            for neurone in neurones.all
            {
                neurone.initBatch(batchSize)
            }
        }
        else if batchSize <= 0 || batchSize > neurones.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    ///
    public func checkStateForwardGPU(batchSize: Int) throws
    {
        if outs == nil
        {
            outs = MetalPrivateBuffer<Float>(
                batchSize * nbNeurones, deviceID: deviceID
            )
        }
        else if batchSize <= 0 || batchSize > outs.nbElems / nbNeurones
        {
            throw LayerError.BatchSize
        }
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' backward state.
    ///
    public func checkStateBackwardGPU(batchSize: Int) throws
    {
        if delta == nil
        {
            delta = MetalPrivateBuffer<Float>(
                batchSize * nbNeurones, deviceID: deviceID
            )
        }
        else if batchSize <= 0 || batchSize > delta.nbElems / nbNeurones
        {
            throw LayerError.BatchSize
        }
    }
    
    ///
    /// Get the outputs of this layer in the CPU execution context.
    ///
    /// - Parameter elem: The batch element to retrieve the outputs from.
    ///
    public func getOutsCPU<T: BinaryFloatingPoint>(elem: Int) -> [T]
    {
        var outs = [T]()
        for neurone in neurones.all
        {
            let out = T(neurone.v[elem].out)
            outs.append(out)
        }
        return outs
    }
    
    ///
    /// Get the outputs of this layer in the GPU execution context.
    ///
    /// - Parameter elem: The batch element to retrieve the outputs from.
    ///
    public func getOutsGPU<T: BinaryFloatingPoint>(elem: Int) -> [T]
    {
        var outs = [T]()
        MetalKernel.get.download([self.outs])
        
        let outsPtr = self.outs.shared.buffer
        for depth in 0..<nbNeurones
        {
            let offset = depth + nbNeurones * elem
            outs.append(T(outsPtr[offset]))
        }
        return outs
    }
    
    ///
    /// Get the delta of this layer in the CPU execution context.
    ///
    /// Throw an error when layer has not been updated through backward pass.
    ///
    /// - Parameter elem: The batch element to retrieve the outputs from.
    ///
    public func getDeltaCPU<T: BinaryFloatingPoint>(elem: Int) throws -> [T]
    {
        if dirty
        {
            throw UpdateError.Dirty
        }
        
        var delta = [T]()
        for neurone in neurones.all
        {
            let out = T(neurone.v[elem].delta)
            delta.append(out)
        }
        return delta
    }
    
    ///
    /// Get the delta of this layer in the GPU execution context.
    ///
    /// Throw an error when layer has not been updated through backward pass.
    ///
    /// - Parameter elem: The batch element to retrieve the outputs from.
    ///
    public func getDeltaGPU<T: BinaryFloatingPoint>(elem: Int) throws -> [T]
    {
        if dirty
        {
            throw UpdateError.Dirty
        }
        
        var delta = [T]()
        MetalKernel.get.download([self.delta])
        
        let deltaPtr = self.delta.shared.buffer
        for depth in 0..<nbNeurones
        {
            let offset = depth + nbNeurones * elem
            delta.append(T(deltaPtr[offset]))
        }
        return delta
    }
}