//
// Layer2D.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// Layer with a 2D shape neural structure.
open class Layer2D: Layer
{
    /// Neural structure used in the CPU execution context.
    public internal(set) var neurones: [GridNeurones] = []
    
    /// Output buffer (result of the forward pass) used in the GPU execution context.
    public internal(set) var outs: MetalPrivateBuffer<Float>! = nil
    /// Gradient buffer (result of the backward pass) used in the GPU execution context.
    public internal(set) var delta: MetalPrivateBuffer<Float>! = nil
    
    /// Number of channels.
    public let nbFilters: Int
    /// Height of each channel.
    public let height: Int
    /// Width of each channel.
    public let width: Int
    
    /// Number of different weigths for which we are estimating the gradient during Gradient Checking.
    public override var nbGC: Int
    {
        get {
            return neurones.first!.get(0)!.nbGC
        }
    }
    
    /// Downscale factor of the resolution (height and width) in cache.
    public var strideFactorCache: Double? = nil
    
    /// Downscale factor of the resolution (height and width).
    open var strideFactor: Double
    {
        get {
            if let layerPrev = self.layerPrev as? Layer2D
            {
                return layerPrev.strideFactor
            }
            return 1.0
        }
    }
    
    /// Cache for the size of the input image this layer is looking at.
    public var receptiveFieldCache: Int? = nil
    
    /// The size of the input image this layer is looking at.
    open var receptiveField: Int
    {
        get {
            if let layerPrev = self.layerPrev as? Layer2D
            {
                return layerPrev.receptiveField
            }
            return 1
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case nbFilters
        case height
        case width
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - nbFilters: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer?, nbFilters: Int, height: Int, width: Int,
                params: MAKit.Model.Params)
    {
        self.nbFilters = nbFilters
        self.height = height
        self.width = width
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
        nbFilters = try container.decode(Int.self, forKey: .nbFilters)
        height = try container.decode(Int.self, forKey: .height)
        width = try container.decode(Int.self, forKey: .width)
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
        try container.encode(nbFilters, forKey: .nbFilters)
        try container.encode(height, forKey: .height)
        try container.encode(width, forKey: .width)
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
        strideFactorCache = nil
        receptiveFieldCache = nil
        neurones = []
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    open override func resetKernelGPU()
    {
        super.resetKernelGPU()
        strideFactorCache = nil
        receptiveFieldCache = nil
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
        if neurones.count == 0
        {
            neurones = []
            for _ in 0..<nbFilters
            {
                neurones.append(GridNeurones(width: width, height: height))
            }
            for grid in neurones {
            for neurone in grid.all
            {
                neurone.initBatch(batchSize)
            }}
        }
        else if batchSize <= 0 || batchSize > neurones.first!.get(0)!.v.count
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
                batchSize * nbFilters * width * height, deviceID: deviceID
            )
        }
        else if batchSize <= 0 ||
                batchSize > outs.nbElems / (nbFilters * width * height)
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
                batchSize * nbFilters * width * height, deviceID: deviceID
            )
        }
        else if batchSize <= 0 ||
                batchSize > delta.nbElems / (nbFilters * width * height)
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
        for grid in neurones {
        for neurone in grid.all
        {
            let out = T(neurone.v[elem].out)
            outs.append(out)
        }}
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
        for depth in 0..<nbFilters
        {
            let offsetStart = (depth + nbFilters * elem) * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                outs.append(T(outsPtr[offset]))
            }}
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
        for grid in neurones {
        for neurone in grid.all
        {
            let out = T(neurone.v[elem].delta)
            delta.append(out)
        }}
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
        for depth in 0..<nbFilters
        {
            let offsetStart = (depth + nbFilters * elem) * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                delta.append(T(deltaPtr[offset]))
            }}
        }
        return delta
    }
}