//
// Layer2D.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

open class Layer2D: Layer
{
    public internal(set) var neurones: [GridNeurones] = []
    
    public internal(set) var outs: MetalPrivateBuffer<Float>! = nil
    public internal(set) var delta: MetalPrivateBuffer<Float>! = nil
    
    public let nbFilters: Int
    public let height: Int
    public let width: Int
    
    public override var nbGC: Int
    {
        get {
            return neurones.first!.get(0)!.nbGC
        }
    }
    
    public var strideFactorCache: Double? = nil
    
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
    
    public var receptiveFieldCache: Int? = nil
    
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
    
    public init(layerPrev: Layer?, nbFilters: Int, height: Int, width: Int,
                params: MAKit.Model.Params)
    {
        self.nbFilters = nbFilters
        self.height = height
        self.width = width
        super.init(layerPrev: layerPrev, params: params)
    }
    
    public required init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        nbFilters = try container.decode(Int.self, forKey: .nbFilters)
        height = try container.decode(Int.self, forKey: .height)
        width = try container.decode(Int.self, forKey: .width)
        try super.init(from: decoder)
    }
    
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
    /// We clean the neurones' state (forward and backward).
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
    /// We clean the neurones' state (forward and backward).
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
    /// We initialize the neurones' state (forward and backward).
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
    /// We initialize the neurones' forward state.
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
    /// We initialize the neurones' backward state.
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
    /// Throws an error when layer has not been updated through backward pass.
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
    /// Throws an error when layer has not been updated through backward pass.
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
