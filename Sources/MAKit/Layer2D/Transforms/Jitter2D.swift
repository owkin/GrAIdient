//
// Jitter2D.swift
// MAKit
//
// Created by Jean-François Reboud on 06/12/2022.
//

///
/// Layer with a 2D shape neural structure.
///
/// Extract output grids that fit into taller input ones.
///
/// Note that his layer can be called in a deterministic way thanks to the `offsetI` & `offsetJ`
/// parameters.
///
public class Jitter2D: Layer2D
{
    let _jitterDimension: Int
    
    var _offsetI: Int = 0
    var _offsetJ: Int = 0
    var _doNotRandom: Bool = false
    
    private enum Keys: String, CodingKey
    {
        case jitterDimension
        case doNotRandom
        case offsetI
        case offsetJ
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - jitterDimension: Dimension to remove on each border of the input grids.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                jitterDimension: Int,
                params: MAKit.Model.Params)
    {
        _jitterDimension = jitterDimension
        
        let width = layerPrev.width - jitterDimension
        let height = layerPrev.height - jitterDimension
        let nbChannels = layerPrev.nbChannels
        
        if width <= 0 || height <= 0
        {
            fatalError(
                "`jitterDimension` should be lower than width and height"
            )
        }
        super.init(layerPrev: layerPrev,
                   nbChannels: nbChannels,
                   height: height,
                   width: width,
                   params: params)
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// This API enables to build a deterministic layer where `offsetI` & `offsetJ` specify
    /// where to select the "inner window".
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - jitterDimension: Dimension to remove on each border of the input grids.
    ///     - offsetI: Height offset.
    ///     - offsetJ: Width offset.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                jitterDimension: Int,
                offsetI: Int,
                offsetJ: Int,
                params: MAKit.Model.Params)
    {
        _doNotRandom = true
        _offsetI = offsetI
        _offsetJ = offsetJ
        _jitterDimension = jitterDimension
        
        let width = layerPrev.width - jitterDimension
        let height = layerPrev.height - jitterDimension
        let nbChannels = layerPrev.nbChannels
        
        if width <= 0 || height <= 0
        {
            fatalError(
                "`jitterDimension` should be lower than width and height."
            )
        }
        if offsetI < 0 || offsetJ < 0 ||
           offsetI >= jitterDimension || offsetJ >= jitterDimension
        {
            fatalError(
                """
                `offsetI` and `offsetJ` should be lower than `jitterDimension`
                and higher than 0.
                """
            )
        }
        super.init(layerPrev: layerPrev,
                   nbChannels: nbChannels,
                   height: height,
                   width: width,
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
        let values = try decoder.container(keyedBy: Keys.self)
        _jitterDimension = try values.decode(
            Int.self, forKey: Keys.jitterDimension
        )
        _doNotRandom = try values.decode(Bool.self, forKey: Keys.doNotRandom)
        _offsetI = try values.decode(Int.self, forKey: Keys.offsetI)
        _offsetJ = try values.decode(Int.self, forKey: Keys.offsetJ)
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
        try container.encode(_jitterDimension, forKey: Keys.jitterDimension)
        try container.encode(_doNotRandom, forKey: Keys.doNotRandom)
        try container.encode(_offsetI, forKey: Keys.offsetI)
        try container.encode(_offsetJ, forKey: Keys.offsetJ)
        try super.encode(to: encoder)
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
        
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
            
        let layer: Jitter2D
        if !_doNotRandom
        {
            layer = Jitter2D(
                layerPrev: layerPrev,
                jitterDimension: _jitterDimension,
                params: params
            )
        }
        else
        {
            layer = Jitter2D(
                layerPrev: layerPrev,
                jitterDimension: _jitterDimension,
                offsetI: _offsetI,
                offsetJ: _offsetJ,
                params: params
            )
        }
        return layer
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.initGC(
                        batchSize: batchSize,
                        nbGC: nbGC
                    )
                }}
            }
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.gc[batch][elem].out =
                        neuronsPrev[depth].get(i+_offsetI,
                                               j+_offsetJ)!.gc[batch][elem].out
                }}
            }}}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try forwardGCCPU()
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            if !_doNotRandom
            {
                _offsetI = Int.random(in: 0..<_jitterDimension)
                _offsetJ = Int.random(in: 0..<_jitterDimension)
            }
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.v[elem].out =
                        neuronsPrev[depth].get(
                            i+_offsetI,
                            j+_offsetJ)!.v[elem].out
                }}
            }}
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            if !_doNotRandom
            {
                _offsetI = Int.random(in: 0..<_jitterDimension)
                _offsetJ = Int.random(in: 0..<_jitterDimension)
            }
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pJitterDimension: [UInt32] = [UInt32(_jitterDimension)]
            let pJitterOffsets: [UInt32] = [UInt32(_offsetI), UInt32(_offsetJ)]
            
            let command = MetalKernel.get.createCommand(
                "jitter2DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pJitterDimension, atIndex: 3)
            command.setBytes(pJitterOffsets, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: width * nbChannels,
                height: height * batchSize
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let widthPrev = layerPrev.width
            let heightPrev = layerPrev.height
            let neuronsPrev = layerPrev.neurons
            
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<heightPrev {
                for j in 0..<widthPrev
                {
                    if layerPrev.dirty &&
                      (i < _offsetI || i >= height + _offsetI ||
                       j < _offsetJ || j >= width + _offsetJ)
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta = 0.0
                    }
                    else if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta =
                            neurons[depth].get(
                                i-_offsetI, j-_offsetJ)!.v[elem].delta
                    }
                    else if i >= _offsetI && i < height + _offsetI &&
                            j >= _offsetJ && j < width + _offsetJ
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            neurons[depth].get(
                                i-_offsetI, j-_offsetJ)!.v[elem].delta
                    }
                }}
            }}
            propagateDirty()
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pJitterDimension: [UInt32] = [UInt32(_jitterDimension)]
            let pJitterOffsets: [UInt32] = [UInt32(_offsetI), UInt32(_offsetJ)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "jitter2DBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pJitterDimension, atIndex: 3)
            command.setBytes(pJitterOffsets, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: layerPrev.width * nbChannels,
                height: layerPrev.height * batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
