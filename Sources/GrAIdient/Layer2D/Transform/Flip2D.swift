//
// Flip2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 18/05/2023.
//

///
/// Layer with a 2D shape neural structure.
///
/// This layer flips the input horizontally.
///
public class FlipHorizontal2D: Layer2D
{
    let _probability: Double
    
    var _doFlip = false
    
    var forwardKernel: String
    {
        get {
            return "flipHorizontal2DForward"
        }
    }
    var backwardKernel: String
    {
        get {
            return "flipHorizontal2DBackward"
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case probability
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - probability: Probability to flip horizontally.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                probability: Double,
                params: GrAI.Model.Params)
    {
        _probability = probability
        
        let width = layerPrev.width
        let height = layerPrev.height
        let nbChannels = layerPrev.nbChannels
        
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
        _probability = try values.decode(Double.self, forKey: Keys.probability)
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
        try container.encode(_probability, forKey: Keys.probability)
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
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = FlipHorizontal2D(
            layerPrev: layerPrev,
            probability: _probability,
            params: params
        )
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
                    let newValue: Double
                    if _doFlip
                    {
                        newValue = neuronsPrev[depth].get(i, width-1-j)!
                            .gc[batch][elem].out
                    }
                    else
                    {
                        newValue = neuronsPrev[depth].get(i, j)!
                            .gc[batch][elem].out
                    }
                    neurons[depth].get(i, j)!.gc[batch][elem].out = newValue
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
            
            _doFlip = Double.random(in: 0..<1) < _probability
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let newValue: Double
                    if _doFlip
                    {
                        newValue = neuronsPrev[depth].get(i, width-1-j)!
                            .v[elem].out
                    }
                    else
                    {
                        newValue = neuronsPrev[depth].get(i, j)!.v[elem].out
                    }
                    neurons[depth].get(i, j)!.v[elem].out = newValue
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
            
            _doFlip = Double.random(in: 0..<1) < _probability
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDoFlip: [UInt32] = _doFlip ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                forwardKernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pDoFlip, atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pDimensions, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBuffer(outs.metal, atIndex: 5)
            
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
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let newValue: Double
                    if _doFlip
                    {
                        newValue = neurons[depth].get(i, width-1-j)!
                            .v[elem].delta
                    }
                    else
                    {
                        newValue = neurons[depth].get(i, j)!.v[elem].delta
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta = newValue
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta += newValue
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
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            let pDoFlip: [UInt32] = _doFlip ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                backwardKernel, deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pDoFlip, atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pDimensions, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: width * nbChannels,
                height: height * batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}

///
/// Layer with a 2D shape neural structure.
///
/// This layer flips the input vertically.
///
public class FlipVertical2D: FlipHorizontal2D
{
    override var forwardKernel: String
    {
        get {
            return "flipVertical2DForward"
        }
    }
    override var backwardKernel: String
    {
        get {
            return "flipVertical2DBackward"
        }
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
            
        let layer = FlipVertical2D(
            layerPrev: layerPrev,
            probability: _probability,
            params: params
        )
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
                    let newValue: Double
                    if _doFlip
                    {
                        newValue = neuronsPrev[depth].get(height-1-i, j)!
                            .gc[batch][elem].out
                    }
                    else
                    {
                        newValue = neuronsPrev[depth].get(i, j)!
                            .gc[batch][elem].out
                    }
                    neurons[depth].get(i, j)!.gc[batch][elem].out = newValue
                }}
            }}}
        }
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
            
            _doFlip = Double.random(in: 0..<1) < _probability
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let newValue: Double
                    if _doFlip
                    {
                        newValue = neuronsPrev[depth].get(height-1-i, j)!
                            .v[elem].out
                    }
                    else
                    {
                        newValue = neuronsPrev[depth].get(i, j)!.v[elem].out
                    }
                    neurons[depth].get(i, j)!.v[elem].out = newValue
                }}
            }}
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let newValue: Double
                    if _doFlip
                    {
                        newValue = neurons[depth].get(height-1-i, j)!
                            .v[elem].delta
                    }
                    else
                    {
                        newValue = neurons[depth].get(i, j)!.v[elem].delta
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta = newValue
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta += newValue
                    }
                }}
            }}
            propagateDirty()
        }
    }
}
