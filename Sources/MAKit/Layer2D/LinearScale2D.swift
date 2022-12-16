//
// LinearScale2D.swift
// MAKit
//
// Created by Jean-François Reboud on 29/11/2022.
//

///
/// Layer with a 2D shape neural structure.
///
/// This layer scales the input with a multiplicative weight and an additive bias.
///
public class LinearScale2D: Layer2D
{
    let _weights: [Double]
    
    private enum Keys: String, CodingKey
    {
        case weights
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - weight: Multiplicative number to scale the input.
    ///     - bias: Additive number to scale the input.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                weight: Double,
                bias: Double,
                params: MAKit.Model.Params)
    {
        _weights = [weight, bias]
        
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
        _weights = try values.decode([Double].self, forKey: Keys.weights)
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
        try container.encode(_weights, forKey: Keys.weights)
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
            
        let layer = LinearScale2D(
            layerPrev: layerPrev,
            weight: _weights[0],
            bias: _weights[1],
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
                    neurons[depth].get(i, j)!.gc[batch][elem].out =
                        neuronsPrev[depth].get(i, j)!.gc[batch][elem].out *
                        _weights[0] + _weights[1]
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
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.v[elem].out =
                        neuronsPrev[depth].get(i, j)!.v[elem].out *
                        _weights[0] + _weights[1]
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
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let weights: [Float] = _weights.map { Float($0) }
            
            let command = MetalKernel.get.createCommand(
                "linearScale2DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(weights, atIndex: 1)
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
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta =
                            neurons[depth].get(i, j)!.v[elem].delta *
                            _weights[0]
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            neurons[depth].get(i, j)!.v[elem].delta *
                            _weights[0]
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
            let weights: [Float] = _weights.map { Float($0) }
            
            let command = MetalKernel.get.createCommand(
                "linearScale2DBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(weights, atIndex: 1)
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
