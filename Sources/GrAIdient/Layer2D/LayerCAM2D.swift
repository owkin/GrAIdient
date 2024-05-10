//
// LayerCAM2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 10/02/2024.
//

///
/// Layer with a 2D shape neural structure.
///
/// This layer creates a map of maximal activations with respect to the loss.
///
public class LayerCAM2D: Layer2D
{
    /// Whether to take positive or negative part of gradients.
    public var keepPositive: Bool = true
    
    private enum Keys: String, CodingKey
    {
        case keepPositive
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D, params: GrAI.Model.Params) throws
    {
        super.init(layerPrev: layerPrev,
                   nbChannels: 1,
                   height: layerPrev.height,
                   width: layerPrev.width,
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
        let keepPositive = try container.decode(
            Bool.self, forKey: .keepPositive
        )
        self.keepPositive = keepPositive
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
        try container.encode(keepPositive, forKey: .keepPositive)
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
            
        let layer = try! LayerCAM2D(
            layerPrev: layerPrev,
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
        fatalError("Not implemented.")
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
            let nbChannelsPrev = layerPrev.nbChannels
            
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var sum = 0.0
                    for depthPrev in 0..<nbChannelsPrev
                    {
                        var deltaPrev =
                            neuronsPrev[depthPrev].get(i, j)!.v[elem].delta
                        let outPrev =
                            neuronsPrev[depthPrev].get(i, j)!.v[elem].out
                        
                        if !keepPositive
                        {
                            deltaPrev = -deltaPrev
                        }
                        if deltaPrev < 0
                        {
                            deltaPrev = 0.0
                        }
                        sum += deltaPrev * outPrev
                    }
                    neurons[0].get(i, j)!.v[elem].out = sum
                }}
            }
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
            
            let pNbChannelsPrev: [UInt32] = [UInt32(layerPrev.nbChannels)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pKeepPositive: [UInt32] = keepPositive ? [1] : [0]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            
            let command = MetalKernel.get.createCommand(
                "layerCAM2DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal(), atIndex: 0)
            command.setBuffer(layerPrev.delta.metal(), atIndex: 1)
            command.setBytes(pNbChannelsPrev, atIndex: 2)
            command.setBytes(pDimensions, atIndex: 3)
            command.setBytes(pKeepPositive, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBuffer(outs.metal(), atIndex: 6)
            
            command.dispatchThreads(
                width: height * width,
                height: batchSize
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        fatalError("Not implemented.")
    }
}
