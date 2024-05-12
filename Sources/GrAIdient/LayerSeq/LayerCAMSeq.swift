//
// LayerCAMSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 10/02/2024.
//

///
/// Layer with a sequential shape neural structure.
///
/// This layer creates a map of maximal activations with respect to the loss.
///
public class LayerCAMSeq: LayerSeq
{
    /// Whether to take positive or negative part of gradients.
    public var keepPositive: Bool = true
    
    private enum Keys: String, CodingKey
    {
        case keepPositive
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq, params: GrAI.Model.Params) throws
    {
        super.init(layerPrev: layerPrev,
                   sequence: layerPrev.sequence,
                   nbNeurons: 1,
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
        let layerPrev = mapping[idPrev] as! LayerSeq
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = try! LayerCAMSeq(
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
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons!
            let nbNeuronsPrev = layerPrev.nbNeurons
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                var sum = 0.0
                for depthPrev in 0..<nbNeuronsPrev
                {
                    var deltaPrev =
                        neuronsPrev.get(seq, depthPrev)!.v[elem].delta
                    let outPrev =
                        neuronsPrev.get(seq, depthPrev)!.v[elem].out
                    
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
                neurons.get(seq, 0)!.v[elem].out = sum
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
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbNeuronsPrev: [UInt32] = [UInt32(layerPrev.nbNeurons)]
            let pKeepPositive: [UInt32] = keepPositive ? [1] : [0]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            
            let command = MetalKernel.get.createCommand(
                "layerCAMSeqForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(layerPrev.delta.metal, atIndex: 1)
            command.setBytes(pNbNeuronsPrev, atIndex: 2)
            command.setBytes(pKeepPositive, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: sequence,
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
