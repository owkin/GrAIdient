//
// AvgPoolSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 27/02/2023.
//

///
/// Layer with a 1D shape neural structure.
///
/// This layer transforms a Sequential layer into a 1D layer, averaging the neurons in the different sequences.
///
public class AvgPoolSeq: Layer1D
{
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq, params: GrAI.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbNeurons: layerPrev.nbNeurons,
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
        try super.init(from: decoder)
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
            
        let layer = AvgPoolSeq(layerPrev: layerPrev, params: params)
        return layer
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for depth in 0..<nbNeurons
            {
                neurons.get(depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronsPrev = layerPrev.neurons!
            let sequence = layerPrev.sequence
            
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbNeurons
                {
                    var mean = 0.0
                    for seq in 0..<sequence
                    {
                        mean += neuronsPrev.get(seq, depth)!.gc[batch][elem].out
                    }
                    
                    mean /= Double(sequence)
                    neurons.get(depth)!.gc[batch][elem].out = mean
                }
            }}
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
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons!
            let sequence = layerPrev.sequence
            
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    var mean = 0.0
                    for seq in 0..<sequence
                    {
                        mean += neuronsPrev.get(seq, depth)!.v[elem].out
                    }
                    
                    mean /= Double(sequence)
                    neurons.get(depth)!.v[elem].out = mean
                }
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
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(layerPrev.sequence)]
            
            let command = MetalKernel.get.createCommand(
                "avgPoolSeqForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal(), atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pNbBatch, atIndex: 2)
            command.setBytes(pSequence, atIndex: 3)
            command.setBuffer(outs.metal(), atIndex: 4)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons!
            let sequence = layerPrev.sequence
            
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    let deltaCur = neurons.get(depth)!.v[elem].delta
                    
                    for seq in 0..<sequence
                    {
                        if layerPrev.dirty
                        {
                            neuronsPrev.get(seq, depth)!.v[elem].delta =
                                deltaCur / Double(sequence)
                        }
                        else
                        {
                            neuronsPrev.get(seq, depth)!.v[elem].delta +=
                                deltaCur / Double(sequence)
                        }
                    }
                }
            }
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
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(layerPrev.sequence)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "avgPoolSeqBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal(), atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pNbBatch, atIndex: 2)
            command.setBytes(pSequence, atIndex: 3)
            command.setBytes(pDirty, atIndex: 4)
            command.setBuffer(layerPrev.delta.metal(), atIndex: 5)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * layerPrev.sequence
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
