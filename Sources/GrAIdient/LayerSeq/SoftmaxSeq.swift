//
// SoftmaxSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 10/03/2023.
//

import Foundation

///
/// Layer with a sequential shape neural structure.
///
/// This layer computes the Softmax function of neurons of a sequential layer.
///
public class SoftmaxSeq: LayerSeq
{
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq, params: GrAI.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   sequence: layerPrev.sequence,
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
            
        let layer = SoftmaxSeq(
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
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for seq in 0..<sequence {
            for depth in 0..<nbNeurons
            {
                neurons.get(seq, depth)!.initGC(
                    batchSize: batchSize, nbGC: nbGC
                )
            }}
            
            let neuronsPrev = layerPrev.neurons!
            
            for batch in 0..<batchSize {
            for seq in 0..<sequence {
            for elem in 0..<nbGC
            {
                var sum1 = 0.0
                for depth1 in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev
                        .get(seq, depth1)!.gc[batch][elem].out
                    sum1 += exp(outPrev)
                }
                
                for depth1 in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev
                        .get(seq, depth1)!.gc[batch][elem].out
                    neurons.get(seq, depth1)!.gc[batch][elem].out =
                        exp(outPrev) / sum1
                }
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
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons!
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                var cMax = 0.0
                for depth1 in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(seq, depth1)!.v[elem].out
                    if outPrev > cMax
                    {
                        cMax = outPrev
                    }
                }
                
                var sum1 = 0.0
                for depth1 in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(seq, depth1)!.v[elem].out
                    sum1 += exp(outPrev - cMax)
                }
                
                for depth1 in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(seq, depth1)!.v[elem].out
                    neurons.get(seq, depth1)!.v[elem].out =
                        exp(outPrev - cMax) / sum1
                }
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
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            
            let command = MetalKernel.get.createCommand(
                "softmaxSeqForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pNbBatch, atIndex: 2)
            command.setBytes(pSequence, atIndex: 3)
            command.setBuffer(outs.metal, atIndex: 4)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * sequence
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
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                var cMax = 0.0
                for depth1 in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(seq, depth1)!.v[elem].out
                    if outPrev > cMax
                    {
                        cMax = outPrev
                    }
                }
                
                var sum1: Double = 0.0
                for depth1 in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(seq, depth1)!.v[elem].out
                    sum1 += exp(outPrev - cMax)
                }
                
                for depth1 in 0..<nbNeurons
                {
                    let outPrev1 = neuronsPrev.get(seq, depth1)!.v[elem].out
                    let deltaCur1 = neurons.get(seq, depth1)!.v[elem].delta
                    
                    var sum2: Double = 0.0
                    for depth2 in 0..<nbNeurons
                    {
                        let deltaCur2 = neurons.get(seq, depth2)!.v[elem].delta
                        let outPrev2 = neuronsPrev.get(seq, depth2)!.v[elem].out
                        sum2 += exp(outPrev1 + outPrev2 - 2 * cMax) * deltaCur2
                    }
                    
                    let tmp = exp(outPrev1 - cMax) * deltaCur1 / sum1
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(seq, depth1)!.v[elem].delta =
                            -sum2 / (sum1 * sum1) + tmp
                    }
                    else
                    {
                        neuronsPrev.get(seq, depth1)!.v[elem].delta +=
                            -sum2 / (sum1 * sum1) + tmp
                    }
                }
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
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "softmaxSeqBackward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pSequence, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * sequence
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
