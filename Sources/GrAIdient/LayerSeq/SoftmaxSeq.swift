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
    let _nbHeads: Int
    
    private enum Keys: String, CodingKey
    {
        case nbHeads
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - nbHeads: Number of heads (groups) of neurons.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq, nbHeads: Int, params: GrAI.Model.Params)
    {
        let nbNeurons = layerPrev.nbNeurons
        if nbNeurons % nbHeads != 0
        {
            fatalError(
                "'nbNeurons' (\(nbNeurons) " +
                "should be a multiple of nbHeads (\(nbHeads))."
            )
        }
        
        _nbHeads = nbHeads
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
        let values = try decoder.container(keyedBy: Keys.self)
        _nbHeads = try values.decode(Int.self, forKey: Keys.nbHeads)
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
        try container.encode(_nbHeads, forKey: Keys.nbHeads)
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
            
        let layer = SoftmaxSeq(
            layerPrev: layerPrev,
            nbHeads: _nbHeads,
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
            
            let size = nbNeurons / _nbHeads
            let neuronsPrev = layerPrev.neurons!
            
            for batch in 0..<batchSize {
            for seq in 0..<sequence {
            for elem in 0..<nbGC
            {
                for head in 0..<_nbHeads
                {
                    var cMax = neuronsPrev
                        .get(seq, 0 + head * size)!.gc[batch][elem].out
                    for j in 0..<size
                    {
                        let outPrev = neuronsPrev
                            .get(seq, j + head * size)!.gc[batch][elem].out
                        if outPrev > cMax
                        {
                            cMax = outPrev
                        }
                    }
                    
                    var sum1 = 0.0
                    for j in 0..<size
                    {
                        let outPrev = neuronsPrev
                            .get(seq, j + head * size)!.gc[batch][elem].out
                        sum1 += exp(outPrev - cMax)
                    }
                    
                    for j in 0..<size
                    {
                        let outPrev = neuronsPrev
                            .get(seq, j + head * size)!.gc[batch][elem].out
                        neurons.get(seq, j + head * size)!.gc[batch][elem].out =
                            exp(outPrev - cMax) / sum1
                    }
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
            
            let size = nbNeurons / _nbHeads
            let neuronsPrev = layerPrev.neurons!
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence {
            for head in 0..<_nbHeads
            {
                var cMax = neuronsPrev.get(seq, 0 + head * size)!.v[elem].out
                for j in 0..<size
                {
                    let outPrev = neuronsPrev
                        .get(seq, j + head * size)!.v[elem].out
                    if outPrev > cMax
                    {
                        cMax = outPrev
                    }
                }
                
                var sum1 = 0.0
                for j in 0..<size
                {
                    let outPrev = neuronsPrev
                        .get(seq, j + head * size)!.v[elem].out
                    sum1 += exp(outPrev - cMax)
                }
                
                for j in 0..<size
                {
                    let outPrev = neuronsPrev
                        .get(seq, j + head * size)!.v[elem].out
                    neurons.get(seq, j + head * size)!.v[elem].out =
                        exp(outPrev - cMax) / sum1
                }
            }}}
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
            
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            
            let command = MetalKernel.get.createCommand(
                "softmaxSeqForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbHeads, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pSequence, atIndex: 4)
            command.setBuffer(outs.metal, atIndex: 5)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * sequence
            )
            command.enqueue()
            
            /*MetalKernel.get.download([outs])
            var hum1 = [Float]()
            var hum2 = [Double]()
            let buffer = outs.shared.buffer
            let size = nbNeurons / _nbHeads
            for elem in 0..<batchSize {
            for seq in 0..<sequence {
            for j in 0..<size
            {
                var avg = 0.0
                for head in 0..<_nbHeads
                {
                    let offset = j+head*size +
                        nbNeurons * seq + sequence * nbNeurons * elem
                    avg += Double(buffer[offset])
                }
                avg /= Double(_nbHeads)
                hum2.append(avg)
            }}}
            for elem in buffer
            {
                hum1.append(elem)
            }
            print("COUCOU")*/
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            let size = nbNeurons / _nbHeads
            let neuronsPrev = layerPrev.neurons!
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                for head in 0..<_nbHeads {
                for j in 0..<size
                {
                    let outCur = neurons.get(seq, j + head * size)!.v[elem].out
                    let deltaCur = neurons
                        .get(seq, j + head * size)!.v[elem].delta
                    
                    var sum1: Double = 0.0
                    for j1 in 0..<size
                    {
                        let deltaCur1 = neurons
                            .get(seq, j1 + head * size)!.v[elem].delta
                        let outCur1 = neurons
                            .get(seq, j1 + head * size)!.v[elem].out
                        sum1 += outCur1 * deltaCur1
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(seq, j + head * size)!.v[elem].delta =
                            outCur * (deltaCur - sum1)
                    }
                    else
                    {
                        neuronsPrev.get(seq, j + head * size)!.v[elem].delta +=
                            outCur * (deltaCur - sum1)
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
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "softmaxSeqBackward", deviceID: deviceID
            )
            command.setBuffer(outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * sequence
            )
            command.enqueue()
            
            propagateDirty()
            
            MetalKernel.get.download([layerPrev.delta])
            var hum = [Float]()
            let buffer = layerPrev.delta.shared.buffer
            for elem in buffer
            {
                hum.append(elem)
            }
            print("COUCOU")
        }
    }
}
