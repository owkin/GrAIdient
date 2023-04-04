//
// QuerySeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 09/03/2023.
//

import Foundation

///
/// Layer with a sequential shape neural structure.
///
/// This layer computes the attention scores between a query layer and a key layer.
///
public class QuerySeq: LayerMergeSeq
{
    /// Number of heads (groups) of neurons.
    let _nbHeads: Int
    
    private enum Keys: String, CodingKey
    {
        case nbHeads
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - query: Previous layer containing the query to look for.
    ///     - key: Previous layer containing the keys of reference.
    ///     - nbHeads: Number of heads (groups) of neurons.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(query: LayerSeq, key: LayerSeq, nbHeads: Int,
                params: GrAI.Model.Params)
    {
        if query.nbNeurons % nbHeads != 0
        {
            fatalError(
                "'nbNeurons' (\(query.nbNeurons)) " +
                "should be a multiple of nbHeads (\(nbHeads))."
            )
        }
        if query.nbNeurons != key.nbNeurons ||
           query.sequence != key.sequence
        {
            fatalError("Layer structure error.")
        }
        
        _nbHeads = nbHeads
        super.init(layersPrev: [query, key],
                   sequence: query.sequence,
                   nbNeurons: query.sequence * nbHeads,
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
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        var layersPrev = [LayerSeq]()
        for idPrev in _idsPrev
        {
            layersPrev.append(mapping[idPrev] as! LayerSeq)
        }
        
        let layer = QuerySeq(
            query: layersPrev[0], key: layersPrev[1], nbHeads: _nbHeads,
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
        try checkStateCPU(batchSize: batchSize)
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for seqQ in 0..<sequence {
        for seqK in 0..<nbNeurons
        {
            neurons.get(seqQ, seqK)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }}
        
        let query = (_layersPrev[0] as! LayerSeq).neurons!
        let key = (_layersPrev[1] as! LayerSeq).neurons!
        let size = (_layersPrev[0] as! LayerSeq).nbNeurons / _nbHeads
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for seqK in 0..<sequence {
        for elem in 0..<nbSameElems
        {
            var sum = 0.0
            for j in 0..<size
            {
                let depthPrev = j + head * size
                
                let queryTmp = query.get(seqQ, depthPrev)!.gc[batch][elem].out
                let keyTmp = key.get(seqK, depthPrev)!.gc[batch][elem].out
                
                sum += queryTmp * keyTmp
            }
            
            neurons.get(seqQ, seqK + head * sequence)!.gc[batch][elem].out =
                sum / sqrt(Double(size))
        }}}}}
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for seqK in 0..<sequence {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp
        {
            var sum = 0.0
            for j in 0..<size
            {
                let depthPrev = j + head * size
                
                let queryTmp: Double
                let keyTmp: Double
                
                if index == 0
                {
                    queryTmp = query.get(seqQ, depthPrev)!
                        .gc[batch][nbLastElems[index]+elem].out
                    keyTmp = key.get(seqK, depthPrev)!.v[batch].out
                }
                else
                {
                    queryTmp = query.get(seqQ, depthPrev)!.v[batch].out
                    keyTmp = key.get(seqK, depthPrev)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += queryTmp * keyTmp
            }
            
            neurons.get(seqQ, seqK + head * sequence)!
                .gc[batch][offset+elem].out = sum / sqrt(Double(size))
        }
        
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}}}}
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        for num in 0..<_layersPrev.count
        {
            MetalKernel.get.download([(_layersPrev[num] as! LayerSeq).outs])
        }
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for seqQ in 0..<sequence {
        for seqK in 0..<nbNeurons
        {
            neurons.get(seqQ, seqK)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }}
        
        let query = (_layersPrev[0] as! LayerSeq).neurons!
        let key = (_layersPrev[1] as! LayerSeq).neurons!
        let nbNeuronsPrev = (_layersPrev[0] as! LayerSeq).nbNeurons
        let size = (_layersPrev[0] as! LayerSeq).nbNeurons / _nbHeads
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for seqK in 0..<sequence {
        for elem in 0..<nbSameElems
        {
            var sum = 0.0
            for j in 0..<size
            {
                let depthPrev = j + head * size
                
                let queryTmp = query.get(seqQ, depthPrev)!.gc[batch][elem].out
                let keyTmp = key.get(seqK, depthPrev)!.gc[batch][elem].out
                
                sum += queryTmp * keyTmp
            }
            
            neurons.get(seqQ, seqK + head * sequence)!.gc[batch][elem].out =
                sum / sqrt(Double(size))
        }}}}}
        
        let queryBuffer =
            (_layersPrev[0] as! LayerSeq).outs.shared.buffer
        let keyBuffer =
            (_layersPrev[1] as! LayerSeq).outs.shared.buffer
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for seqK in 0..<sequence {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp
        {
            var sum = 0.0
            for j in 0..<size
            {
                let depthPrev = j + head * size
                
                let queryTmp: Double
                let keyTmp: Double
                
                if index == 0
                {
                    queryTmp = query.get(seqQ, depthPrev)!
                        .gc[batch][nbLastElems[index]+elem].out
                    
                    let offsetTmp = depthPrev + nbNeuronsPrev * seqK +
                        sequence * nbNeuronsPrev * batch
                    
                    keyTmp = Double(keyBuffer[offsetTmp])
                }
                else
                {
                    let offsetTmp = depthPrev + nbNeuronsPrev * seqQ +
                        sequence * nbNeuronsPrev * batch
                    
                    queryTmp = Double(queryBuffer[offsetTmp])
                    
                    keyTmp = key.get(seqK, depthPrev)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += queryTmp * keyTmp
            }
            
            neurons.get(seqQ, seqK + head * sequence)!
                .gc[batch][offset+elem].out = sum / sqrt(Double(size))
        }
        
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}}}}
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        let query = (_layersPrev[0] as! LayerSeq).neurons!
        let key = (_layersPrev[1] as! LayerSeq).neurons!
        let size = (_layersPrev[0] as! LayerSeq).nbNeurons / _nbHeads
        
        for elem in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for seqK in 0..<sequence
        {
            var sum = 0.0
            for j in 0..<size
            {
                let depthPrev = j + head * size
                
                let queryTmp = query.get(seqQ, depthPrev)!.v[elem].out
                let keyTmp = key.get(seqK, depthPrev)!.v[elem].out
                
                sum += queryTmp * keyTmp
            }
            
            neurons.get(seqQ, seqK + head * sequence)!.v[elem].out =
                sum / sqrt(Double(size))
        }}}}
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        try checkStateForwardGPU(batchSize: batchSize)
        
        let query = _layersPrev[0] as! LayerSeq
        let key = _layersPrev[1] as! LayerSeq
        let nbNeuronsPrev = query.nbNeurons
        
        let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let command = MetalKernel.get.createCommand(
            "querySeqForward", deviceID: deviceID
        )
        command.setBuffer(query.outs.metal, atIndex: 0)
        command.setBuffer(key.outs.metal, atIndex: 1)
        command.setBytes(pNbHeads, atIndex: 2)
        command.setBytes(pNbNeurons, atIndex: 3)
        command.setBytes(pNbNeuronsPrev, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
        command.setBytes(pSequence, atIndex: 6)
        command.setBuffer(outs.metal, atIndex: 7)
        
        command.dispatchThreads(
            width: nbNeurons,
            height: batchSize * sequence
        )
        command.enqueue()
        
        MetalKernel.get.download([outs])
        var hum2 = [Float]()
        let buffer2 = outs.shared.buffer
        for elem in buffer2
        {
            hum2.append(elem)
            if elem.isNaN
            {
                print("NaN")
            }
        }
        print("COUCOU")
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if !mustComputeBackward
        {
            return
        }
        
        let query = (_layersPrev[0] as! LayerSeq).neurons!
        let key = (_layersPrev[1] as! LayerSeq).neurons!
        let size = (_layersPrev[0] as! LayerSeq).nbNeurons / _nbHeads
        
        if _layersPrev[0].computeDelta
        {
            for elem in 0..<batchSize {
            for head in 0..<_nbHeads {
            for seqQ in 0..<sequence {
            for j in 0..<size
            {
                let depthPrev = j + head * size
                
                var sum = 0.0
                for seqK in 0..<sequence
                {
                    let deltaCur = neurons
                        .get(seqQ, seqK + head * sequence)!.v[elem].delta
                    let keyTmp = key.get(seqK, depthPrev)!.v[elem].out
                    
                    sum += deltaCur * keyTmp
                }
                
                if _layersPrev[0].dirty
                {
                    query.get(seqQ, depthPrev)!.v[elem].delta =
                        sum / sqrt(Double(size))
                }
                else
                {
                    query.get(seqQ, depthPrev)!.v[elem].delta +=
                        sum / sqrt(Double(size))
                }
            }}}}
        }
        if _layersPrev[1].computeDelta
        {
            for elem in 0..<batchSize {
            for head in 0..<_nbHeads {
            for seqK in 0..<sequence {
            for j in 0..<size
            {
                let depthPrev = j + head * size
                
                var sum = 0.0
                for seqQ in 0..<sequence
                {
                    let deltaCur = neurons
                        .get(seqQ, seqK + head * sequence)!.v[elem].delta
                    let queryTmp = query.get(seqQ, depthPrev)!.v[elem].out
                    
                    sum += deltaCur * queryTmp
                }
                
                if _layersPrev[1].dirty
                {
                    key.get(seqK, depthPrev)!.v[elem].delta =
                        sum / sqrt(Double(size))
                }
                else
                {
                    key.get(seqK, depthPrev)!.v[elem].delta +=
                        sum / sqrt(Double(size))
                }
            }}}}
        }
        propagateDirty()
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        if !mustComputeBackward
        {
            return
        }
        
        let query = _layersPrev[0] as! LayerSeq
        let key = _layersPrev[1] as! LayerSeq
        let nbNeuronsPrev = query.nbNeurons
        
        let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        if query.computeDelta
        {
            try query.checkStateBackwardGPU(batchSize: batchSize)
            
            let pDirty: [UInt32] = query.dirty ? [1] : [0]
            
            command = metalKernel.createCommand(
                "queryQuerySeqBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(key.outs.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbNeuronsPrev, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pSequence, atIndex: 6)
            command.setBytes(pDirty, atIndex: 7)
            command.setBuffer(query.delta.metal, atIndex: 8)
            
            command.dispatchThreads(
                width: nbNeuronsPrev,
                height: batchSize * sequence
            )
            command.enqueue()
        }
        if key.computeDelta
        {
            try key.checkStateBackwardGPU(batchSize: batchSize)
            
            let pDirty: [UInt32] = key.dirty ? [1] : [0]
            
            command = metalKernel.createCommand(
                "queryKeySeqBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(query.outs.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbNeuronsPrev, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pSequence, atIndex: 6)
            command.setBytes(pDirty, atIndex: 7)
            command.setBuffer(key.delta.metal, atIndex: 8)
            
            command.dispatchThreads(
                width: nbNeuronsPrev,
                height: batchSize * sequence
            )
            command.enqueue()
        }
        propagateDirty()
    }
}
