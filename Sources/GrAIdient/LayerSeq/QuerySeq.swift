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
                params: GrAI.Model.Params) throws
    {
        if query.nbNeurons % nbHeads != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(query.nbNeurons)) " +
                "should be a multiple of nbHeads (\(nbHeads))."
            )
        }
        if query.nbNeurons != key.nbNeurons ||
           query.sequence != key.sequence
        {
            throw LayerError.Init(message: "Layer structure error.")
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
        
        let layer = try! QuerySeq(
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
        
        let kernel = (nbNeuronsPrev / _nbHeads) % 4 == 0 ?
            "querySeq4Forward" : "querySeqForward"
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: deviceID
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
            
            let kernel = (nbNeuronsPrev / _nbHeads) % 4 == 0 ?
                "queryQuerySeq4Backward" : "queryQuerySeqBackward"
            let coeff = (nbNeuronsPrev / _nbHeads) % 4 == 0 ? 4 : 1
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
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
                width: nbNeuronsPrev / coeff,
                height: batchSize * sequence
            )
            command.enqueue()
        }
        if key.computeDelta
        {
            try key.checkStateBackwardGPU(batchSize: batchSize)
            
            let pDirty: [UInt32] = key.dirty ? [1] : [0]
            
            let kernel = (nbNeuronsPrev / _nbHeads) % 4 == 0 ?
                "queryKeySeq4Backward" : "queryKeySeqBackward"
            let coeff = (nbNeuronsPrev / _nbHeads) % 4 == 0 ? 4 : 1
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
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
                width: nbNeuronsPrev / coeff,
                height: batchSize * sequence
            )
            command.enqueue()
        }
        propagateDirty()
    }
}

///
/// Layer with a sequential shape neural structure.
///
/// This layer computes the attention scores between 2 different groups of neurons in the previous layer.
///
public class QuerySelfSeq: LayerSeq
{
    
    /// Number of heads (groups) of neurons in the current layer.
    let _nbHeads: Int
    /// Offset of neurons for the query in the previous layer.
    let _queryOffset: Int
    /// Offset of neurons for the key in the previous layer.
    let _keyOffset: Int
    /// Number of different groups of neurons in the previous layer.
    let _nbBlocksPrev: Int
    
    private enum Keys: String, CodingKey
    {
        case nbHeads
        case queryOffset
        case keyOffset
        case nbBlocksPrev
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer.
    ///     - query: Offset of neurons for the query in the previous layer.
    ///     - key: Offset of neurons for the key in the previous layer.
    ///     - nbBlocksPrev: Number of different groups of neurons in the previous layer.
    ///     - nbHeads: Number of heads (groups) of neurons in the current layer.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq, 
                query: Int, key: Int,
                nbBlocksPrev: Int, nbHeads: Int,
                params: GrAI.Model.Params) throws
    {
        if layerPrev.nbNeurons % nbBlocksPrev != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(layerPrev.nbNeurons)) " +
                "should be a multiple of nbBlocks (\(nbBlocksPrev))."
            )
        }
        if layerPrev.nbNeurons % nbHeads != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(layerPrev.nbNeurons)) " +
                "should be a multiple of nbHeads (\(nbHeads))."
            )
        }
        
        _nbHeads = nbHeads
        _nbBlocksPrev = nbBlocksPrev
        _queryOffset = query
        _keyOffset = key
        
        super.init(layerPrev: layerPrev,
                   sequence: layerPrev.sequence,
                   nbNeurons: layerPrev.sequence * nbHeads,
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
        _queryOffset = try values.decode(Int.self, forKey: Keys.queryOffset)
        _keyOffset = try values.decode(Int.self, forKey: Keys.keyOffset)
        _nbBlocksPrev = try values.decode(Int.self, forKey: Keys.nbBlocksPrev)
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
        try container.encode(_queryOffset, forKey: Keys.queryOffset)
        try container.encode(_keyOffset, forKey: Keys.keyOffset)
        try container.encode(_nbBlocksPrev, forKey: Keys.nbBlocksPrev)
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
        
        let layer = try! QuerySelfSeq(
            layerPrev: layerPrev,
            query: _queryOffset,
            key: _keyOffset,
            nbBlocksPrev: _nbBlocksPrev,
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
            for seqQ in 0..<sequence {
            for seqK in 0..<nbNeurons
            {
                neurons.get(seqQ, seqK)!.initGC(
                    batchSize: batchSize, nbGC: nbGC
                )
            }}
            
            let neuronsPrev = layerPrev.neurons!
            let nbNeuronsPrev = layerPrev.nbNeurons / _nbBlocksPrev
            let size = nbNeuronsPrev / _nbHeads
            
            for batch in 0..<batchSize {
            for head in 0..<_nbHeads {
            for seqQ in 0..<sequence {
            for seqK in 0..<sequence {
            for elem in 0..<nbGC
            {
                var sum = 0.0
                for j in 0..<size
                {
                    let depthPrev = j + head * size
                    
                    let queryTmp = neuronsPrev.get(
                        seqQ, depthPrev + _queryOffset * nbNeuronsPrev
                    )!.gc[batch][elem].out
                    let keyTmp = neuronsPrev.get(
                        seqK, depthPrev + _keyOffset * nbNeuronsPrev
                    )!.gc[batch][elem].out
                    
                    sum += queryTmp * keyTmp
                }
                
                neurons.get(seqQ, seqK + head * sequence)!.gc[batch][elem].out =
                    sum / sqrt(Double(size))
            }}}}}
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
            let nbNeuronsPrev = layerPrev.nbNeurons / _nbBlocksPrev
            let size = nbNeuronsPrev / _nbHeads
            
            for elem in 0..<batchSize {
            for head in 0..<_nbHeads {
            for seqQ in 0..<sequence {
            for seqK in 0..<sequence
            {
                var sum = 0.0
                for j in 0..<size
                {
                    let depthPrev = j + head * size
                    
                    let queryTmp = neuronsPrev.get(
                        seqQ, depthPrev + _queryOffset * nbNeuronsPrev
                    )!.v[elem].out
                    let keyTmp = neuronsPrev.get(
                        seqK, depthPrev + _keyOffset * nbNeuronsPrev
                    )!.v[elem].out
                    
                    sum += queryTmp * keyTmp
                }
                
                neurons.get(seqQ, seqK + head * sequence)!.v[elem].out =
                    sum / sqrt(Double(size))
            }}}}
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
            
            let nbNeuronsPrev1 = layerPrev.nbNeurons
            let nbNeuronsPrev2 = nbNeuronsPrev1 / _nbBlocksPrev
            
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev1)]
            let pNbBlocksPrev: [UInt32] = [UInt32(_nbBlocksPrev)]
            let pGlobalOffset: [UInt32] = [
                UInt32(_queryOffset), UInt32(_keyOffset)
            ]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            
            let kernel = (nbNeuronsPrev2 / _nbHeads) % 4 == 0 ?
                "querySelfSeq4Forward" : "querySelfSeqForward"
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbHeads, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbNeuronsPrev, atIndex: 3)
            command.setBytes(pNbBlocksPrev, atIndex: 4)
            command.setBytes(pGlobalOffset, atIndex: 5)
            command.setBytes(pNbBatch, atIndex: 6)
            command.setBytes(pSequence, atIndex: 7)
            command.setBuffer(outs.metal, atIndex: 8)
            
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
            let nbNeuronsPrev = layerPrev.nbNeurons / _nbBlocksPrev
            let size = nbNeuronsPrev / _nbHeads
            
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
                    let keyTmp = neuronsPrev.get(
                        seqK, depthPrev + _keyOffset * nbNeuronsPrev
                    )!.v[elem].out
                    
                    sum += deltaCur * keyTmp
                }
                
                if layerPrev.dirty
                {
                    neuronsPrev.get(
                        seqQ, depthPrev + _queryOffset * nbNeuronsPrev
                    )!.v[elem].delta = sum / sqrt(Double(size))
                }
                else
                {
                    neuronsPrev.get(
                        seqQ, depthPrev + _queryOffset * nbNeuronsPrev
                    )!.v[elem].delta += sum / sqrt(Double(size))
                }
            }}}}
        
        
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
                    let queryTmp = neuronsPrev.get(
                        seqQ, depthPrev + _queryOffset)!.v[elem].out
                    
                    sum += deltaCur * queryTmp
                }
                
                if layerPrev.dirty
                {
                    neuronsPrev.get(
                        seqK, depthPrev + _keyOffset * nbNeuronsPrev
                    )!.v[elem].delta = sum / sqrt(Double(size))
                }
                else
                {
                    neuronsPrev.get(
                        seqK, depthPrev + _keyOffset * nbNeuronsPrev
                    )!.v[elem].delta += sum / sqrt(Double(size))
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
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let nbNeuronsPrev1 = layerPrev.nbNeurons
            let nbNeuronsPrev2 = nbNeuronsPrev1 / _nbBlocksPrev
            
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev1)]
            let pNbBlocksPrev: [UInt32] = [UInt32(_nbBlocksPrev)]
            let pGlobalOffset: [UInt32] = [
                UInt32(_queryOffset), UInt32(_keyOffset)
            ]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            
            let metalKernel = MetalKernel.get
            var command: MetalCommand
                
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            var kernel = (nbNeuronsPrev2 / _nbHeads) % 4 == 0 ?
                "querySelfQuerySeq4Backward" : "querySelfQuerySeqBackward"
            let coeff = (nbNeuronsPrev2 / _nbHeads) % 4 == 0 ? 4 : 1
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbNeuronsPrev, atIndex: 4)
            command.setBytes(pNbBlocksPrev, atIndex: 5)
            command.setBytes(pGlobalOffset, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pSequence, atIndex: 8)
            command.setBytes(pDirty, atIndex: 9)
            command.setBuffer(layerPrev.delta.metal, atIndex: 10)
            
            command.dispatchThreads(
                width: nbNeuronsPrev2 / coeff,
                height: batchSize * sequence
            )
            command.enqueue()
            
            kernel = (nbNeuronsPrev2 / _nbHeads) % 4 == 0 ?
                "querySelfKeySeq4Backward" : "querySelfKeySeqBackward"
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbNeuronsPrev, atIndex: 4)
            command.setBytes(pNbBlocksPrev, atIndex: 5)
            command.setBytes(pGlobalOffset, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pSequence, atIndex: 8)
            command.setBytes(pDirty, atIndex: 9)
            command.setBuffer(layerPrev.delta.metal, atIndex: 10)
            
            command.dispatchThreads(
                width: nbNeuronsPrev2 / coeff,
                height: batchSize * sequence
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
