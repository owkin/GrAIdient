//
// ValueSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 10/03/2023.
//

///
/// Layer with a sequential shape neural structure.
///
/// This layer computes the values (value layer) that are scaled through attention scores (score layer).
///
public class ValueSeq: LayerMergeSeq
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
    ///     - value: Previous layer containing the value.
    ///     - score: Previous layer contianing the attention scores per sequence.
    ///     - nbHeads: Number of heads (groups) of neurons.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(value: LayerSeq, score: LayerSeq, nbHeads: Int,
                params: GrAI.Model.Params) throws
    {
        if value.nbNeurons % nbHeads != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(value.nbNeurons)) " +
                "should be a multiple of nbHeads (\(nbHeads))."
            )
        }
        if score.nbNeurons % nbHeads != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(score.nbNeurons)) " +
                "should be a multiple of nbHeads (\(nbHeads))."
            )
        }
        if value.sequence != score.sequence
        {
            throw LayerError.Init(message: "Layer structure error.")
        }

        _nbHeads = nbHeads
        super.init(layersPrev: [value, score],
                   sequence: value.sequence,
                   nbNeurons: value.nbNeurons,
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
        for idPrev in idsPrev
        {
            layersPrev.append(mapping[idPrev] as! LayerSeq)
        }
        
        let layer = try! ValueSeq(
            value: layersPrev[0], score: layersPrev[1], nbHeads: _nbHeads,
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
        
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }}
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        let size = (layersPrev[0] as! LayerSeq).nbNeurons / _nbHeads
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for j in 0..<size {
        for elem in 0..<nbSameElems
        {
            let depth = j + head * size
            
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp = value.get(seqK, depth)!.gc[batch][elem].out
                let scoreTmp = score
                    .get(seqQ, seqK + head * sequence)!.gc[batch][elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][elem].out = sum
        }}}}}
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for j in 0..<size {
        let depth = j + head * size
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp
        {
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp: Double
                let scoreTmp: Double
                
                if index == 0
                {
                    valueTmp = value.get(seqK, depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                    scoreTmp = score
                        .get(seqQ, seqK + head * sequence)!.v[batch].out
                }
                else
                {
                    valueTmp = value.get(seqK, depth)!.v[batch].out
                    scoreTmp = score.get(seqQ, seqK + head * sequence)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][offset+elem].out = sum
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
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }}
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        let size = (layersPrev[0] as! LayerSeq).nbNeurons / _nbHeads
        let nbNeuronsPrev = (layersPrev[1] as! LayerSeq).nbNeurons
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for j in 0..<size {
        for elem in 0..<nbSameElems
        {
            let depth = j + head * size
            
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp = value.get(seqK, depth)!.gc[batch][elem].out
                let scoreTmp = score
                    .get(seqQ, seqK + head * sequence)!.gc[batch][elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][elem].out = sum
        }}}}}
        
        let valueBuffer = (layersPrev[0] as! LayerSeq).outs.download()
        let scoreBuffer = (layersPrev[1] as! LayerSeq).outs.download()
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for j in 0..<size {
        let depth = j + head * size
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp
        {
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp: Double
                let scoreTmp: Double
                
                if index == 0
                {
                    valueTmp = value.get(seqK, depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                    
                    let offsetTmp = seqK + head * sequence +
                        nbNeuronsPrev * seqQ +
                        sequence * nbNeuronsPrev * batch
                    
                    scoreTmp = Double(scoreBuffer[offsetTmp])
                }
                else
                {
                    let offsetTmp = depth + nbNeurons * seqK +
                        sequence * nbNeurons * batch
                    
                    valueTmp = Double(valueBuffer[offsetTmp])
                    
                    scoreTmp = score.get(seqQ, seqK + head * sequence)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][offset+elem].out = sum
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
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        let size = (layersPrev[0] as! LayerSeq).nbNeurons / _nbHeads
        
        for elem in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for j in 0..<size
        {
            let depth = j + head * size
            
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp = value.get(seqK, depth)!.v[elem].out
                let scoreTmp = score
                    .get(seqQ, seqK + head * sequence)!.v[elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.v[elem].out = sum
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
        
        let value = layersPrev[0] as! LayerSeq
        let score = layersPrev[1] as! LayerSeq
        let nbNeuronsPrev = score.nbNeurons
        
        let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let kernel = (nbNeurons / _nbHeads) % 4 == 0 ?
            "valueSeq4Forward" : "valueSeqForward"
        let coeff = (nbNeurons / _nbHeads) % 4 == 0 ? 4 : 1
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: deviceID
        )
        command.setBuffer(value.outs.metal, atIndex: 0)
        command.setBuffer(score.outs.metal, atIndex: 1)
        command.setBytes(pNbHeads, atIndex: 2)
        command.setBytes(pNbNeurons, atIndex: 3)
        command.setBytes(pNbNeuronsPrev, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
        command.setBytes(pSequence, atIndex: 6)
        command.setBuffer(outs.metal, atIndex: 7)
        
        command.dispatchThreads(
            width: nbNeurons / coeff,
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
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        let size = (layersPrev[0] as! LayerSeq).nbNeurons / _nbHeads
        
        if layersPrev[0].computeDelta
        {
            for elem in 0..<batchSize {
            for head in 0..<_nbHeads {
            for seqK in 0..<sequence {
            for j in 0..<size
            {
                let depth = j + head * size
                
                var sum = 0.0
                for seqQ in 0..<sequence
                {
                    let deltaCur = neurons.get(seqQ, depth)!.v[elem].delta
                    let scoreTmp = score
                        .get(seqQ, seqK + head * sequence)!.v[elem].out
                    
                    sum += deltaCur * scoreTmp
                }
                
                if layersPrev[0].dirty
                {
                    value.get(seqK, depth)!.v[elem].delta = sum
                }
                else
                {
                    value.get(seqK, depth)!.v[elem].delta += sum
                }
            }}}}
        }
        if layersPrev[1].computeDelta
        {
            for elem in 0..<batchSize {
            for head in 0..<_nbHeads {
            for seqQ in 0..<sequence {
            for seqK in 0..<sequence
            {
                var sum = 0.0
                for j in 0..<size
                {
                    let depth = j + head * size
                    
                    let deltaCur = neurons.get(seqQ, depth)!.v[elem].delta
                    let valueTmp = value.get(seqK, depth)!.v[elem].out
                    
                    sum += deltaCur * valueTmp
                }
                
                if layersPrev[1].dirty
                {
                    score.get(seqQ, seqK + head * sequence)!
                        .v[elem].delta = sum
                }
                else
                {
                    score.get(seqQ, seqK + head * sequence)!
                        .v[elem].delta += sum
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
        
        let value = layersPrev[0] as! LayerSeq
        let score = layersPrev[1] as! LayerSeq
        let nbNeuronsPrev = score.nbNeurons
        
        let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        if value.computeDelta
        {
            try value.checkStateBackwardGPU(batchSize: batchSize)
            
            let pDirty: [UInt32] = value.dirty ? [1] : [0]
            
            let kernel = (nbNeurons / _nbHeads) % 4 == 0 ?
                "valueValueSeq4Backward" : "valueValueSeqBackward"
            let coeff = (nbNeurons / _nbHeads) % 4 == 0 ? 4 : 1
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(score.outs.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbNeuronsPrev, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pSequence, atIndex: 6)
            command.setBytes(pDirty, atIndex: 7)
            command.setBuffer(value.delta.metal, atIndex: 8)
            
            command.dispatchThreads(
                width: nbNeurons / coeff,
                height: batchSize * sequence
            )
            command.enqueue()
        }
        if score.computeDelta
        {
            try score.checkStateBackwardGPU(batchSize: batchSize)
            
            let pDirty: [UInt32] = score.dirty ? [1] : [0]
            
            let kernel = (nbNeurons / _nbHeads) % 4 == 0 ?
                "valueScoreSeq4Backward" : "valueScoreSeqBackward"
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(value.outs.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbNeuronsPrev, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pSequence, atIndex: 6)
            command.setBytes(pDirty, atIndex: 7)
            command.setBuffer(score.delta.metal, atIndex: 8)
            
            command.dispatchThreads(
                width: nbNeuronsPrev,
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
/// This layer computes the values (value layer) that are scaled through attention scores (score layer).
///
public class ValueSelfSeq: LayerMergeSeq
{
    /// Number of heads (groups) of neurons.
    let _nbHeads: Int
    /// Offset of neurons for the value in the previous layer.
    let _valueOffset: Int
    /// Number of different groups of neurons in the previous layer.
    let _nbBlocksPrev: Int
    
    private enum Keys: String, CodingKey
    {
        case nbHeads
        case valueOffset
        case nbBlocksPrev
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - value: Previous layer containing the value.
    ///     - score: Previous layer contianing the attention scores per sequence.
    ///     - offset: Offset of neurons for the value in the previous layer.
    ///     - nbBlocksPrev: Number of different groups of neurons in the previous layer.
    ///     - nbHeads: Number of heads (groups) of neurons.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(value: LayerSeq, score: LayerSeq, 
                offset: Int, nbBlocksPrev: Int, nbHeads: Int,
                params: GrAI.Model.Params) throws
    {
        if value.nbNeurons % nbHeads != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(value.nbNeurons)) " +
                "should be a multiple of nbHeads (\(nbHeads))."
            )
        }
        if score.nbNeurons % nbHeads != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(score.nbNeurons)) " +
                "should be a multiple of nbHeads (\(nbHeads))."
            )
        }
        if value.sequence != score.sequence
        {
            throw LayerError.Init(message: "Layer structure error.")
        }

        _nbHeads = nbHeads
        _nbBlocksPrev = nbBlocksPrev
        _valueOffset = offset
        
        super.init(layersPrev: [value, score],
                   sequence: value.sequence,
                   nbNeurons: value.nbNeurons / nbBlocksPrev,
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
        _valueOffset = try values.decode(Int.self, forKey: Keys.valueOffset)
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
        try container.encode(_valueOffset, forKey: Keys.valueOffset)
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
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        var layersPrev = [LayerSeq]()
        for idPrev in idsPrev
        {
            layersPrev.append(mapping[idPrev] as! LayerSeq)
        }
        
        let layer = try! ValueSelfSeq(
            value: layersPrev[0], score: layersPrev[1], 
            offset: _valueOffset, nbBlocksPrev: _nbBlocksPrev,
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
        try checkStateCPU(batchSize: batchSize)
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }}
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        let size = nbNeurons / _nbHeads
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for j in 0..<size {
        for elem in 0..<nbSameElems
        {
            let depth = j + head * size
            
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp = value.get(
                    seqK, depth + _valueOffset * nbNeurons
                )!.gc[batch][elem].out
                let scoreTmp = score
                    .get(seqQ, seqK + head * sequence)!.gc[batch][elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][elem].out = sum
        }}}}}
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for j in 0..<size {
        let depth = j + head * size
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp
        {
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp: Double
                let scoreTmp: Double
                
                if index == 0
                {
                    valueTmp = value.get(
                        seqK, depth + _valueOffset * nbNeurons
                    )!.gc[batch][nbLastElems[index]+elem].out
                    scoreTmp = score
                        .get(seqQ, seqK + head * sequence)!.v[batch].out
                }
                else
                {
                    valueTmp = value.get(
                        seqK, depth + _valueOffset * nbNeurons
                    )!.v[batch].out
                    scoreTmp = score.get(seqQ, seqK + head * sequence)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][offset+elem].out = sum
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
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }}
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        let size = nbNeurons / _nbHeads
        
        let nbNeurons1 = nbNeurons * _nbBlocksPrev
        let nbNeurons2 = nbNeurons
        let nbNeuronsPrev = (layersPrev[1] as! LayerSeq).nbNeurons
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for j in 0..<size {
        for elem in 0..<nbSameElems
        {
            let depth = j + head * size
            
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp = value.get(
                    seqK, depth + _valueOffset * nbNeurons2
                )!.gc[batch][elem].out
                let scoreTmp = score
                    .get(seqQ, seqK + head * sequence)!.gc[batch][elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][elem].out = sum
        }}}}}
        
        let valueBuffer = (layersPrev[0] as! LayerSeq).outs.download()
        let scoreBuffer = (layersPrev[1] as! LayerSeq).outs.download()
        
        for batch in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for j in 0..<size {
        let depth = j + head * size
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp
        {
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp: Double
                let scoreTmp: Double
                
                if index == 0
                {
                    valueTmp = value.get(
                        seqK, depth + _valueOffset * nbNeurons2
                    )!.gc[batch][nbLastElems[index]+elem].out
                    
                    let offsetTmp = seqK + head * sequence +
                        nbNeuronsPrev * seqQ +
                        sequence * nbNeuronsPrev * batch
                    
                    scoreTmp = Double(scoreBuffer[offsetTmp])
                }
                else
                {
                    let offsetTmp = depth + _valueOffset * nbNeurons2 +
                        nbNeurons1 * seqK + sequence * nbNeurons1 * batch
                    
                    valueTmp = Double(valueBuffer[offsetTmp])
                    
                    scoreTmp = score.get(seqQ, seqK + head * sequence)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][offset+elem].out = sum
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
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        let size = nbNeurons / _nbHeads
        
        for elem in 0..<batchSize {
        for head in 0..<_nbHeads {
        for seqQ in 0..<sequence {
        for j in 0..<size
        {
            let depth = j + head * size
            
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp = value.get(
                    seqK, depth + _valueOffset * nbNeurons
                )!.v[elem].out
                let scoreTmp = score
                    .get(seqQ, seqK + head * sequence)!.v[elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.v[elem].out = sum
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
        
        let value = layersPrev[0] as! LayerSeq
        let score = layersPrev[1] as! LayerSeq
        let nbNeuronsPrev = score.nbNeurons
        
        let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
        let pNbBlocksPrev: [UInt32] = [UInt32(_nbBlocksPrev)]
        let pGlobalOffset: [UInt32] = [UInt32(_valueOffset)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let kernel = (nbNeurons / _nbHeads) % 4 == 0 ?
            "valueSelfSeq4Forward" : "valueSelfSeqForward"
        let coeff = (nbNeurons / _nbHeads) % 4 == 0 ? 4 : 1
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: deviceID
        )
        command.setBuffer(value.outs.metal, atIndex: 0)
        command.setBuffer(score.outs.metal, atIndex: 1)
        command.setBytes(pNbHeads, atIndex: 2)
        command.setBytes(pNbNeurons, atIndex: 3)
        command.setBytes(pNbNeuronsPrev, atIndex: 4)
        command.setBytes(pNbBlocksPrev, atIndex: 5)
        command.setBytes(pGlobalOffset, atIndex: 6)
        command.setBytes(pNbBatch, atIndex: 7)
        command.setBytes(pSequence, atIndex: 8)
        command.setBuffer(outs.metal, atIndex: 9)
        
        command.dispatchThreads(
            width: nbNeurons / coeff,
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
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        let size = nbNeurons / _nbHeads
        
        if layersPrev[0].computeDelta
        {
            if layersPrev[0].dirty
            {
                for elem in 0..<batchSize {
                for seqK in 0..<sequence {
                for depth in 0..<nbNeurons * _nbBlocksPrev
                {
                    value.get(seqK, depth)!.v[elem].delta = 0
                }}}
            }
            
            for elem in 0..<batchSize {
            for head in 0..<_nbHeads {
            for seqK in 0..<sequence {
            for j in 0..<size
            {
                let depth = j + head * size
                
                var sum = 0.0
                for seqQ in 0..<sequence
                {
                    let deltaCur = neurons.get(seqQ, depth)!.v[elem].delta
                    let scoreTmp = score
                        .get(seqQ, seqK + head * sequence)!.v[elem].out
                    
                    sum += deltaCur * scoreTmp
                }
                
                value.get(
                    seqK, depth + _valueOffset * nbNeurons
                )!.v[elem].delta += sum
            }}}}
        }
        if layersPrev[1].computeDelta
        {
            for elem in 0..<batchSize {
            for head in 0..<_nbHeads {
            for seqQ in 0..<sequence {
            for seqK in 0..<sequence
            {
                var sum = 0.0
                for j in 0..<size
                {
                    let depth = j + head * size
                    
                    let deltaCur = neurons.get(seqQ, depth)!.v[elem].delta
                    let valueTmp = value.get(
                        seqK, depth + _valueOffset * nbNeurons
                    )!.v[elem].out
                    
                    sum += deltaCur * valueTmp
                }
                
                if layersPrev[1].dirty
                {
                    score.get(seqQ, seqK + head * sequence)!
                        .v[elem].delta = sum
                }
                else
                {
                    score.get(seqQ, seqK + head * sequence)!
                        .v[elem].delta += sum
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
        
        let value = layersPrev[0] as! LayerSeq
        let score = layersPrev[1] as! LayerSeq
        let nbNeuronsPrev = score.nbNeurons
        
        let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
        let pNbBlocksPrev: [UInt32] = [UInt32(_nbBlocksPrev)]
        let pGlobalOffset: [UInt32] = [UInt32(_valueOffset)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        if value.computeDelta
        {
            try value.checkStateBackwardGPU(batchSize: batchSize)
            
            if value.dirty
            {
                let nbElems = value.delta.nbElems
                let pNbElems: [UInt32] = [UInt32(nbElems)]
                
                command = MetalKernel.get.createCommand(
                    "reset", deviceID: deviceID
                )
                command.setBytes(pNbElems, atIndex: 0)
                command.setBuffer(value.delta.metal, atIndex: 1)
                
                command.dispatchThreads(nbElems)
                command.enqueue()
            }
            
            let kernel = (nbNeurons / _nbHeads) % 4 == 0 ?
                "valueSelfValueSeq4Backward" : "valueSelfValueSeqBackward"
            let coeff = (nbNeurons / _nbHeads) % 4 == 0 ? 4 : 1
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(score.outs.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbNeuronsPrev, atIndex: 4)
            command.setBytes(pNbBlocksPrev, atIndex: 5)
            command.setBytes(pGlobalOffset, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pSequence, atIndex: 8)
            command.setBuffer(value.delta.metal, atIndex: 9)
            
            command.dispatchThreads(
                width: nbNeurons / coeff,
                height: batchSize * sequence
            )
            command.enqueue()
        }
        if score.computeDelta
        {
            try score.checkStateBackwardGPU(batchSize: batchSize)
            
            let pDirty: [UInt32] = score.dirty ? [1] : [0]
            
            let kernel = (nbNeurons / _nbHeads) % 4 == 0 ?
                "valueSelfScoreSeq4Backward" : "valueSelfScoreSeqBackward"
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(value.outs.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbNeuronsPrev, atIndex: 4)
            command.setBytes(pNbBlocksPrev, atIndex: 5)
            command.setBytes(pGlobalOffset, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pSequence, atIndex: 8)
            command.setBytes(pDirty, atIndex: 9)
            command.setBuffer(score.delta.metal, atIndex: 10)
            
            command.dispatchThreads(
                width: nbNeuronsPrev,
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
/// This layer computes the values (value layer) that are scaled through causal attention scores (score layer).
///
public class ValueCausalSeq: LayerMergeSeq
{
    /// Number of heads (groups) of neurons for score.
    let _nbHeadsScore: Int
    /// Number of heads (groups) of neurons for value.
    let _nbHeadsValue: Int
    
    /// Cache value of shape (batch, cacheSeqMax, nbNeurons).
    public var cacheValue: FloatBuffer! = nil
    /// Cache value of shape (batch, cacheSeqMax, nbNeurons).
    var _cacheValueTmp: FloatBuffer! = nil
    
    /// Maximal sequence of cache.
    public var cacheSeqMax = 128
    
    /// Current cache sequence.
    public var cacheSeq: Int! = nil
    
    private enum Keys: String, CodingKey
    {
        case nbHeadsScore
        case nbHeadsValue
        case cacheSeqMax
        case cacheSeq
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - value: Previous layer containing the value.
    ///     - score: Previous layer contianing the attention scores per sequence.
    ///     - nbHeadsValue: Number of heads (groups) of neurons for value.
    ///     - nbHeadsScore: Number of heads (groups) of neurons for score.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(value: LayerSeq, score: LayerSeq,
                nbHeadsValue: Int, nbHeadsScore: Int,
                params: GrAI.Model.Params) throws
    {
        if value.nbNeurons % nbHeadsValue != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(value.nbNeurons)) " +
                "should be a multiple of `nbHeadsValue` (\(nbHeadsValue))."
            )
        }
        if score.nbNeurons % nbHeadsScore != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(score.nbNeurons)) " +
                "should be a multiple of `nbHeadsScore` (\(nbHeadsScore))."
            )
        }
        if nbHeadsScore % nbHeadsValue != 0
        {
            throw LayerError.Init(message:
                "`nbHeadsScore` should be a multiple of `nbHeadsValue`"
            )
        }
        if value.sequence != score.sequence
        {
            throw LayerError.Init(message: "Layer structure error.")
        }

        _nbHeadsScore = nbHeadsScore
        _nbHeadsValue = nbHeadsValue
        
        super.init(layersPrev: [value, score],
                   sequence: value.sequence,
                   nbNeurons: (value.nbNeurons / nbHeadsValue) * nbHeadsScore,
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
        _nbHeadsScore = try values.decode(Int.self, forKey: Keys.nbHeadsScore)
        _nbHeadsValue = try values.decode(Int.self, forKey: Keys.nbHeadsValue)
        cacheSeqMax = try values.decode(Int.self, forKey: Keys.cacheSeqMax)
        cacheSeq = try values.decodeIfPresent(Int.self, forKey: .cacheSeq)
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
        try container.encode(_nbHeadsScore, forKey: Keys.nbHeadsScore)
        try container.encode(_nbHeadsValue, forKey: Keys.nbHeadsValue)
        try container.encode(cacheSeqMax, forKey: Keys.cacheSeqMax)
        if cacheSeq != nil
        {
            try container.encode(cacheSeq, forKey: Keys.cacheSeq)
        }
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
        for idPrev in idsPrev
        {
            layersPrev.append(mapping[idPrev] as! LayerSeq)
        }
        
        let layer = try! ValueCausalSeq(
            value: layersPrev[0], score: layersPrev[1],
            nbHeadsValue: _nbHeadsValue,
            nbHeadsScore: _nbHeadsScore,
            params: params
        )
        
        layer.cacheSeqMax = cacheSeqMax
        layer.cacheSeq = cacheSeq
        
        return layer
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We first clean the neurons' state (forward and backward).
    /// We do not clean weights and biases but must reset their delta (dependent on batch size) and
    /// momentum state.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        
        cacheValue = nil
        _cacheValueTmp = nil
        cacheSeq = nil
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        let value = layersPrev[0] as! LayerSeq
        let nbNeuronsPrevValue = value.nbNeurons
        
        if cacheValue != nil && cacheSeq != nil &&
           cacheValue.nbElems != batchSize * cacheSeqMax * nbNeuronsPrevValue
        {
            let cacheValueTmp = FloatBuffer(
                nbElems: batchSize * cacheSeqMax * nbNeuronsPrevValue,
                deviceID: deviceID
            )
            
            let nbElems = batchSize * cacheSeq * nbNeuronsPrevValue
            _copyGPU(nbElems: nbElems, from: cacheValue, to: cacheValueTmp)
            
            cacheValue = FloatBuffer(
                nbElems: batchSize * cacheSeqMax * nbNeuronsPrevValue,
                deviceID: deviceID
            )
            
            _copyGPU(nbElems: nbElems, from: cacheValueTmp, to: cacheValue)
            
            if batchSize > 1
            {
                _cacheValueTmp = cacheValueTmp
            }
        }
    }
    
    ///
    /// Copy buffer.
    ///
    /// - Parameters:
    ///     - nbElems: Number of elements to copy.
    ///     - from: Input buffer.
    ///     - to: Ouptut buffer.
    ///
    private func _copyGPU(
        nbElems: Int, from: FloatBuffer, to: FloatBuffer)
    {
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        
        let kernel = nbElems % 4 == 0 ? "sum14" : "sum1"
        let coeff = nbElems % 4 == 0 ? 4 : 1
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: deviceID
        )
        command.setBuffer(from.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBuffer(to.metal, atIndex: 2)
        
        command.dispatchThreads(nbElems / coeff)
        command.enqueue()
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
        
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }}
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        
        let size = (layersPrev[0] as! LayerSeq).nbNeurons / _nbHeadsValue
        let nbBlocksHead = _nbHeadsScore / _nbHeadsValue
        
        for batch in 0..<batchSize {
        for headScore in 0..<_nbHeadsScore {
        let headValue = headScore / nbBlocksHead
        for seqQ in 0..<sequence {
        for j in 0..<size {
        for elem in 0..<nbSameElems
        {
            let depthScore = j + headScore * size
            let depthValue = j + headValue * size
            
            var sum = 0.0
            for seqK in 0...seqQ
            {
                let valueTmp = value.get(seqK, depthValue)!.gc[batch][elem].out
                let scoreTmp = score
                    .get(seqQ, seqK + headScore * sequence)!.gc[batch][elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depthScore)!.gc[batch][elem].out = sum
        }}}}}
        
        for batch in 0..<batchSize {
        for headScore in 0..<_nbHeadsScore {
        let headValue = headScore / nbBlocksHead
        for seqQ in 0..<sequence {
        for j in 0..<size {
        let depthScore = j + headScore * size
        let depthValue = j + headValue * size
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp
        {
            var sum = 0.0
            for seqK in 0...seqQ
            {
                let valueTmp: Double
                let scoreTmp: Double
                
                if index == 0
                {
                    valueTmp = value.get(seqK, depthValue)!
                        .gc[batch][nbLastElems[index]+elem].out
                    scoreTmp = score
                        .get(seqQ, seqK + headScore * sequence)!.v[batch].out
                }
                else
                {
                    valueTmp = value.get(seqK, depthValue)!.v[batch].out
                    scoreTmp = score.get(seqQ, seqK + headScore * sequence)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depthScore)!.gc[batch][offset+elem].out = sum
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
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }}
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        
        let nbNeuronsPrevValue = (layersPrev[0] as! LayerSeq).nbNeurons
        let nbNeuronsPrevScore = (layersPrev[1] as! LayerSeq).nbNeurons
        
        let size = (layersPrev[0] as! LayerSeq).nbNeurons / _nbHeadsValue
        let nbBlocksHead = _nbHeadsScore / _nbHeadsValue
        
        for batch in 0..<batchSize {
        for headScore in 0..<_nbHeadsScore {
        let headValue = headScore / nbBlocksHead
        for seqQ in 0..<sequence {
        for j in 0..<size {
        for elem in 0..<nbSameElems
        {
            let depthScore = j + headScore * size
            let depthValue = j + headValue * size
            
            var sum = 0.0
            for seqK in 0...seqQ
            {
                let valueTmp = value.get(seqK, depthValue)!.gc[batch][elem].out
                let scoreTmp = score
                    .get(seqQ, seqK + headScore * sequence)!.gc[batch][elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depthScore)!.gc[batch][elem].out = sum
        }}}}}
        
        let valueBuffer = (layersPrev[0] as! LayerSeq).outs.download()
        let scoreBuffer = (layersPrev[1] as! LayerSeq).outs.download()
        
        for batch in 0..<batchSize {
        for headScore in 0..<_nbHeadsScore {
        let headValue = headScore / nbBlocksHead
        for seqQ in 0..<sequence {
        for j in 0..<size {
        let depthScore = j + headScore * size
        let depthValue = j + headValue * size
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp
        {
            var sum = 0.0
            for seqK in 0...seqQ
            {
                let valueTmp: Double
                let scoreTmp: Double
                
                if index == 0
                {
                    valueTmp = value.get(seqK, depthValue)!
                        .gc[batch][nbLastElems[index]+elem].out
                    
                    let offsetTmp = seqK + headScore * sequence +
                        nbNeuronsPrevScore * seqQ +
                        sequence * nbNeuronsPrevScore * batch
                    
                    scoreTmp = Double(scoreBuffer[offsetTmp])
                }
                else
                {
                    let offsetTmp = depthValue + nbNeuronsPrevValue * seqK +
                        sequence * nbNeuronsPrevValue * batch
                    
                    valueTmp = Double(valueBuffer[offsetTmp])
                    
                    scoreTmp = score.get(seqQ, seqK + headScore * sequence)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depthScore)!.gc[batch][offset+elem].out = sum
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
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        
        let size = (layersPrev[0] as! LayerSeq).nbNeurons / _nbHeadsValue
        let nbBlocksHead = _nbHeadsScore / _nbHeadsValue
        
        for elem in 0..<batchSize {
        for headScore in 0..<_nbHeadsScore {
        let headValue = headScore / nbBlocksHead
        for seqQ in 0..<sequence {
        for j in 0..<size
        {
            let depthScore = j + headScore * size
            let depthValue = j + headValue * size
            
            var sum = 0.0
            for seqK in 0...seqQ
            {
                let valueTmp = value.get(seqK, depthValue)!.v[elem].out
                let scoreTmp = score
                    .get(seqQ, seqK + headScore * sequence)!.v[elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depthScore)!.v[elem].out = sum
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
        
        if cacheValue != nil && cacheSeq != nil
        {
            try _generateGPU()
        }
        else
        {
            _forwardGPU()
        }
    }
    
    /// Apply the generate pass in the GPU execution context.
    private func _generateGPU() throws
    {
        if sequence != 1
        {
            throw LayerError.Init(message: "`sequence` should be 1.")
        }
        
        _mergeCacheGPU()
        
        let value = layersPrev[0] as! LayerSeq
        let score = layersPrev[1] as! LayerSeq
        let nbNeuronsPrevValue = value.nbNeurons
        let nbNeuronsPrevScore = min(cacheSeq + 1, cacheSeqMax) * _nbHeadsScore
        
        let pNbHeadsValue: [UInt32] = [UInt32(_nbHeadsValue)]
        let pNbHeadsScore: [UInt32] = [UInt32(_nbHeadsScore)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbNeuronsPrevValue: [UInt32] = [UInt32(nbNeuronsPrevValue)]
        let pNbNeuronsPrevScore: [UInt32] = [UInt32(nbNeuronsPrevScore)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(min(cacheSeq + 1, cacheSeqMax))]
        
        let kernel = (nbNeurons / _nbHeadsScore) % 4 == 0 ?
            "valueCausalSeq4Generate" : "valueCausalSeqGenerate"
        let coeff = (nbNeurons / _nbHeadsScore) % 4 == 0 ? 4 : 1
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: deviceID
        )
        command.setBuffer(_getValueCacheOutputGPU()!.metal, atIndex: 0)
        command.setBuffer(score.outs.metal, atIndex: 1)
        command.setBytes(pNbHeadsValue, atIndex: 2)
        command.setBytes(pNbHeadsScore, atIndex: 3)
        command.setBytes(pNbNeurons, atIndex: 4)
        command.setBytes(pNbNeuronsPrevValue, atIndex: 5)
        command.setBytes(pNbNeuronsPrevScore, atIndex: 6)
        command.setBytes(pNbBatch, atIndex: 7)
        command.setBytes(pSequence, atIndex: 8)
        command.setBuffer(outs.metal, atIndex: 9)
        
        command.dispatchThreads(
            width: nbNeurons / coeff,
            height: batchSize
        )
        command.enqueue()
        
        cacheSeq += 1
    }
    
    /// Concatenate cache to key.
    private func _mergeCacheGPU()
    {
        let slidingWindow: Bool
        if cacheSeq >= cacheSeqMax
        {
            slidingWindow = true
        }
        else
        {
            slidingWindow = false
        }
        
        let value = layersPrev[0] as! LayerSeq
        let nbNeuronsPrevValue = value.nbNeurons
        let nbNeurons = nbNeuronsPrevValue
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(min(cacheSeq + 1, cacheSeqMax))]
        let pSequenceCache: [UInt32] = [UInt32(cacheSeq)]
        let pSequenceValue: [UInt32] = [UInt32(1)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        var globalOffset = 0
        let kernel = nbNeurons % 4 == 0 ?
            "concat1Seq4Forward" : "concat1SeqForward"
        let coeff = nbNeurons % 4 == 0 ? 4 : 1
        
        if batchSize != 1 && !slidingWindow
        {
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(_getValueCacheInputGPU()!.metal, atIndex: 0)
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pSequence, atIndex: 4)
            command.setBytes(pSequenceCache, atIndex: 5)
            command.setBuffer(_getValueCacheOutputGPU()!.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons / coeff,
                height: batchSize * cacheSeq
            )
            command.enqueue()
        }
        
        globalOffset += cacheSeq % cacheSeqMax
        
        let pGlobalOffset = [UInt32(globalOffset)]
        
        command = metalKernel.createCommand(
            kernel, deviceID: deviceID
        )
        command.setBuffer(value.outs.metal, atIndex: 0)
        command.setBytes(pGlobalOffset, atIndex: 1)
        command.setBytes(pNbNeurons, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pSequence, atIndex: 4)
        command.setBytes(pSequenceValue, atIndex: 5)
        command.setBuffer(_getValueCacheOutputGPU()!.metal, atIndex: 6)
        
        command.dispatchThreads(
            width: nbNeurons / coeff,
            height: batchSize * 1
        )
        command.enqueue()
    }
    
    ///
    /// Get value cache buffer to use as input in Metal kernel.
    ///
    /// - Returns: value cache to use as input.
    ///
    private func _getValueCacheInputGPU() -> FloatBuffer?
    {
        if cacheSeq != nil
        {
            if cacheSeq % 2 == 0
            {
                return _cacheValueTmp
            }
            else
            {
                return cacheValue
            }
        }
        return nil
    }
    
    ///
    /// Get value cache buffer to use as input in Metal kernel.
    ///
    /// - Returns: value cache to use as input.
    ///
    private func _getValueCacheOutputGPU() -> FloatBuffer?
    {
        if cacheSeq != nil
        {
            if batchSize == 1
            {
                return cacheValue
            }
            else
            {
                if cacheSeq >= cacheSeqMax  // sliding window
                {
                    // The cache key has not changed.
                    if (cacheSeqMax - 1) % 2 == 0
                    {
                        return cacheValue
                    }
                    else
                    {
                        return _cacheValueTmp
                    }
                }
                else if cacheSeq % 2 == 0
                {
                    return cacheValue
                }
                else
                {
                    return _cacheValueTmp
                }
            }
        }
        return nil
    }
    
    /// Apply the forward pass in the GPU execution context.
    private func _forwardGPU()
    {
        let value = layersPrev[0] as! LayerSeq
        let score = layersPrev[1] as! LayerSeq
        let nbNeuronsPrevValue = value.nbNeurons
        let nbNeuronsPrevScore = score.nbNeurons
        
        let pNbHeadsValue: [UInt32] = [UInt32(_nbHeadsValue)]
        let pNbHeadsScore: [UInt32] = [UInt32(_nbHeadsScore)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbNeuronsPrevValue: [UInt32] = [UInt32(nbNeuronsPrevValue)]
        let pNbNeuronsPrevScore: [UInt32] = [UInt32(nbNeuronsPrevScore)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let kernel = (nbNeurons / _nbHeadsScore) % 4 == 0 ?
            "valueCausalSeq4Forward" : "valueCausalSeqForward"
        let coeff = (nbNeurons / _nbHeadsScore) % 4 == 0 ? 4 : 1
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: deviceID
        )
        command.setBuffer(value.outs.metal, atIndex: 0)
        command.setBuffer(score.outs.metal, atIndex: 1)
        command.setBytes(pNbHeadsValue, atIndex: 2)
        command.setBytes(pNbHeadsScore, atIndex: 3)
        command.setBytes(pNbNeurons, atIndex: 4)
        command.setBytes(pNbNeuronsPrevValue, atIndex: 5)
        command.setBytes(pNbNeuronsPrevScore, atIndex: 6)
        command.setBytes(pNbBatch, atIndex: 7)
        command.setBytes(pSequence, atIndex: 8)
        command.setBuffer(outs.metal, atIndex: 9)
        
        command.dispatchThreads(
            width: nbNeurons / coeff,
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
        
        let value = (layersPrev[0] as! LayerSeq).neurons!
        let score = (layersPrev[1] as! LayerSeq).neurons!
        
        let size = (layersPrev[0] as! LayerSeq).nbNeurons / _nbHeadsValue
        let nbBlocksHead = _nbHeadsScore / _nbHeadsValue
        
        if layersPrev[0].computeDelta
        {
            for elem in 0..<batchSize {
            for headValue in 0..<_nbHeadsValue {
            for seqK in 0..<sequence {
            for j in 0..<size
            {
                let depthValue = j + headValue * size
                
                var sum = 0.0
                for blockHead in 0..<nbBlocksHead
                {
                    let headScore = blockHead + nbBlocksHead * headValue
                    let depthScore = j + headScore * size
                    
                    for seqQ in seqK..<sequence
                    {
                        let deltaCur = neurons.get(
                            seqQ, depthScore
                        )!.v[elem].delta
                        let scoreTmp = score.get(
                            seqQ, seqK + headScore * sequence
                        )!.v[elem].out
                        
                        sum += deltaCur * scoreTmp
                    }
                }
                
                if layersPrev[0].dirty
                {
                    value.get(seqK, depthValue)!.v[elem].delta = sum
                }
                else
                {
                    value.get(seqK, depthValue)!.v[elem].delta += sum
                }
            }}}}
        }
        if layersPrev[1].computeDelta
        {
            for elem in 0..<batchSize {
            for headScore in 0..<_nbHeadsScore {
            let headValue = headScore / nbBlocksHead
            for seqQ in 0..<sequence {
            for seqK in 0...seqQ
            {
                var sum = 0.0
                for j in 0..<size
                {
                    let depthScore = j + headScore * size
                    let depthValue = j + headValue * size
                    
                    let deltaCur = neurons.get(seqQ, depthScore)!.v[elem].delta
                    let valueTmp = value.get(seqK, depthValue)!.v[elem].out
                    
                    sum += deltaCur * valueTmp
                }
                
                if layersPrev[1].dirty
                {
                    score.get(seqQ, seqK + headScore * sequence)!
                        .v[elem].delta = sum
                }
                else
                {
                    score.get(seqQ, seqK + headScore * sequence)!
                        .v[elem].delta += sum
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
        
        let value = layersPrev[0] as! LayerSeq
        let score = layersPrev[1] as! LayerSeq
        let nbNeuronsPrevValue = value.nbNeurons
        let nbNeuronsPrevScore = score.nbNeurons
        
        let pNbHeadsValue: [UInt32] = [UInt32(_nbHeadsValue)]
        let pNbHeadsScore: [UInt32] = [UInt32(_nbHeadsScore)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbNeuronsPrevValue: [UInt32] = [UInt32(nbNeuronsPrevValue)]
        let pNbNeuronsPrevScore: [UInt32] = [UInt32(nbNeuronsPrevScore)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        if value.computeDelta
        {
            try value.checkStateBackwardGPU(batchSize: batchSize)
            
            let pDirty: [UInt32] = value.dirty ? [1] : [0]
            
            let kernel = (nbNeuronsPrevValue / _nbHeadsValue) % 4 == 0 ?
                "valueCausalValueSeq4Backward" : "valueCausalValueSeqBackward"
            let coeff = (nbNeuronsPrevValue / _nbHeadsValue) % 4 == 0 ? 4 : 1
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(score.outs.metal, atIndex: 1)
            command.setBytes(pNbHeadsValue, atIndex: 2)
            command.setBytes(pNbHeadsScore, atIndex: 3)
            command.setBytes(pNbNeurons, atIndex: 4)
            command.setBytes(pNbNeuronsPrevValue, atIndex: 5)
            command.setBytes(pNbNeuronsPrevScore, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pSequence, atIndex: 8)
            command.setBytes(pDirty, atIndex: 9)
            command.setBuffer(value.delta.metal, atIndex: 10)
            
            command.dispatchThreads(
                width: nbNeuronsPrevValue / coeff,
                height: batchSize * sequence
            )
            command.enqueue()
        }
        if score.computeDelta
        {
            try score.checkStateBackwardGPU(batchSize: batchSize)
            
            let pDirty: [UInt32] = score.dirty ? [1] : [0]
            
            let kernel = (nbNeurons / _nbHeadsScore) % 4 == 0 ?
                "valueCausalScoreSeq4Backward" : "valueCausalScoreSeqBackward"
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(value.outs.metal, atIndex: 1)
            command.setBytes(pNbHeadsValue, atIndex: 2)
            command.setBytes(pNbHeadsScore, atIndex: 3)
            command.setBytes(pNbNeurons, atIndex: 4)
            command.setBytes(pNbNeuronsPrevValue, atIndex: 5)
            command.setBytes(pNbNeuronsPrevScore, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pSequence, atIndex: 8)
            command.setBytes(pDirty, atIndex: 9)
            command.setBuffer(score.delta.metal, atIndex: 10)
            
            command.dispatchThreads(
                width: nbNeuronsPrevScore,
                height: batchSize * sequence
            )
            command.enqueue()
        }
        propagateDirty()
    }
}
