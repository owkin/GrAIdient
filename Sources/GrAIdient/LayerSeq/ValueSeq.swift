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
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - value: Previous layer containing the value.
    ///     - score: Previous layer contianing the attention scores per sequence.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(value: LayerSeq, score: LayerSeq, params: GrAI.Model.Params)
    {
        if value.sequence != score.sequence ||
           score.sequence != score.nbNeurons
        {
            fatalError("Layer structure error.")
        }

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
        try super.init(from: decoder)
    }
    
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
        
        let layer = ValueSeq(
            value: layersPrev[0], score: layersPrev[1], params: params
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
        
        let value = (_layersPrev[0] as! LayerSeq).neurons!
        let score = (_layersPrev[1] as! LayerSeq).neurons!
        
        for batch in 0..<batchSize {
        for seqQ in 0..<sequence {
        for depth in 0..<nbNeurons {
        for elem in 0..<nbSameElems
        {
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp = value.get(seqK, depth)!.gc[batch][elem].out
                let scoreTmp = score.get(seqQ, seqK)!.gc[batch][elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][elem].out = sum
        }}}}
        
        for batch in 0..<batchSize {
        for seqQ in 0..<sequence {
        for depth in 0..<nbNeurons {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
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
                    scoreTmp = score.get(seqQ, seqK)!.v[batch].out
                }
                else
                {
                    valueTmp = value.get(seqK, depth)!.v[batch].out
                    scoreTmp = score.get(seqQ, seqK)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][offset+elem].out = sum
        }
        
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}}}
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
        
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }}
        
        let value = (_layersPrev[0] as! LayerSeq).neurons!
        let score = (_layersPrev[1] as! LayerSeq).neurons!
        
        for batch in 0..<batchSize {
        for seqQ in 0..<sequence {
        for depth in 0..<nbNeurons {
        for elem in 0..<nbSameElems
        {
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp = value.get(seqK, depth)!.gc[batch][elem].out
                let scoreTmp = score.get(seqQ, seqK)!.gc[batch][elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][elem].out = sum
        }}}}
        
        let valueBuffer =
            (_layersPrev[0] as! LayerSeq).outs.shared.buffer
        let scoreBuffer =
            (_layersPrev[1] as! LayerSeq).outs.shared.buffer
        
        for batch in 0..<batchSize {
        for seqQ in 0..<sequence {
        for depth in 0..<nbNeurons {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
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
                    
                    let offsetTmp = seqK + sequence * seqQ +
                        sequence * sequence * batch
                    
                    scoreTmp = Double(scoreBuffer[offsetTmp])
                }
                else
                {
                    let offsetTmp = depth + nbNeurons * seqK +
                        sequence * nbNeurons * batch
                    
                    valueTmp = Double(valueBuffer[offsetTmp])
                    
                    scoreTmp = score.get(seqQ, seqK)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.gc[batch][offset+elem].out = sum
        }
        
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}}}
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        let value = (_layersPrev[0] as! LayerSeq).neurons!
        let score = (_layersPrev[1] as! LayerSeq).neurons!
        
        for elem in 0..<batchSize {
        for seqQ in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            var sum = 0.0
            for seqK in 0..<sequence
            {
                let valueTmp = value.get(seqK, depth)!.v[elem].out
                let scoreTmp = score.get(seqQ, seqK)!.v[elem].out
                
                sum += valueTmp * scoreTmp
            }
            
            neurons.get(seqQ, depth)!.v[elem].out = sum
        }}}
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        try checkStateForwardGPU(batchSize: batchSize)
        
        let value = _layersPrev[0] as! LayerSeq
        let score = _layersPrev[1] as! LayerSeq
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let command = MetalKernel.get.createCommand(
            "valueSeqForward", deviceID: deviceID
        )
        command.setBuffer(value.outs.metal, atIndex: 0)
        command.setBuffer(score.outs.metal, atIndex: 1)
        command.setBytes(pNbNeurons, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pSequence, atIndex: 4)
        command.setBuffer(outs.metal, atIndex: 5)
        
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
        
        let value = (_layersPrev[0] as! LayerSeq).neurons!
        let score = (_layersPrev[1] as! LayerSeq).neurons!
        
        if _layersPrev[0].computeDelta
        {
            for elem in 0..<batchSize {
            for seqQ in 0..<sequence {
            for depth in 0..<nbNeurons
            {
                var sum = 0.0
                for seqK in 0..<sequence
                {
                    let deltaCur = neurons.get(seqQ, depth)!.v[elem].delta
                    let scoreTmp = score.get(seqK, seqQ)!.v[elem].out
                    
                    sum += deltaCur * scoreTmp
                }
                
                if _layersPrev[0].dirty
                {
                    value.get(seqQ, depth)!.v[elem].delta = sum
                }
                else
                {
                    value.get(seqQ, depth)!.v[elem].delta += sum
                }
            }}}
        }
        if _layersPrev[1].computeDelta
        {
            for elem in 0..<batchSize {
            for seqQ in 0..<sequence {
            for seqK in 0..<sequence
            {
                var sum = 0.0
                for depth in 0..<nbNeurons
                {
                    let deltaCur = neurons.get(seqQ, depth)!.v[elem].delta
                    let valueTmp = value.get(seqK, depth)!.v[elem].out
                    
                    sum += deltaCur * valueTmp
                }
                
                if _layersPrev[1].dirty
                {
                    score.get(seqQ, seqK)!.v[elem].delta = sum
                }
                else
                {
                    score.get(seqQ, seqK)!.v[elem].delta += sum
                }
            }}}
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
        
        let value = _layersPrev[0] as! LayerSeq
        let score = _layersPrev[1] as! LayerSeq
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        if value.computeDelta
        {
            try value.checkStateBackwardGPU(batchSize: batchSize)
            
            let pDirty: [UInt32] = value.dirty ? [1] : [0]
            
            command = metalKernel.createCommand(
                "valueValueSeqBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(score.outs.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pSequence, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(value.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * sequence
            )
            command.enqueue()
        }
        if score.computeDelta
        {
            try score.checkStateBackwardGPU(batchSize: batchSize)
            
            let pDirty: [UInt32] = score.dirty ? [1] : [0]
            
            command = metalKernel.createCommand(
                "valueScoreSeqBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(value.outs.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pSequence, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(score.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: sequence,
                height: batchSize * sequence
            )
            command.enqueue()
        }
        propagateDirty()
    }
}
