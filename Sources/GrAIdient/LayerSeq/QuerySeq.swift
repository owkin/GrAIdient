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
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - query: Previous layer containing the query to look for.
    ///     - key: Previous layer contianing the keys.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(query: LayerSeq, key: LayerSeq, params: GrAI.Model.Params)
    {
        if query.nbNeurons != key.nbNeurons ||
           query.sequence != key.sequence
        {
            fatalError("Layer structure error.")
        }

        super.init(layersPrev: [query, key],
                   sequence: query.sequence,
                   nbNeurons: query.sequence,
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
        
        let layer = QuerySeq(
            query: layersPrev[0], key: layersPrev[1], params: params
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
        let nbNeuronsPrev = (_layersPrev[0] as! LayerSeq).nbNeurons
        
        for batch in 0..<batchSize {
        for seqQ in 0..<sequence {
        for seqK in 0..<nbNeurons {
        for elem in 0..<nbSameElems
        {
            var sum = 0.0
            for depthPrev in 0..<nbNeuronsPrev
            {
                let queryTmp = query.get(seqQ, depthPrev)!.gc[batch][elem].out
                let keyTmp = key.get(seqK, depthPrev)!.gc[batch][elem].out
                
                sum += queryTmp * keyTmp
            }
            
            neurons.get(seqQ, seqK)!.gc[batch][elem].out =
                sum / sqrt(Double(nbNeuronsPrev))
        }}}}
        
        for batch in 0..<batchSize {
        for seqQ in 0..<sequence {
        for seqK in 0..<nbNeurons {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp
        {
            var sum = 0.0
            for depthPrev in 0..<nbNeuronsPrev
            {
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
            
            neurons.get(seqQ, seqK)!.gc[batch][offset+elem].out =
                sum / sqrt(Double(nbNeuronsPrev))
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
        
        for seqQ in 0..<sequence {
        for seqK in 0..<nbNeurons
        {
            neurons.get(seqQ, seqK)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }}
        
        let query = (_layersPrev[0] as! LayerSeq).neurons!
        let key = (_layersPrev[1] as! LayerSeq).neurons!
        let nbNeuronsPrev = (_layersPrev[0] as! LayerSeq).nbNeurons
        
        for batch in 0..<batchSize {
        for seqQ in 0..<sequence {
        for seqK in 0..<nbNeurons {
        for elem in 0..<nbSameElems
        {
            var sum = 0.0
            for depthPrev in 0..<nbNeuronsPrev
            {
                let queryTmp = query.get(seqQ, depthPrev)!.gc[batch][elem].out
                let keyTmp = key.get(seqK, depthPrev)!.gc[batch][elem].out
                
                sum += queryTmp * keyTmp
            }
            
            neurons.get(seqQ, seqK)!.gc[batch][elem].out =
                sum / sqrt(Double(nbNeuronsPrev))
        }}}}
        
        let queryBuffer =
            (_layersPrev[0] as! LayerSeq).outs.shared.buffer
        let keyBuffer =
            (_layersPrev[1] as! LayerSeq).outs.shared.buffer
        
        for batch in 0..<batchSize {
        for seqQ in 0..<sequence {
        for seqK in 0..<nbNeurons {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp
        {
            var sum = 0.0
            for depthPrev in 0..<nbNeuronsPrev
            {
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
            
            neurons.get(seqQ, seqK)!.gc[batch][offset+elem].out =
                sum / sqrt(Double(nbNeuronsPrev))
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
        
        let query = (_layersPrev[0] as! LayerSeq).neurons!
        let key = (_layersPrev[1] as! LayerSeq).neurons!
        let nbNeuronsPrev = (_layersPrev[0] as! LayerSeq).nbNeurons
        
        for elem in 0..<batchSize {
        for seqQ in 0..<sequence {
        for seqK in 0..<nbNeurons
        {
            var sum = 0.0
            for depthPrev in 0..<nbNeuronsPrev
            {
                let queryTmp = query.get(seqQ, depthPrev)!.v[elem].out
                let keyTmp = key.get(seqK, depthPrev)!.v[elem].out
                
                sum += queryTmp * keyTmp
            }
            
            neurons.get(seqQ, seqK)!.v[elem].out =
                sum / sqrt(Double(nbNeuronsPrev))
        }}}
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    /*public override func forwardGPU() throws
    {
        try checkStateForwardGPU(batchSize: batchSize)
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let sequencePrev = layerPrev.sequence
            
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            let pSequencePrev: [UInt32] = [UInt32(sequencePrev)]
            
            command = metalKernel.createCommand(
                "concat1SeqForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pSequence, atIndex: 4)
            command.setBytes(pSequencePrev, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * sequencePrev
            )
            command.enqueue()
            
            globalOffset += sequencePrev
        }
    }*/
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if !mustComputeBackward
        {
            return
        }
        
        let query = (_layersPrev[0] as! LayerSeq).neurons!
        let key = (_layersPrev[1] as! LayerSeq).neurons!
        let nbNeuronsPrev = (_layersPrev[0] as! LayerSeq).nbNeurons
        
        if _layersPrev[0].computeDelta
        {
            for elem in 0..<batchSize {
            for seqQ in 0..<sequence {
            for depthPrev in 0..<nbNeuronsPrev
            {
                var sum = 0.0
                for seqK in 0..<nbNeurons
                {
                    let deltaCur = neurons.get(seqQ, seqK)!.v[elem].delta
                    let keyTmp = key.get(seqK, depthPrev)!.v[elem].out
                    
                    sum += deltaCur * keyTmp
                }
                
                if _layersPrev[0].dirty
                {
                    query.get(seqQ, depthPrev)!.v[elem].delta =
                        sum / sqrt(Double(nbNeuronsPrev))
                }
                else
                {
                    query.get(seqQ, depthPrev)!.v[elem].delta +=
                        sum / sqrt(Double(nbNeuronsPrev))
                }
            }}}
        }
        if _layersPrev[1].computeDelta
        {
            for elem in 0..<batchSize {
            for seqK in 0..<nbNeurons {
            for depthPrev in 0..<nbNeuronsPrev
            {
                var sum = 0.0
                for seqQ in 0..<sequence
                {
                    let deltaCur = neurons.get(seqQ, seqK)!.v[elem].delta
                    let queryTmp = query.get(seqQ, depthPrev)!.v[elem].out
                    
                    sum += deltaCur * queryTmp
                }
                
                if _layersPrev[0].dirty
                {
                    key.get(seqK, depthPrev)!.v[elem].delta =
                        sum / sqrt(Double(nbNeuronsPrev))
                }
                else
                {
                    key.get(seqK, depthPrev)!.v[elem].delta +=
                        sum / sqrt(Double(nbNeuronsPrev))
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
    /*public override func backwardGPU() throws
    {
        if !mustComputeBackward
        {
            return
        }
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let sequencePrev = layerPrev.sequence
            
            if !_layersPrev[num].computeDelta
            {
                globalOffset += sequencePrev
                continue
            }
            
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            let pSequencePrev: [UInt32] = [UInt32(sequencePrev)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            command = metalKernel.createCommand(
                "concat1SeqBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pSequence, atIndex: 4)
            command.setBytes(pSequencePrev, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * sequencePrev
            )
            command.enqueue()
            
            globalOffset += sequencePrev
        }
        propagateDirty()
    }*/
}
