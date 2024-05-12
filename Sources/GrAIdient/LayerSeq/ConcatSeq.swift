//
// ConcatSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 05/03/2023.
//

///
/// Layer with a sequential shape neural structure.
///
/// This layer merges multiple sequential layers,
/// concatenating the neurons in the sequential dimension together.
///
public class Concat1Seq: LayerMergeSeq
{
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layersPrev: List of previous layers that have been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layersPrev: [LayerSeq], params: GrAI.Model.Params) throws
    {
        let layer0 = layersPrev[0]
        let nbNeurons = layer0.nbNeurons
        for layerPrev in layersPrev
        {
            if layerPrev.nbNeurons != nbNeurons
            {
                throw LayerError.Init(message: "Layer structure error.")
            }
        }
        
        var sequence = 0
        for layer in layersPrev
        {
            sequence += layer.sequence
        }
        super.init(layersPrev: layersPrev,
                   sequence: sequence,
                   nbNeurons: nbNeurons,
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
        
        let layer = try! Concat1Seq(layersPrev: layersPrev, params: params)
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
        
        for batch in 0..<batchSize {
        for depth in 0..<nbNeurons {
        for elem in 0..<nbSameElems {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let neuronsPrev = layerPrev.neurons!
            let sequence = layerPrev.sequence
            
            for seq in 0..<sequence
            {
                let outPrev = neuronsPrev.get(seq, depth)!.gc[batch][elem].out
                neurons.get(curElem+seq, depth)!.gc[batch][elem].out = outPrev
            }
            
            curElem += sequence
        }}}}
        
        for batch in 0..<batchSize {
        for depth in 0..<nbNeurons {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let neuronsPrev = layerPrev.neurons!
            let sequence = layerPrev.sequence
            
            for seq in 0..<sequence
            {
                let outPrev: Double
                if num == index
                {
                    outPrev = neuronsPrev.get(seq, depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    outPrev = neuronsPrev.get(seq, depth)!.v[batch].out
                }
                
                neurons.get(curElem+seq, depth)!.gc[batch][offset+elem].out =
                    outPrev
            }
            
            curElem += sequence
        }}
        
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}}
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        var buffersPrev = [[Float]]()
        for num in 0..<_layersPrev.count
        {
            buffersPrev.append((_layersPrev[num] as! LayerSeq).outs.download())
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
        
        for batch in 0..<batchSize {
        for depth in 0..<nbNeurons {
        for elem in 0..<nbSameElems {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let neuronsPrev = layerPrev.neurons!
            let sequence = layerPrev.sequence
            
            for seq in 0..<sequence
            {
                let outPrev = neuronsPrev.get(seq, depth)!.gc[batch][elem].out
                neurons.get(curElem+seq, depth)!.gc[batch][elem].out = outPrev
            }
            
            curElem += sequence
        }}}}
        
        for batch in 0..<batchSize {
        for depth in 0..<nbNeurons {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let outsPrevPtr = buffersPrev[num]
            let neuronsPrev = layerPrev.neurons!
            let sequence = layerPrev.sequence
            
            for seq in 0..<sequence
            {
                let outPrev: Double
                if num == index
                {
                    outPrev = neuronsPrev.get(seq, depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    let offsetTmp = depth + nbNeurons * seq +
                        sequence * nbNeurons * batch
                    outPrev = Double(outsPrevPtr[offsetTmp])
                }
                
                neurons.get(curElem+seq, depth)!.gc[batch][offset+elem].out =
                    outPrev
            }
            
            curElem += sequence
        }}
        
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}}
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        for elem in 0..<batchSize {
        for depth in 0..<nbNeurons
        {
            var curElem = 0
            for num in 0..<_layersPrev.count
            {
                let layerPrev = _layersPrev[num] as! LayerSeq
                let neuronsPrev = layerPrev.neurons!
                let sequence = layerPrev.sequence
                
                for seq in 0..<sequence
                {
                    let outPrev = neuronsPrev.get(seq, depth)!.v[elem].out
                    neurons.get(curElem+seq, depth)!.v[elem].out = outPrev
                }
                
                curElem += sequence
            }
        }}
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        try checkStateForwardGPU(batchSize: batchSize)
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let metalKernel = MetalKernel.get
        var kernel: String
        var coeff: Int
        var command: MetalCommand
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let sequencePrev = layerPrev.sequence
            
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            let pSequencePrev: [UInt32] = [UInt32(sequencePrev)]
            
            kernel = nbNeurons % 4 == 0 ?
                "concat1Seq4Forward" : "concat1SeqForward"
            coeff = nbNeurons % 4 == 0 ? 4 : 1
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pSequence, atIndex: 4)
            command.setBytes(pSequencePrev, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons / coeff,
                height: batchSize * sequencePrev
            )
            command.enqueue()
            
            globalOffset += sequencePrev
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if !mustComputeBackward
        {
            return
        }
        
        for elem in 0..<batchSize {
        for depth in 0..<nbNeurons
        {
            var curElem = 0
            for num in 0..<_layersPrev.count
            {
                let layerPrev = _layersPrev[num] as! LayerSeq
                let neuronsPrev = layerPrev.neurons!
                let sequence = layerPrev.sequence
                
                if !_layersPrev[num].computeDelta
                {
                    curElem += sequence
                    continue
                }
                
                for seq in 0..<sequence
                {
                    let deltaCur =
                        neurons.get(curElem+seq, depth)!.v[elem].delta
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(seq, depth)!.v[elem].delta = deltaCur
                    }
                    else
                    {
                        neuronsPrev.get(seq, depth)!.v[elem].delta += deltaCur
                    }
                }
                
                curElem += sequence
            }
        }}
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
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let metalKernel = MetalKernel.get
        var kernel: String
        var coeff: Int
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
            
            kernel = nbNeurons % 4 == 0 ?
                "concat1Seq4Backward" : "concat1SeqBackward"
            coeff = nbNeurons % 4 == 0 ? 4 : 1
            command = metalKernel.createCommand(
                kernel, deviceID: deviceID
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
                width: nbNeurons / coeff,
                height: batchSize * sequencePrev
            )
            command.enqueue()
            
            globalOffset += sequencePrev
        }
        propagateDirty()
    }
}

///
/// Layer with a sequential shape neural structure.
///
/// This layer merges multiple sequential layers,
/// concatenating the neurons in the "neural" dimension together.
///
public class Concat2Seq: LayerMergeSeq
{
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layersPrev: List of previous layers that have been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layersPrev: [LayerSeq], params: GrAI.Model.Params)
    {
        let layer0 = layersPrev[0]
        let sequence = layer0.sequence
        for layerPrev in layersPrev
        {
            if layerPrev.sequence != sequence
            {
                fatalError("Layer structure error.")
            }
        }
        
        var nbNeurons = 0
        for layer in layersPrev
        {
            nbNeurons += layer.nbNeurons
        }
        super.init(layersPrev: layersPrev,
                   sequence: sequence,
                   nbNeurons: nbNeurons,
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
        
        let layer = Concat2Seq(layersPrev: layersPrev, params: params)
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
        
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        for elem in 0..<nbSameElems {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let neuronsPrev = layerPrev.neurons!
            let nbNeurons = layerPrev.nbNeurons
            
            for depth in 0..<nbNeurons
            {
                let outPrev = neuronsPrev.get(seq, depth)!.gc[batch][elem].out
                neurons.get(seq, curElem+depth)!.gc[batch][elem].out = outPrev
            }
            
            curElem += nbNeurons
        }}}}
        
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let neuronsPrev = layerPrev.neurons!
            let nbNeurons = layerPrev.nbNeurons
            
            for depth in 0..<nbNeurons
            {
                let outPrev: Double
                if num == index
                {
                    outPrev = neuronsPrev.get(seq, depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    outPrev = neuronsPrev.get(seq, depth)!.v[batch].out
                }
                
                neurons.get(seq, curElem+depth)!.gc[batch][offset+elem].out =
                    outPrev
            }
            
            curElem += nbNeurons
        }}
        
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}}
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        var buffersPrev = [[Float]]()
        for num in 0..<_layersPrev.count
        {
            buffersPrev.append((_layersPrev[num] as! LayerSeq).outs.download())
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
        
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        for elem in 0..<nbSameElems {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let neuronsPrev = layerPrev.neurons!
            let nbNeurons = layerPrev.nbNeurons
            
            for depth in 0..<nbNeurons
            {
                let outPrev = neuronsPrev.get(seq, depth)!.gc[batch][elem].out
                neurons.get(seq, curElem+depth)!.gc[batch][elem].out = outPrev
            }
            
            curElem += nbNeurons
        }}}}
        
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let outsPrevPtr = buffersPrev[num]
            let neuronsPrev = layerPrev.neurons!
            let nbNeurons = layerPrev.nbNeurons
            
            for depth in 0..<nbNeurons
            {
                let outPrev: Double
                if num == index
                {
                    outPrev = neuronsPrev.get(seq, depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    let offsetTmp = depth + nbNeurons * seq +
                        sequence * nbNeurons * batch
                    outPrev = Double(outsPrevPtr[offsetTmp])
                }
                
                neurons.get(seq, curElem+depth)!.gc[batch][offset+elem].out =
                    outPrev
            }
            
            curElem += nbNeurons
        }}
        
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}}
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        for elem in 0..<batchSize {
        for seq in 0..<sequence
        {
            var curElem = 0
            for num in 0..<_layersPrev.count
            {
                let layerPrev = _layersPrev[num] as! LayerSeq
                let neuronsPrev = layerPrev.neurons!
                let nbNeurons = layerPrev.nbNeurons
                
                for depth in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(seq, depth)!.v[elem].out
                    neurons.get(seq, curElem+depth)!.v[elem].out = outPrev
                }
                
                curElem += nbNeurons
            }
        }}
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
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
            let nbNeuronsPrev = layerPrev.nbNeurons
            
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
            
            command = metalKernel.createCommand(
                "concat2SeqForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbNeuronsPrev, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeuronsPrev,
                height: batchSize * sequence
            )
            command.enqueue()
            
            globalOffset += nbNeuronsPrev
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if !mustComputeBackward
        {
            return
        }
        
        for elem in 0..<batchSize {
        for seq in 0..<sequence
        {
            var curElem = 0
            for num in 0..<_layersPrev.count
            {
                let layerPrev = _layersPrev[num] as! LayerSeq
                let neuronsPrev = layerPrev.neurons!
                let nbNeurons = layerPrev.nbNeurons
                
                if !_layersPrev[num].computeDelta
                {
                    curElem += nbNeurons
                    continue
                }
                
                for depth in 0..<nbNeurons
                {
                    let deltaCur =
                        neurons.get(seq, curElem+depth)!.v[elem].delta
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(seq, depth)!.v[elem].delta = deltaCur
                    }
                    else
                    {
                        neuronsPrev.get(seq, depth)!.v[elem].delta += deltaCur
                    }
                }
                
                curElem += nbNeurons
            }
        }}
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
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! LayerSeq
            let nbNeuronsPrev = layerPrev.nbNeurons
            
            if !_layersPrev[num].computeDelta
            {
                globalOffset += nbNeuronsPrev
                continue
            }
            
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            command = metalKernel.createCommand(
                "concat2SeqBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbNeuronsPrev, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: nbNeuronsPrev,
                height: batchSize * sequence
            )
            command.enqueue()
            
            globalOffset += nbNeuronsPrev
        }
        propagateDirty()
    }
}
