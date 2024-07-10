//
// MutiplySeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 01/07/2024.
//

///
/// Layer with a sequential shape neural structure.
///
/// This layer merges multiple sequential layers, multiplying the neurons together.
///
public class MultiplySeq: LayerMergeSeq
{
    ///
    /// List of output buffers for CPU usage.
    /// Shape ~ (batch, sequence, nbNeurons).
    ///
    var _otherOuts1: [[Double]] = []
    ///
    /// List of output buffers for GPU usage.
    /// Shape ~ (batch, sequence, nbNeurons).
    ///
    var _otherOuts2: [FloatBuffer] = []
    
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
        let sequence = layer0.sequence
        let nbNeurons = layer0.nbNeurons
        for layerPrev in layersPrev
        {
            if layerPrev.nbNeurons != nbNeurons ||
               layerPrev.sequence != sequence
            {
                throw LayerError.Init(message: "Layer structure error.")
            }
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
        
        let layer = try! MultiplySeq(layersPrev: layersPrev, params: params)
        return layer
    }
    
    ///
    /// Clean state resources in the CPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        _otherOuts1 = []
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _otherOuts2 = []
    }
    
    ///
    /// Initialize state resources in the CPU execution context.
    ///
    /// We initialize the neurons' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        try super.checkStateCPU(batchSize: batchSize)
        
        if phase != nil && (phase == .Training || phase == .InferenceBackward) {
        if _otherOuts1.count == 0
        {
            for _ in 0..<layersPrev.count
            {
                _otherOuts1.append([Double](
                    repeating: 0.0,
                    count: batchSize * sequence * nbNeurons
                ))
            }
        }}
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if phase != nil && (phase == .Training || phase == .InferenceBackward) {
        if _otherOuts2.count == 0
        {
            for _ in 0..<layersPrev.count
            {
                let buffer = FloatBuffer(
                    nbElems: batchSize * sequence * nbNeurons,
                    deviceID: deviceID
                )
                _otherOuts2.append(buffer)
            }
        }}
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
        for depth in 0..<nbNeurons
        {
            var value = 1.0
            for num in 0..<layersPrev.count
            {
                let neuronsPrev = (layersPrev[num] as! LayerSeq).neurons!
                value *= neuronsPrev.get(seq, depth)!.gc[batch][elem].out
            }
            neurons.get(seq, depth)!.gc[batch][elem].out = value
        }}}}
    
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        for depth in 0..<nbNeurons
        {
            var value = 1.0
            for num in 0..<layersPrev.count
            {
                let neuronsPrev = (layersPrev[num] as! LayerSeq).neurons!
                
                if num == index
                {
                    value *= neuronsPrev.get(seq, depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    value *= neuronsPrev.get(seq, depth)!.v[batch].out
                }
            }
            
            neurons.get(seq, depth)!.gc[batch][offset+elem].out = value
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
        for num in 0..<layersPrev.count
        {
            buffersPrev.append((layersPrev[num] as! LayerSeq).outs.download())
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
        for depth in 0..<nbNeurons
        {
            var value = 1.0
            for num in 0..<layersPrev.count
            {
                let neuronsPrev = (layersPrev[num] as! LayerSeq).neurons!
                value *= neuronsPrev.get(seq, depth)!.gc[batch][elem].out
            }
            neurons.get(seq, depth)!.gc[batch][elem].out = value
        }}}}
        
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        for depth in 0..<nbNeurons
        {
            var value = 1.0
            for num in 0..<layersPrev.count
            {
                let outsPrevPtr = buffersPrev[num]
                let neuronsPrev =
                    (layersPrev[num] as! LayerSeq).neurons!
                
                if num == index
                {
                    value *= neuronsPrev.get(seq, depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    let offsetTmp = depth + nbNeurons * seq +
                        sequence * nbNeurons * batch
                    value *= Double(outsPrevPtr[offsetTmp])
                }
            }
            
            neurons.get(seq, depth)!.gc[batch][offset+elem].out = value
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
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            let offset = depth + nbNeurons * seq +
                sequence * nbNeurons * elem
            
            var value = 1.0
            for num in 0..<layersPrev.count
            {
                let neuronsPrev = (layersPrev[num] as! LayerSeq).neurons!
                value *= neuronsPrev.get(seq, depth)!.v[elem].out
            }
            neurons.get(seq, depth)!.v[elem].out = value
            
            if phase != nil &&
               (phase == .Training || phase == .InferenceBackward) {
            for num1 in 0..<layersPrev.count
            {
                value = 1.0
                for num2 in 0..<layersPrev.count {
                if num2 != num1
                {
                    let neuronsPrev = (layersPrev[num2] as! LayerSeq).neurons!
                    value *= neuronsPrev.get(seq, depth)!.v[elem].out
                }}
                _otherOuts1[num1][offset] = value
            }}
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
        
        var first1 = true
        for num1 in 0..<layersPrev.count
        {
            let nbElems = (layersPrev[num1] as! LayerSeq).outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            var command: MetalCommand
            if first1
            {
                command = MetalKernel.get.createCommand(
                    "sum1", deviceID: deviceID
                )
                first1 = false
            }
            else
            {
                command = MetalKernel.get.createCommand(
                    "multiplyForward", deviceID: deviceID
                )
            }
            
            command.setBuffer(
                (layersPrev[num1] as! LayerSeq).outs.metal, atIndex: 0
            )
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
            
            if phase != nil &&
               (phase == .Training || phase == .InferenceBackward) {
            var first2 = true
            for num2 in 0..<layersPrev.count {
            if num2 != num1
            {
                if first2
                {
                    command = MetalKernel.get.createCommand(
                        "sum1", deviceID: deviceID
                    )
                    first2 = false
                }
                else
                {
                    command = MetalKernel.get.createCommand(
                        "multiplyForward", deviceID: deviceID
                    )
                }
                
                command.setBuffer(
                    (layersPrev[num2] as! LayerSeq).outs.metal, atIndex: 0
                )
                command.setBytes(pNbElems, atIndex: 1)
                command.setBuffer(_otherOuts2[num1].metal, atIndex: 2)
                
                command.dispatchThreads(nbElems)
                command.enqueue()
            }}}
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if !mustComputeBackward
        {
            return
        }
        
        for num in 0..<layersPrev.count
        {
            if !layersPrev[num].computeDelta
            {
                continue
            }
            
            let neuronsPrev = (layersPrev[num] as! LayerSeq).neurons!
            let buffer = _otherOuts1[num]
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence {
            for depth in 0..<nbNeurons
            {
                let offset = depth + nbNeurons * seq +
                    sequence * nbNeurons * elem
                
                let tmp = Double(buffer[offset])
                let deltaCur = neurons.get(seq, depth)!.v[elem].delta
                
                if layersPrev[num].dirty
                {
                    neuronsPrev.get(seq, depth)!.v[elem].delta = deltaCur * tmp
                }
                else
                {
                    neuronsPrev.get(seq, depth)!.v[elem].delta += deltaCur * tmp
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
        
        for num in 0..<layersPrev.count
        {
            if !layersPrev[num].computeDelta
            {
                continue
            }
            let layerPrev = layersPrev[num] as! LayerSeq
            
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "multiplyBackward", deviceID: deviceID
            )
            command.setBuffer(_otherOuts2[num].metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbElems, atIndex: 2)
            command.setBytes(pDirty, atIndex: 3)
            command.setBuffer(layerPrev.delta.metal, atIndex: 4)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
        }
        propagateDirty()
    }
}
