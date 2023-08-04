//
// SumSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 05/03/2023.
//

///
/// Layer with a sequential shape neural structure.
///
/// This layer merges multiple sequential layers, summing the neurons together.
///
public class SumSeq: LayerMergeSeq
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
        for idPrev in _idsPrev
        {
            layersPrev.append(mapping[idPrev] as! LayerSeq)
        }
        
        let layer = try! SumSeq(layersPrev: layersPrev, params: params)
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
        for depth in 0..<nbNeurons
        {
            var sum = 0.0
            for num in 0..<_layersPrev.count
            {
                let neuronsPrev = (_layersPrev[num] as! LayerSeq).neurons!
                sum += neuronsPrev.get(seq, depth)!.gc[batch][elem].out
            }
            neurons.get(seq, depth)!.gc[batch][elem].out = sum
        }}}}
    
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        for depth in 0..<nbNeurons
        {
            var sum = 0.0
            for num in 0..<_layersPrev.count
            {
                let neuronsPrev = (_layersPrev[num] as! LayerSeq).neurons!
                
                if num == index
                {
                    sum += neuronsPrev.get(seq, depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    sum += neuronsPrev.get(seq, depth)!.v[batch].out
                }
            }
            
            neurons.get(seq, depth)!.gc[batch][offset+elem].out = sum
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
        
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        for elem in 0..<nbSameElems {
        for depth in 0..<nbNeurons
        {
            var sum = 0.0
            for num in 0..<_layersPrev.count
            {
                let neuronsPrev = (_layersPrev[num] as! LayerSeq).neurons!
                sum += neuronsPrev.get(seq, depth)!.gc[batch][elem].out
            }
            neurons.get(seq, depth)!.gc[batch][elem].out = sum
        }}}}
        
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        for depth in 0..<nbNeurons
        {
            var sum = 0.0
            for num in 0..<_layersPrev.count
            {
                let outsPrevPtr =
                    (_layersPrev[num] as! LayerSeq).outs.shared.buffer
                let neuronsPrev =
                    (_layersPrev[num] as! LayerSeq).neurons!
                
                if num == index
                {
                    sum += neuronsPrev.get(seq, depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    let offsetTmp = depth + nbNeurons * seq +
                        sequence * nbNeurons * batch
                    sum += Double(outsPrevPtr[offsetTmp])
                }
            }
            
            neurons.get(seq, depth)!.gc[batch][offset+elem].out = sum
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
            var sum = 0.0
            for num in 0..<_layersPrev.count
            {
                let neuronsPrev = (_layersPrev[num] as! LayerSeq).neurons!
                sum += neuronsPrev.get(seq, depth)!.v[elem].out
            }
            neurons.get(seq, depth)!.v[elem].out = sum
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
        
        var first = true
        for num in 0..<_layersPrev.count
        {
            let nbElems = (_layersPrev[num] as! LayerSeq).outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command: MetalCommand
            if first
            {
                command = MetalKernel.get.createCommand(
                    "sum1", deviceID: deviceID
                )
                first = false
            }
            else
            {
                command = MetalKernel.get.createCommand(
                    "sum2", deviceID: deviceID
                )
            }
            
            command.setBuffer(
                (_layersPrev[num] as! LayerSeq).outs.metal, atIndex: 0
            )
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if !mustComputeBackward
        {
            return
        }
        
        for num in 0..<_layersPrev.count
        {
            if !_layersPrev[num].computeDelta
            {
                continue
            }
            
            let neuronsPrev = (_layersPrev[num] as! LayerSeq).neurons!
            for elem in 0..<batchSize {
            for seq in 0..<sequence {
            for depth in 0..<nbNeurons
            {
                let deltaCur = neurons.get(seq, depth)!.v[elem].delta
                
                if _layersPrev[num].dirty
                {
                    neuronsPrev.get(seq, depth)!.v[elem].delta = deltaCur
                }
                else
                {
                    neuronsPrev.get(seq, depth)!.v[elem].delta += deltaCur
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
        
        for num in 0..<_layersPrev.count
        {
            if !_layersPrev[num].computeDelta
            {
                continue
            }
            
            try (_layersPrev[num] as! LayerSeq).checkStateBackwardGPU(
                batchSize: batchSize
            )
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command: MetalCommand
            if _layersPrev[num].dirty
            {
                command = MetalKernel.get.createCommand(
                    "sum1", deviceID: deviceID
                )
            }
            else
            {
                command = MetalKernel.get.createCommand(
                    "sum2", deviceID: deviceID
                )
            }
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(
                (_layersPrev[num] as! LayerSeq).delta.metal, atIndex: 2
            )
            
            command.dispatchThreads(nbElems)
            command.enqueue()
        }
        propagateDirty()
    }
}
