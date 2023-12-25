//
// Concat1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 31/12/2022.
//

///
/// Layer with a 1D shape neural structure.
///
/// This layer merges multiple 1D layers, concatenating the neurons together.
///
public class Concat1D: LayerMerge1D
{
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layersPrev: List of previous layers that have been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layersPrev: [Layer1D], params: GrAI.Model.Params)
    {
        var nbNeurons = 0
        for layer in layersPrev
        {
            nbNeurons += layer.nbNeurons
        }
        super.init(layersPrev: layersPrev,
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
        
        var layersPrev = [Layer1D]()
        for idPrev in _idsPrev
        {
            layersPrev.append(mapping[idPrev] as! Layer1D)
        }
        
        let layer = Concat1D(layersPrev: layersPrev, params: params)
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
        
        for depth in 0..<nbNeurons
        {
            neurons.get(depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }
        
        for batch in 0..<batchSize {
        for elem in 0..<nbSameElems {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let neuronsPrev = (_layersPrev[num] as! Layer1D).neurons
            let nbNeurons = neuronsPrev.nbElems
            
            for depth in 0..<nbNeurons
            {
                let outPrev = neuronsPrev.get(depth)!.gc[batch][elem].out
                neurons.get(curElem+depth)!.gc[batch][elem].out = outPrev
            }
            
            curElem += nbNeurons
        }}}
        
        for batch in 0..<batchSize {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let neuronsPrev = (_layersPrev[num] as! Layer1D).neurons
            let nbNeurons = neuronsPrev.nbElems
            
            for depth in 0..<nbNeurons
            {
                let outPrev: Double
                if num == index
                {
                    outPrev = neuronsPrev.get(depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    outPrev = neuronsPrev.get(depth)!.v[batch].out
                }
                 
                neurons.get(curElem+depth)!.gc[batch][offset+elem].out = outPrev
            }
            
            curElem += nbNeurons
        }}
            
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}
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
            MetalKernel.get.download([(_layersPrev[num] as! Layer1D).outs])
        }
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for depth in 0..<nbNeurons
        {
            neurons.get(depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }
        
        for batch in 0..<batchSize {
        for elem in 0..<nbSameElems {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let neuronsPrev = (_layersPrev[num] as! Layer1D).neurons
            let nbNeurons = neuronsPrev.nbElems
            
            for depth in 0..<nbNeurons
            {
                let outPrev = neuronsPrev.get(depth)!.gc[batch][elem].out
                neurons.get(curElem+depth)!.gc[batch][elem].out = outPrev
            }
            
            curElem += nbNeurons
        }}}
        
        for batch in 0..<batchSize {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let outsPrevPtr = (_layersPrev[num] as! Layer1D).outs.shared.buffer
            let neuronsPrev = (_layersPrev[num] as! Layer1D).neurons
            let nbNeurons = neuronsPrev.nbElems
            
            for depth in 0..<nbNeurons
            {
                let outPrev: Double
                if num == index
                {
                    outPrev = neuronsPrev.get(depth)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    let offsetTmp = depth + nbNeurons * batch
                    outPrev = Double(outsPrevPtr[offsetTmp])
                }
                 
                neurons.get(curElem+depth)!.gc[batch][offset+elem].out = outPrev
            }
            
            curElem += nbNeurons
        }}
            
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        for elem in 0..<batchSize
        {
            var curElem = 0
            for num in 0..<_layersPrev.count
            {
                let neuronsPrev = (_layersPrev[num] as! Layer1D).neurons
                let nbNeurons = neuronsPrev.nbElems
                
                for depth in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(depth)!.v[elem].out
                    neurons.get(curElem+depth)!.v[elem].out = outPrev
                }
                
                curElem += nbNeurons
            }
        }
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
        
        let metalKernel = MetalKernel.get
        var command: MetalEncoder
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let nbNeuronsPrev = (_layersPrev[num] as! Layer1D).nbNeurons
            
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            let pNbNeuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
            
            command = metalKernel.createEncoder(
                "concat1DForward", deviceID: deviceID
            )
            command.setBuffer(
                (_layersPrev[num] as! Layer1D).outs.metal, atIndex: 0
            )
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbNeuronsPrev, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBuffer(outs.metal, atIndex: 5)
            
            command.dispatchThreads(
                width: nbNeuronsPrev,
                height: batchSize
            )
            command.endEncoding()
            
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
        
        for elem in 0..<batchSize
        {
            var curElem = 0
            for num in 0..<_layersPrev.count
            {
                let layerPrev = _layersPrev[num] as! Layer1D
                let neuronsPrev = layerPrev.neurons
                let nbNeurons = layerPrev.nbNeurons
                
                if !_layersPrev[num].computeDelta
                {
                    curElem += nbNeurons
                    continue
                }
                
                for depth in 0..<nbNeurons
                {
                    let deltaCur = neurons.get(curElem+depth)!.v[elem].delta
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(depth)!.v[elem].delta = deltaCur
                    }
                    else
                    {
                        neuronsPrev.get(depth)!.v[elem].delta += deltaCur
                    }
                }
                
                curElem += nbNeurons
            }
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
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let metalKernel = MetalKernel.get
        var command: MetalEncoder
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! Layer1D
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
            
            command = metalKernel.createEncoder(
                "concat1DBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbNeuronsPrev, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeuronsPrev,
                height: batchSize
            )
            command.endEncoding()
            
            globalOffset += nbNeuronsPrev
        }
        propagateDirty()
    }
}
