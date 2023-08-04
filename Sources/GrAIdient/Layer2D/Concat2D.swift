//
// Concat2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 01/01/2023.
//

///
/// Layer with a 2D shape neural structure.
///
/// This layer merges multiple 2D layers, concatenating the neurons at the same localization
/// in the input grids together.
///
public class Concat2D: LayerMerge2D
{
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layersPrev: List of previous layers that have been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layersPrev: [Layer2D], params: GrAI.Model.Params) throws
    {
        var nbChannels = 0
        let layer0 = layersPrev[0]
        for layerPrev in layersPrev
        {
            nbChannels += layerPrev.nbChannels
            
            if layerPrev.height != layer0.height ||
               layerPrev.width != layer0.width
            {
                throw LayerError.Init(message: "Layer structure error.")
            }
        }
        super.init(layersPrev: layersPrev,
                   nbChannels: nbChannels,
                   height: layer0.height,
                   width: layer0.width,
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
        
        var layersPrev = [Layer2D]()
        for idPrev in _idsPrev
        {
            layersPrev.append(mapping[idPrev] as! Layer2D)
        }
        
        let layer = try! Concat2D(layersPrev: layersPrev, params: params)
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
        
        for depth in 0..<nbChannels
        {
            for i in 0..<height {
            for j in 0..<width
            {
                neurons[depth].get(i, j)!.initGC(
                    batchSize: batchSize,
                    nbGC: nbGC
                )
            }}
        }
        
        for batch in 0..<batchSize {
        for elem in 0..<nbSameElems {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            let nbChannels = neuronsPrev.count
            
            for depth in 0..<nbChannels {
            for i in 0..<height {
            for j in 0..<width
            {
                let outPrev = neuronsPrev[depth].get(i, j)!.gc[batch][elem].out
                neurons[curElem+depth].get(i, j)!.gc[batch][elem].out = outPrev
            }}}
            
            curElem += nbChannels
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
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            let nbChannels = neuronsPrev.count
            
            for depth in 0..<nbChannels {
            for i in 0..<height {
            for j in 0..<width
            {
                let outPrev: Double
                if num == index
                {
                    outPrev = neuronsPrev[depth].get(i, j)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    outPrev = neuronsPrev[depth].get(i, j)!.v[batch].out
                }
                
                neurons[curElem+depth].get(i, j)!.gc[batch][offset+elem].out =
                    outPrev
            }}}
            
            curElem += nbChannels
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
            MetalKernel.get.download([(_layersPrev[num] as! Layer2D).outs])
        }
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for depth in 0..<nbChannels
        {
            for i in 0..<height {
            for j in 0..<width
            {
                neurons[depth].get(i, j)!.initGC(
                    batchSize: batchSize,
                    nbGC: nbGC
                )
            }}
        }
        
        for batch in 0..<batchSize {
        for elem in 0..<nbSameElems {
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            let nbChannels = neuronsPrev.count
            
            for depth in 0..<nbChannels {
            for i in 0..<height {
            for j in 0..<width
            {
                let outPrev = neuronsPrev[depth].get(i, j)!.gc[batch][elem].out
                neurons[curElem+depth].get(i, j)!.gc[batch][elem].out = outPrev
            }}}
            
            curElem += nbChannels
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
            let outsPrevPtr = (_layersPrev[num] as! Layer2D).outs.shared.buffer
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            let nbChannels = neuronsPrev.count
            
            for depth in 0..<nbChannels {
            for i in 0..<height {
            for j in 0..<width
            {
                let outPrev: Double
                if num == index
                {
                    outPrev = neuronsPrev[depth].get(i, j)!
                        .gc[batch][nbLastElems[index]+elem].out
                }
                else
                {
                    let offsetStart = (depth + nbChannels * batch) * height
                    let offsetTmp = j + (offsetStart + i) * width
                    
                    outPrev = Double(outsPrevPtr[offsetTmp])
                }
                
                neurons[curElem+depth].get(i, j)!.gc[batch][offset+elem].out =
                    outPrev
            }}}
            
            curElem += nbChannels
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
                let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
                let nbChannels = neuronsPrev.count
                
                for depth in 0..<nbChannels {
                for i in 0..<height {
                for j in 0..<width
                {
                    let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                    neurons[curElem+depth].get(i, j)!.v[elem].out = outPrev
                }}}
                
                curElem += nbChannels
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
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let nbChannelsPrev = (_layersPrev[num] as! Layer2D).nbChannels
            
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            let pNbChannelsPrev: [UInt32] = [UInt32(nbChannelsPrev)]
            
            command = metalKernel.createCommand(
                "concat12DForward", deviceID: deviceID
            )
            command.setBuffer(
                (_layersPrev[num] as! Layer2D).outs.metal, atIndex: 0
            )
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pNbChannelsPrev, atIndex: 3)
            command.setBytes(pDimensions, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: width * nbChannelsPrev,
                height: height * batchSize
            )
            command.enqueue()
            
            globalOffset += nbChannelsPrev
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
                let layerPrev = _layersPrev[num] as! Layer2D
                let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
                let nbChannels = layerPrev.nbChannels
                
                if !_layersPrev[num].computeDelta
                {
                    curElem += nbChannels
                    continue
                }
                
                for depth in 0..<nbChannels {
                for i in 0..<height {
                for j in 0..<width
                {
                    let deltaCur =
                        neurons[curElem+depth].get(i, j)!.v[elem].delta
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta = deltaCur
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta += deltaCur
                    }
                }}}
                
                curElem += nbChannels
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
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! Layer2D
            let nbChannelsPrev = layerPrev.nbChannels
            
            if !_layersPrev[num].computeDelta
            {
                globalOffset += nbChannelsPrev
                continue
            }
            
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            let pNbChannelsPrev: [UInt32] = [UInt32(nbChannelsPrev)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            command = metalKernel.createCommand(
                "concat12DBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pNbChannelsPrev, atIndex: 3)
            command.setBytes(pDimensions, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: width * nbChannelsPrev,
                height: height * batchSize
            )
            command.enqueue()
            
            globalOffset += nbChannelsPrev
        }
        propagateDirty()
    }
}
