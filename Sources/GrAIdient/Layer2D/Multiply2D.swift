//
// Multiply2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 03/12/2022.
//

///
/// Layer with a 2D shape neural structure.
///
/// This layer merges multiple 2D layers together, multipliying the neurons at the same localization
/// in the input grids (pixelwise multiplication).
///
public class Multiply2D: LayerMerge2D
{
    ///
    /// List of output buffers.
    /// Shape ~ (batch, nbChannels, height, width).
    ///
    var _otherOuts: [MetalBuffer<Float>] = []
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layersPrev: List of previous layers that have been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layersPrev: [Layer2D], params: GrAI.Model.Params) throws
    {
        let layer0 = layersPrev[0]
        for layerPrev in layersPrev
        {
            if layerPrev.nbChannels != layer0.nbChannels ||
               layerPrev.height != layer0.height ||
               layerPrev.width != layer0.width
            {
                throw LayerError.Init(message: "Layer structure error.")
            }
        }
        
        super.init(layersPrev: layersPrev,
                   nbChannels: layer0.nbChannels,
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
        
        var layersPrev = [Layer2D]()
        for idPrev in _idsPrev
        {
            layersPrev.append(mapping[idPrev] as! Layer2D)
        }
        
        let layer = try! Multiply2D(layersPrev: layersPrev, params: params)
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
        _otherOuts = []
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _otherOuts = []
    }
    
    ///
    /// Initialize state resources in the CPU execution context.
    ///
    /// We initialize the neurons' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        try super.checkStateCPU(batchSize: batchSize)
        
        if _otherOuts.count == 0
        {
            for _ in 0..<_layersPrev.count
            {
                let buffer = MetalSharedBuffer<Float>(
                    batchSize * nbChannels * height * width,
                    deviceID: deviceID
                )
                _otherOuts.append(buffer)
            }
        }
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if _otherOuts.count == 0
        {
            for _ in 0..<_layersPrev.count
            {
                let buffer = MetalPrivateBuffer<Float>(
                    batchSize * nbChannels * height * width,
                    deviceID: deviceID
                )
                _otherOuts.append(buffer)
            }
        }
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
        for depth in 0..<nbChannels
        {
            for i in 0..<height {
            for j in 0..<width
            {
                var mult = 1.0
                for num in 0..<_layersPrev.count
                {
                    let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
                    mult *= neuronsPrev[depth].get(i, j)!.gc[batch][elem].out
                }
                
                neurons[depth].get(i, j)!.gc[batch][elem].out = mult
            }}
        }}}
    
        for batch in 0..<batchSize {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        for depth in 0..<nbChannels
        {
            for i in 0..<height {
            for j in 0..<width
            {
                var mult = 1.0
                for num in 0..<_layersPrev.count
                {
                    let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
                    
                    if num == index
                    {
                        mult *= neuronsPrev[depth].get(i, j)!
                            .gc[batch][nbLastElems[index]+elem].out
                    }
                    else
                    {
                        mult *= neuronsPrev[depth].get(i, j)!.v[batch].out
                    }
                }
                
                neurons[depth].get(i, j)!
                    .gc[batch][offset+elem].out = mult
            }}
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
        for depth in 0..<nbChannels
        {
            for i in 0..<height {
            for j in 0..<width
            {
                var mult = 1.0
                for num in 0..<_layersPrev.count
                {
                    let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
                    mult *= neuronsPrev[depth].get(i, j)!.gc[batch][elem].out
                }
                
                neurons[depth].get(i, j)!.gc[batch][elem].out = mult
            }}
        }}}
        
        for batch in 0..<batchSize {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        for depth in 0..<nbChannels
        {
            for i in 0..<height {
            for j in 0..<width
            {
                var mult = 1.0
                for num in 0..<_layersPrev.count
                {
                    let outsPrevPtr =
                        (_layersPrev[num] as! Layer2D).outs.shared.buffer
                    let neuronsPrev =
                        (_layersPrev[num] as! Layer2D).neurons
                    
                    if num == index
                    {
                        mult *= neuronsPrev[depth].get(i, j)!
                            .gc[batch][nbLastElems[index]+elem].out
                    }
                    else
                    {
                        let offsetStart = (depth + nbChannels * batch) * height
                        let offsetTmp = j + (offsetStart + i) * width
                        
                        mult *= Double(outsPrevPtr[offsetTmp])
                    }
                }
                
                neurons[depth].get(i, j)!.gc[batch][offset+elem].out = mult
            }}
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
        
        for elem in 0..<batchSize {
        for depth in 0..<nbChannels
        {
            let offsetStart = (depth + nbChannels * elem) * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                
                var mult = 1.0
                for num in 0..<_layersPrev.count
                {
                    let neuronsPrev =
                        (_layersPrev[num] as! Layer2D).neurons
                    mult *= neuronsPrev[depth].get(i, j)!.v[elem].out
                }
                neurons[depth].get(i, j)!.v[elem].out = mult
                
                for num1 in 0..<_layersPrev.count
                {
                    let buffer = (_otherOuts[num1] as! MetalSharedBuffer).buffer
                    
                    mult = 1.0
                    for num2 in 0..<_layersPrev.count {
                    if num2 != num1
                    {
                        let neuronsPrev =
                            (_layersPrev[num2] as! Layer2D).neurons
                        mult *= neuronsPrev[depth].get(i, j)!.v[elem].out
                    }}
                    
                    buffer[offset] = Float(mult)
                }
            }}
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
        
        var first1 = true
        for num1 in 0..<_layersPrev.count
        {
            let nbElems = (_layersPrev[num1] as! Layer2D).outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            var command: MetalEncoder
            if first1
            {
                command = MetalKernel.get.createEncoder(
                    "sum1", deviceID: deviceID
                )
                first1 = false
            }
            else
            {
                command = MetalKernel.get.createEncoder(
                    "multiplyForward", deviceID: deviceID
                )
            }
            
            command.setBuffer(
                (_layersPrev[num1] as! Layer2D).outs.metal, atIndex: 0
            )
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems)
            command.endEncoding()
            
            var first2 = true
            for num2 in 0..<_layersPrev.count {
            if num2 != num1
            {
                if first2
                {
                    command = MetalKernel.get.createEncoder(
                        "sum1", deviceID: deviceID
                    )
                    first2 = false
                }
                else
                {
                    command = MetalKernel.get.createEncoder(
                        "multiplyForward", deviceID: deviceID
                    )
                }
                
                command.setBuffer(
                    (_layersPrev[num2] as! Layer2D).outs.metal, atIndex: 0
                )
                command.setBytes(pNbElems, atIndex: 1)
                command.setBuffer(_otherOuts[num1].metal, atIndex: 2)
                
                command.dispatchThreads(nbElems)
                command.endEncoding()
            }}
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
            
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            let buffer = (_otherOuts[num] as! MetalSharedBuffer).buffer
            
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                let offsetStart = (depth + nbChannels * elem) * height
                
                for i in 0..<height {
                for j in 0..<width
                {
                    let offset = j + (offsetStart + i) * width
                    
                    let tmp = Double(buffer[offset])
                    let deltaCur = neurons[depth].get(i, j)!.v[elem].delta
                    
                    if _layersPrev[num].dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta =
                            deltaCur * tmp
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            deltaCur * tmp
                    }
                }}
            }}
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
            let layerPrev = _layersPrev[num] as! Layer2D
            
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createEncoder(
                "multiplyBackward", deviceID: deviceID
            )
            command.setBuffer(_otherOuts[num].metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbElems, atIndex: 2)
            command.setBytes(pDirty, atIndex: 3)
            command.setBuffer(layerPrev.delta.metal, atIndex: 4)
            
            command.dispatchThreads(nbElems)
            command.endEncoding()
        }
        propagateDirty()
    }
}
