//
// Sum2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

///
/// Layer with a 2D shape neural structure.
///
/// This layer merges multiple 2D layers together, summing the neurons at the same localization
/// in the input grids (pixelwise sum).
///
public class Sum2D: LayerMerge2D
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
        
        let layer = try! Sum2D(layersPrev: layersPrev, params: params)
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
        for depth in 0..<nbChannels
        {
            for i in 0..<height {
            for j in 0..<width
            {
                var sum = 0.0
                for num in 0..<_layersPrev.count
                {
                    let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
                    sum += neuronsPrev[depth].get(i, j)!.gc[batch][elem].out
                }
                
                neurons[depth].get(i, j)!.gc[batch][elem].out = sum
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
                var sum = 0.0
                for num in 0..<_layersPrev.count
                {
                    let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
                    
                    if num == index
                    {
                        sum += neuronsPrev[depth].get(i, j)!
                            .gc[batch][nbLastElems[index]+elem].out
                    }
                    else
                    {
                        sum += neuronsPrev[depth].get(i, j)!.v[batch].out
                    }
                }
                
                neurons[depth].get(i, j)!
                    .gc[batch][offset+elem].out = sum
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
        
        var buffersPrev = [[Float]]()
        for num in 0..<_layersPrev.count
        {
            buffersPrev.append((_layersPrev[num] as! Layer2D).outs.download())
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
                var sum = 0.0
                for num in 0..<_layersPrev.count
                {
                    let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
                    sum += neuronsPrev[depth].get(i, j)!.gc[batch][elem].out
                }
                
                neurons[depth].get(i, j)!.gc[batch][elem].out = sum
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
                var sum = 0.0
                for num in 0..<_layersPrev.count
                {
                    let outsPrevPtr = buffersPrev[num]
                    let neuronsPrev =
                        (_layersPrev[num] as! Layer2D).neurons
                    
                    if num == index
                    {
                        sum += neuronsPrev[depth].get(i, j)!
                            .gc[batch][nbLastElems[index]+elem].out
                    }
                    else
                    {
                        let offsetStart = (depth + nbChannels * batch) * height
                        let offsetTmp = j + (offsetStart + i) * width
                        
                        sum += Double(outsPrevPtr[offsetTmp])
                    }
                }
                
                neurons[depth].get(i, j)!.gc[batch][offset+elem].out = sum
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
            for i in 0..<height {
            for j in 0..<width
            {
                var sum = 0.0
                for num in 0..<_layersPrev.count
                {
                    let neuronsPrev =
                        (_layersPrev[num] as! Layer2D).neurons
                    sum += neuronsPrev[depth].get(i, j)!.v[elem].out
                }
                
                neurons[depth].get(i, j)!.v[elem].out = sum
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
        
        var first = true
        for num in 0..<_layersPrev.count
        {
            let nbElems = (_layersPrev[num] as! Layer2D).outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let kernel: String
            let coeff = nbElems % 4 == 0 ? 4 : 1
            if first
            {
                kernel = nbElems % 4 == 0 ? "sum14" : "sum1"
                first = false
            }
            else
            {
                kernel = nbElems % 4 == 0 ? "sum24" : "sum2"
            }
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            
            command.setBuffer(
                (_layersPrev[num] as! Layer2D).outs.metal, atIndex: 0
            )
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems / coeff)
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
            
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let deltaCur = neurons[depth].get(i, j)!.v[elem].delta
                    
                    if _layersPrev[num].dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta =
                            deltaCur
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            deltaCur
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
            
            try (_layersPrev[num] as! Layer2D).checkStateBackwardGPU(
                batchSize: batchSize
            )
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let kernel: String
            let coeff = nbElems % 4 == 0 ? 4 : 1
            if _layersPrev[num].dirty
            {
                kernel = nbElems % 4 == 0 ? "sum14" : "sum1"
            }
            else
            {
                kernel = nbElems % 4 == 0 ? "sum24" : "sum2"
            }
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(
                (_layersPrev[num] as! Layer2D).delta.metal, atIndex: 2
            )
            
            command.dispatchThreads(nbElems / coeff)
            command.enqueue()
        }
        propagateDirty()
    }
}
