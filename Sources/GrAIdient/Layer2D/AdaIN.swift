//
// AdaIN.swift
// GrAIdient
//
// Created by Jean-François Reboud on 18/02/2023.
//

///
/// Layer with a 2D shape neural structure.
///
/// This layer merges multiple (2) layers together, using style of second layer into the first one.
///
public class AdaIN: LayerMerge2D
{
    /// Instance normalization by default or instance normalization in the CPU execution context.
    var _norm: LayerWeightsNormalization? = nil
    /// Instance normalization in the GPU execution context.
    var _normGPU: InstanceNormalizationGPU? = nil
    
    /// Get instance normalization in the CPU execution context.
    var norm: InstanceNormalization?
    {
        get {
            return _norm as? InstanceNormalization
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case norm
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layersPrev: List of previous layers that have been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layersPrev: [Layer], params: GrAI.Model.Params)
    {
        if layersPrev.count != 2 ||
           (layersPrev.first as? Layer2D) == nil ||
           (layersPrev.last as? Layer1D) == nil
        {
            fatalError("Layer structure error.")
        }
        
        let layerFirst = layersPrev.first as! Layer2D
        let layerLast = layersPrev.last as! Layer1D
        
        if layerLast.nbNeurons != 2 * layerFirst.nbChannels
        {
            fatalError("Layer structure error.")
        }
        
        super.init(layersPrev: layersPrev,
                   nbChannels: layerFirst.nbChannels,
                   height: layerFirst.height,
                   width: layerFirst.width,
                   params: params)
        
        _norm = LayerWeightsNormalization(self)
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
        _norm = try values.decodeIfPresent(
            LayerWeightsNormalization.self, forKey: .norm
        )
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
        if let norm = _normGPU
        {
            try container.encode(norm, forKey: Keys.norm)
        }
        else if let norm = _norm
        {
            try container.encode(norm, forKey: Keys.norm)
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
        
        var layersPrev = [Layer]()
        for idPrev in _idsPrev
        {
            layersPrev.append(mapping[idPrev]!)
        }
        
        let layer = AdaIN(layersPrev: layersPrev, params: params)
        if inPlace
        {
            layer._norm = _norm
            layer._normGPU = _normGPU
        }
        else
        {
            // only one of them should be cloned
            if let norm = _normGPU
            {
                layer._norm = norm.clone()
            }
            else if let norm = _norm
            {
                layer._norm = norm.clone()
            }
        }
        return layer
    }
    
    ///
    /// Clean state resources in the CPU execution context.
    ///
    /// We reset batch normalization.
    ///
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        norm?.resetKernel()
    }
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We reset batch normalization.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _normGPU?.resetKernel()
    }
    
    ///
    /// Initialize hard resources in the CPU execution context.
    ///
    /// We initialize batch normalization.
    ///
    public override func initKernelCPU()
    {
        super.initKernelCPU()
        
        if let norm = _normGPU
        {
            _norm = InstanceNormalization(norm: norm)
        }
        else if let norm = _norm
        {
            _norm = InstanceNormalization(norm: norm)
        }
        
        if !GrAI.Loop.gradientChecking
        {
            _normGPU = nil
        }
    }
    
    ///
    /// Initialize hard resources in the GPU execution context.
    ///
    /// We initialize batch normalization.
    ///
    public override func initKernelGPU()
    {
        super.initKernelGPU()
        
        if let norm = _normGPU
        {
            _normGPU = InstanceNormalizationGPU(norm: norm)
        }
        else if let norm = _norm
        {
            _normGPU = InstanceNormalizationGPU(norm: norm)
        }
        _normGPU?.initKernel(deviceID: deviceID)
        
        if !GrAI.Loop.gradientChecking
        {
            _norm = nil
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
                (_layersPrev[num1] as! Layer2D).outs.metal, atIndex: 0
            )
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
            
            var first2 = true
            for num2 in 0..<_layersPrev.count {
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
                    (_layersPrev[num2] as! Layer2D).outs.metal, atIndex: 0
                )
                command.setBytes(pNbElems, atIndex: 1)
                command.setBuffer(_otherOuts[num1].metal, atIndex: 2)
                
                command.dispatchThreads(nbElems)
                command.enqueue()
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
            
            let command = MetalKernel.get.createCommand(
                "multiplyBackward", deviceID: deviceID
            )
            command.setBuffer(_otherOuts[num].metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbElems, atIndex: 2)
            command.setBytes(pDirty, atIndex: 3)
            command.setBuffer(layerPrev.delta.metal, atIndex: 4)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
        }
        propagateDirty()
    }
    
    ///
    /// Get the outputs of the previous main branch for Gradient Checking (result of the forward pass)
    /// in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index of sample in the mini batch.
    ///     - elem: Weight estimation index during the Gradient Checking.
    /// - Returns: The outputs.
    ///
    func getOutsPrevGC(depth: Int, batch: Int, elem: Int) -> [Double]
    {
        let layerFirst = _layersPrev.first as! Layer2D
        var outs = [Double](repeating: 0.0, count: height * width)
        
        for i in 0..<height {
        for j in 0..<width
        {
            let offset = j + i * width
            outs[offset] =
                layerFirst.neurons[depth].get(i, j)!.gc[batch][elem].out
        }}
        return outs
    }
    
    ///
    /// Get the output of the previous style branch for Gradient Checking (result of the forward pass)
    /// in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index of sample in the mini batch.
    ///     - elem: Weight estimation index during the Gradient Checking.
    /// - Returns: The outputs.
    ///
    func getOutStyleGC(depth: Int, batch: Int, elem: Int) -> Double
    {
        let layerLast = _layersPrev.last as! Layer1D
        return layerLast.neurons.get(depth)!.gc[batch][elem].out
    }
    
    ///
    /// Set the outputs of Gradient Checking (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index sample in the mini batch.
    ///     - elem: Weight estimation index during the Gradient Checking.
    ///     - outs: The outputs to set.
    ///
    func setOutsGC(depth: Int, batch: Int, elem: Int, outs: [Double])
    {
        for i in 0..<height {
        for j in 0..<width
        {
            let offset = j + i * width
            neurons[depth].get(i, j)!.gc[batch][elem].out = outs[offset]
        }}
    }
    
    ///
    /// Get the outputs of the previous main branch (result of the forward pass)
    /// in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index sample in the mini batch.
    /// - Returns: The outputs.
    ///
    func getOutsPrev(depth: Int, batch: Int) -> [Double]
    {
        let layerFirst = _layersPrev.first as! Layer2D
        var outs = [Double](repeating: 0.0, count: height * width)
        
        for i in 0..<height {
        for j in 0..<width
        {
            let offset = j + i * width
            outs[offset] = layerFirst.neurons[depth].get(i, j)!.v[batch].out
        }}
        return outs
    }
    
    ///
    /// Get the output of the previous style branch (result of the forward pass)
    /// in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index sample in the mini batch.
    /// - Returns: The outputs.
    ///
    func getOutStyle(depth: Int, batch: Int) -> Double
    {
        let layerLast = _layersPrev.last as! Layer1D
        return layerLast.neurons.get(depth)!.v[batch].out
    }
    
    ///
    /// Set the outputs (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index sample in the mini batch.
    ///     - outs: The outputs to set.
    ///
    func setOuts(depth: Int, batch: Int, outs: [Double])
    {
        for i in 0..<height {
        for j in 0..<width
        {
            let offset = j + i * width
            neurons[depth].get(i, j)!.v[batch].out = outs[offset]
        }}
    }
    
    ///
    /// Get the gradients (result of the backward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index sample in the mini batch.
    /// - Returns: The gradients.
    ///
    func getDelta(depth: Int, batch: Int) -> [Double]
    {
        var delta = [Double](repeating: 0.0, count: height * width)
        for i in 0..<height {
        for j in 0..<width
        {
            let offset = j + i * width
            delta[offset] = neurons[depth].get(i, j)!.v[batch].delta
        }}
        return delta
    }
    
    ///
    /// Set the gradients of the previous main branch (result of the backward pass)
    /// in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index sample in the mini batch.
    ///     - outs: The gradients to set.
    ///
    func setDeltaPrev(depth: Int, batch: Int, delta: [Double])
    {
        let layerFirst = _layersPrev.first as! Layer2D
        
        for i in 0..<height {
        for j in 0..<width
        {
            let offset = j + i * width
            
            if layerFirst.dirty
            {
                layerFirst.neurons[depth].get(i, j)!.v[batch].delta =
                    delta[offset]
            }
            else
            {
                layerFirst.neurons[depth].get(i, j)!.v[batch].delta +=
                    delta[offset]
            }
        }}
    }
    
    ///
    /// Set the gradients of the previous style branch (result of the backward pass)
    /// in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index sample in the mini batch.
    ///     - delta: The gradient to set.
    ///
    func setDeltaStyle(depth: Int, batch: Int, delta: Double)
    {
        let layerLast = _layersPrev.last as! Layer1D
        
        if layerLast.dirty
        {
            layerLast.neurons.get(depth)!.v[batch].out = delta
        }
        else
        {
            layerLast.neurons.get(depth)!.v[batch].out += delta
        }
    }
}
