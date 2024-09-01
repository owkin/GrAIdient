//
// DotProduct1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 28/01/2023.
//

///
/// Layer with a 1D shape neural structure.
///
/// This layer merges multiple (2) 1D layers,
/// multiplying then summing the neurons in the same blocks together.
///
public class DotProduct1D: LayerMerge1D
{
    let _size: Int
    
    private enum Keys: String, CodingKey
    {
        case size
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layersPrev: List of previous layers that have been queued to the model.
    ///     - size: The number of neurons per block.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layersPrev: [Layer1D],
                size: Int,
                params: GrAI.Model.Params) throws
    {
        if layersPrev.count != 2 ||
           layersPrev[0].nbNeurons != layersPrev[1].nbNeurons ||
           layersPrev[0].nbNeurons % size != 0
        {
            throw LayerError.Init(message: "Inconsistent number of neurons.")
        }
        
        _size = size
        super.init(layersPrev: layersPrev,
                   nbNeurons: layersPrev[0].nbNeurons / size,
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
        let values = try decoder.container(keyedBy: Keys.self)
        _size = try values.decode(Int.self, forKey: Keys.size)
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
        try container.encode(_size, forKey: Keys.size)
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
        
        var layersPrev = [Layer1D]()
        for idPrev in idsPrev
        {
            layersPrev.append(mapping[idPrev] as! Layer1D)
        }
        
        let layer = try! DotProduct1D(
            layersPrev: layersPrev,
            size: _size,
            params: params
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
        
        for depth in 0..<nbNeurons
        {
            neurons.get(depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
        }
        
        let neuronsPrev1 = (layersPrev[0] as! Layer1D).neurons
        let neuronsPrev2 = (layersPrev[1] as! Layer1D).neurons
        
        for batch in 0..<batchSize {
        for elem in 0..<nbSameElems
        {
            for block in 0..<nbNeurons
            {
                var sum = 0.0
                for j in 0..<_size
                {
                    let neuronPrev1 = neuronsPrev1.get(j + block * _size)!
                    let neuronPrev2 = neuronsPrev2.get(j + block * _size)!
                    let out1 = neuronPrev1.gc[batch][elem].out
                    let out2 = neuronPrev2.gc[batch][elem].out
                    sum += out1 * out2
                }
                neurons.get(block)!.gc[batch][elem].out = sum
            }
        }}
        
        for batch in 0..<batchSize {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        for block in 0..<nbNeurons
        {
            var sum = 0.0
            for j in 0..<_size
            {
                let neuronPrev1 = neuronsPrev1.get(j + block * _size)!
                let neuronPrev2 = neuronsPrev2.get(j + block * _size)!
                
                let out1: Double
                let out2: Double
                if index == 0
                {
                    out1 = neuronPrev1.gc[batch][nbLastElems[index]+elem].out
                    out2 = neuronPrev2.v[batch].out
                }
                else
                {
                    out1 = neuronPrev1.v[batch].out
                    out2 = neuronPrev2.gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += out1 * out2
            }
            neurons.get(block)!.gc[batch][offset+elem].out = sum
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
        
        let neuronsPrev1 = (layersPrev[0] as! Layer1D).neurons
        let neuronsPrev2 = (layersPrev[1] as! Layer1D).neurons
        
        for batch in 0..<batchSize {
        for elem in 0..<nbSameElems
        {
            for block in 0..<nbNeurons
            {
                var sum = 0.0
                for j in 0..<_size
                {
                    let neuronPrev1 = neuronsPrev1.get(j + block * _size)!
                    let neuronPrev2 = neuronsPrev2.get(j + block * _size)!
                    let out1 = neuronPrev1.gc[batch][elem].out
                    let out2 = neuronPrev2.gc[batch][elem].out
                    sum += out1 * out2
                }
                neurons.get(block)!.gc[batch][elem].out = sum
            }
        }}
        
        let buffer1 = (layersPrev[0] as! Layer1D).outs.download()
        let buffer2 = (layersPrev[1] as! Layer1D).outs.download()
        
        for batch in 0..<batchSize {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        for block in 0..<nbNeurons
        {
            var sum = 0.0
            for j in 0..<_size
            {
                let neuronPrev1 = neuronsPrev1.get(j + block * _size)!
                let neuronPrev2 = neuronsPrev2.get(j + block * _size)!
                
                let out1: Double
                let out2: Double
                if index == 0
                {
                    out1 = neuronPrev1.gc[batch][nbLastElems[index]+elem].out
                    out2 = Double(buffer2[j+(block+nbNeurons*batch)*_size])
                }
                else
                {
                    out1 = Double(buffer1[j+(block+nbNeurons*batch)*_size])
                    out2 = neuronPrev2.gc[batch][nbLastElems[index]+elem].out
                }
                
                sum += out1 * out2
            }
            neurons.get(block)!.gc[batch][offset+elem].out = sum
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
    public override func forwardCPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        let neuronsPrev1 = (layersPrev[0] as! Layer1D).neurons
        let neuronsPrev2 = (layersPrev[1] as! Layer1D).neurons
        
        for elem in 0..<batchSize
        {
            for block in 0..<nbNeurons
            {
                var sum = 0.0
                for j in 0..<_size
                {
                    let neuronPrev1 = neuronsPrev1.get(j + block * _size)!
                    let neuronPrev2 = neuronsPrev2.get(j + block * _size)!
                    let out1 = neuronPrev1.v[elem].out
                    let out2 = neuronPrev2.v[elem].out
                    sum += out1 * out2
                }
                neurons.get(block)!.v[elem].out = sum
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
        
        let nbNeuronsPrev = (layersPrev[0] as! Layer1D).nbNeurons
        
        let pSize: [UInt32] = [UInt32(_size)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbneuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let command = MetalKernel.get.createCommand(
            "dotProduct1DForward", deviceID: deviceID
        )
        command.setBuffer((layersPrev[0] as! Layer1D).outs.metal, atIndex: 0)
        command.setBuffer((layersPrev[1] as! Layer1D).outs.metal, atIndex: 1)
        command.setBytes(pSize, atIndex: 2)
        command.setBytes(pNbNeurons, atIndex: 3)
        command.setBytes(pNbneuronsPrev, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
        command.setBuffer(outs.metal, atIndex: 6)
        
        command.dispatchThreads(
            width: nbNeurons,
            height: batchSize
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
        
        let neuronsPrev1 = (layersPrev[0] as! Layer1D).neurons
        let neuronsPrev2 = (layersPrev[1] as! Layer1D).neurons
        
        for elem in 0..<batchSize
        {
            for block in 0..<nbNeurons
            {
                let deltaCur = neurons.get(block)!.v[elem].delta
                for j in 0..<_size
                {
                    let neuronPrev1 = neuronsPrev1.get(j + block * _size)!
                    let neuronPrev2 = neuronsPrev2.get(j + block * _size)!
                    let out1 = neuronPrev1.v[elem].out
                    let out2 = neuronPrev2.v[elem].out
                
                    if layersPrev[0].dirty && layersPrev[0].computeDelta
                    {
                        neuronPrev1.v[elem].delta = deltaCur * out2
                    }
                    else if layersPrev[0].computeDelta
                    {
                        neuronPrev1.v[elem].delta += deltaCur * out2
                    }
                    
                    if layersPrev[1].dirty && layersPrev[1].computeDelta
                    {
                        neuronPrev2.v[elem].delta = deltaCur * out1
                    }
                    else if layersPrev[1].computeDelta
                    {
                        neuronPrev2.v[elem].delta += deltaCur * out1
                    }
                }
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
        
        let layerPrev1 = layersPrev[0] as! Layer1D
        let layerPrev2 = layersPrev[1] as! Layer1D
        
        let nbNeuronsPrev = layerPrev1.nbNeurons
        
        let pSize: [UInt32] = [UInt32(_size)]
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbneuronsPrev: [UInt32] = [UInt32(nbNeuronsPrev)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        var command: MetalCommand
        if layerPrev1.computeDelta
        {
            try layerPrev1.checkStateBackwardGPU(
                batchSize: batchSize
            )
            
            let pDirty: [UInt32] = layerPrev1.dirty ? [1] : [0]
            
            command = MetalKernel.get.createCommand(
                "dotProduct1DBackward", deviceID: deviceID
            )
            command.setBuffer(layerPrev2.outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pSize, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbneuronsPrev, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev1.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
        }
        
        if layerPrev2.computeDelta
        {
            try layerPrev2.checkStateBackwardGPU(
                batchSize: batchSize
            )
            
            let pDirty: [UInt32] = layerPrev2.dirty ? [1] : [0]
            
            command = MetalKernel.get.createCommand(
                "dotProduct1DBackward", deviceID: deviceID
            )
            command.setBuffer(layerPrev1.outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pSize, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbneuronsPrev, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev2.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
        }
        propagateDirty()
    }
}
