//
// Dropout1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 07/10/2023.
//

///
/// Layer with a 1D shape neural structure.
///
/// This layer randomly sets neurons to zero.
///
public class Dropout1D: Layer1D
{
    /// Probability for each neuron to be zeroed.
    public var coeff: Double = 0.5
    
    ///
    /// Whether each neurons is zeroed or not.
    /// ~ (batch, nbNeurons)
    ///
    var _dropout: MetalSharedBuffer<Bool>! = nil
    
    private enum Keys: String, CodingKey
    {
        case coeff
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - coeff: Probability for each neuron to be zeroed.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer1D,
                coeff: Double,
                params: GrAI.Model.Params)
    {
        self.coeff = coeff
        super.init(layerPrev: layerPrev,
                   nbNeurons: layerPrev.nbNeurons,
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
        coeff = try values.decode(Double.self, forKey: Keys.coeff)
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
        try container.encode(coeff, forKey: Keys.coeff)
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
        let layerPrev = mapping[idPrev] as! Layer1D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = Dropout1D(
            layerPrev: layerPrev,
            coeff: coeff,
            params: params
        )
        return layer
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _dropout = nil
    }
    
    ///
    /// Initialize state resources in the CPU execution context.
    ///
    /// We initialize the neurons' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        try super.checkStateCPU(batchSize: batchSize)
        
        if _dropout == nil
        {
            _dropout = MetalSharedBuffer<Bool>(
                batchSize * nbNeurons,
                deviceID: deviceID
            )
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
        
        if _dropout == nil
        {
            _dropout = MetalSharedBuffer<Bool>(
                batchSize * nbNeurons,
                deviceID: deviceID
            )
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let applyDropout = phase != nil && phase == .Training
            let dropoutPtr = _dropout.buffer
            
            let nbGC = layerPrev.nbGC
            for j in 0..<nbNeurons
            {
                neurons.get(j)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbNeurons
                {
                    if applyDropout && !dropoutPtr[depth + nbNeurons * batch]
                    {
                        neurons.get(depth)!.gc[batch][elem].out =
                            1.0 / (1.0 - coeff) *
                            neuronsPrev.get(depth)!.gc[batch][elem].out
                    }
                    else if applyDropout
                    {
                        neurons.get(depth)!.gc[batch][elem].out = 0
                    }
                    else
                    {
                        neurons.get(depth)!.gc[batch][elem].out =
                            neuronsPrev.get(depth)!.gc[batch][elem].out
                    }
                }
            }}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let applyDropout = phase != nil && phase == .Training
            let dropoutPtr = _dropout.download()
            
            let nbGC = layerPrev.nbGC
            for j in 0..<nbNeurons
            {
                neurons.get(j)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbNeurons
                {
                    if applyDropout && !dropoutPtr[depth + nbNeurons * batch]
                    {
                        neurons.get(depth)!.gc[batch][elem].out =
                            1.0 / (1.0 - coeff) *
                            neuronsPrev.get(depth)!.gc[batch][elem].out
                    }
                    else if applyDropout
                    {
                        neurons.get(depth)!.gc[batch][elem].out = 0
                    }
                    else
                    {
                        neurons.get(depth)!.gc[batch][elem].out =
                            neuronsPrev.get(depth)!.gc[batch][elem].out
                    }
                }
            }}
        }
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let applyDropout = phase != nil && phase == .Training
            let dropoutPtr = _dropout.buffer
            
            for elem in 0..<batchSize {
            for depth in 0..<nbNeurons
            {
                let alea = Double.random(in: 0 ..< 1)
                if alea <= coeff
                {
                    dropoutPtr[depth + nbNeurons * elem] = true
                }
                else
                {
                    dropoutPtr[depth + nbNeurons * elem] = false
                }
            }}
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    if applyDropout && !dropoutPtr[depth + nbNeurons * elem]
                    {
                        neurons.get(depth)!.v[elem].out =
                            1.0 / (1.0 - coeff) *
                            neuronsPrev.get(depth)!.v[elem].out
                    }
                    else if applyDropout
                    {
                        neurons.get(depth)!.v[elem].out = 0
                    }
                    else
                    {
                        neurons.get(depth)!.v[elem].out =
                            neuronsPrev.get(depth)!.v[elem].out
                    }
                }
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
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let applyDropout = phase != nil && phase == .Training
            if applyDropout
            {
                let dropoutPtr = _dropout.buffer
                for elem in 0..<batchSize {
                for depth in 0..<nbNeurons
                {
                    let alea = Double.random(in: 0 ..< 1)
                    if alea <= coeff
                    {
                        dropoutPtr[depth + nbNeurons * elem] = true
                    }
                    else
                    {
                        dropoutPtr[depth + nbNeurons * elem] = false
                    }
                }}
                _dropout.upload()
            }
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pApplyDropout: [Bool] = [applyDropout]
            let pCoeff: [Float16] = [Float16(coeff)]
            
            let command = MetalKernel.get.createCommand(
                "dropout1DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(_dropout.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pApplyDropout, atIndex: 4)
            command.setBytes(pCoeff, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            let applyDropout = phase != nil && phase == .Training
            let dropoutPtr = _dropout.buffer
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    let newValue: Double
                    if applyDropout && !dropoutPtr[depth + nbNeurons * elem]
                    {
                        newValue = 1.0 / (1.0 - coeff) *
                            neurons.get(depth)!.v[elem].delta
                    }
                    else if applyDropout
                    {
                        newValue = 0
                    }
                    else
                    {
                        newValue = neurons.get(depth)!.v[elem].delta
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(depth)!.v[elem].delta = newValue
                    }
                    else
                    {
                        neuronsPrev.get(depth)!.v[elem].delta += newValue
                    }
                }
            }
            propagateDirty()
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let applyDropout = phase != nil && phase == .Training
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pApplyDropout: [Bool] = [applyDropout]
            let pCoeff: [Float16] = [Float16(coeff)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "dropout1DBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(_dropout.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pApplyDropout, atIndex: 4)
            command.setBytes(pCoeff, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
