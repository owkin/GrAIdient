//
// ActivationSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 21/02/2023.
//

/// Layer with a sequential shape neural structure and an activation function.
public class ActivationSeq: LayerSeq
{
    /// The activation function.
    let _activation: ActivationFunction?
    
    ///
    /// Pre output buffer (result of the forward pass before applying activation)
    /// used in the GPU execution context.
    /// Shape ~ (batch, nbNeurons).
    ///
    var _tmp: MetalPrivateBuffer<Float>! = nil
    
    /// Get coefficient (depending on activation function) to apply during the weights initialization.
    public var coeffInitWeights: Float
    {
        get {
            if let activation = _activation
            {
                return activation.coeffInitWeights
            }
            return 1.0
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case activation
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - activation: The activation function.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq,
                activation: String,
                params: GrAI.Model.Params)
    {
        _activation = GrAI.Model.Activation.build(activation)
        
        super.init(layerPrev: layerPrev,
                   sequence: layerPrev.sequence,
                   nbNeurons: layerPrev.nbNeurons,
                   params: params)
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - nbNeurons: Number of neurons.
    ///     - activation: The activation function.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer?,
                sequence: Int, nbNeurons: Int, activation: String?,
                params: GrAI.Model.Params)
    {
        if let activationStr = activation
        {
            _activation = GrAI.Model.Activation.build(activationStr)
        }
        else
        {
            _activation = nil
        }
        
        super.init(layerPrev: layerPrev,
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
        let container = try decoder.container(keyedBy: Keys.self)
        _activation =
            try container.decodeIfPresent(ActivationContainer.self,
                                          forKey: .activation)?.activation
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
        if let activation = _activation
        {
            try container.encode(ActivationContainer(activation),
                                 forKey: Keys.activation)
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
        let layerPrev = mapping[idPrev] as! LayerSeq
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = ActivationSeq(
            layerPrev: layerPrev,
            activation: _activation!.name,
            params: params
        )
        return layer
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// State resources are the resources that are dependent on the batch size.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _tmp = nil
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        try _forwardGC()
        _activation!.forwardGC(self)
    }
    
    ///
    /// Apply the forward pass (until the activation function) of the Gradient Checking.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    private func _forwardGC() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for seq in 0..<sequence {
            for depth in 0..<nbNeurons
            {
                neurons.get(seq, depth)!.initGC(
                    batchSize: batchSize, nbGC: nbGC
                )
            }}
            
            let neuronsPrev = layerPrev.neurons!
            for batch in 0..<batchSize {
            for seq in 0..<sequence {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbNeurons
                {
                    neurons.get(seq, depth)!.gc[batch][elem].out =
                        neuronsPrev.get(seq, depth)!.gc[batch][elem].out
                }
            }}}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try _forwardGC()
        _activation!.forwardGC(self)
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons!
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                for depth in 0..<nbNeurons
                {
                    neurons.get(seq, depth)!.v[elem].out =
                        neuronsPrev.get(seq, depth)!.v[elem].out
                }
            }}
            
            _activation!.forwardCPU(self)
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let nbElems = outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let kernel = nbElems % 4 == 0 ? "sum14" : "sum1"
            let coeff = nbElems % 4 == 0 ? 4 : 1
            let command = MetalKernel.get.createEncoder(
                kernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems / coeff)
            command.endEncoding()
            
            _activation!.forwardGPU(self)
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        _activation!.backwardCPU(self)
        
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons!
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                for depth in 0..<nbNeurons
                {
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(seq, depth)!.v[elem].delta =
                            neurons.get(seq, depth)!.v[elem].delta
                    }
                    else
                    {
                        neuronsPrev.get(seq, depth)!.v[elem].delta +=
                            neurons.get(seq, depth)!.v[elem].delta
                    }
                }
            }}
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
        _activation!.backwardGPU(self)
        
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let kernel: String
            let coeff = nbElems % 4 == 0 ? 4 : 1
            if layerPrev.dirty
            {
                kernel = nbElems % 4 == 0 ? "sum14" : "sum1"
            }
            else
            {
                kernel = nbElems % 4 == 0 ? "sum24" : "sum2"
            }
            let command = MetalKernel.get.createEncoder(
                kernel, deviceID: deviceID
            )
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(layerPrev.delta.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems / coeff)
            command.endEncoding()
            
            propagateDirty()
        }
    }
}
