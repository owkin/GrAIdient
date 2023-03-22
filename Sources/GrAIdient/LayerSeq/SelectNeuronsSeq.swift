//
// SelectNeuronsSeq.swift
// GrAIdient
//
// Created by Aurélien PEDEN on 17/03/2023.
//

///
/// Layer with a 1D shape neural structure.
///
/// This layer selects one element of a sequence and transforms a LayerSeq into a Layer1D.
///
public class SelectNeuronsSeq: Layer1D
{
    /// Sequence of the selected neurons.
    let _targetSeq: Int
    
    /// List of neurons to select.
    let _neurons: [Int]
    /// List of coefficients to scale each selected neuron.
    let _coeffs: [Double]
    
    ///
    /// Indices of selected neurons.
    /// Shape ~ (nbNeurons,).
    ///
    var _neuronsBuffer: MetalPrivateBuffer<UInt32>! = nil
    ///
    /// Coefficients of selected neurons.
    /// Shape ~ (nbNeurons,).
    ///
    var _coeffsBuffer: MetalPrivateBuffer<Float>! = nil
    
    private enum Keys: String, CodingKey
    {
        case targetSeq
        case neurons
        case coeffs
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - targetSeq: Sequence of the selected neurons.
    ///     - neurons: The list of neurons to select.
    ///     - coeffs: The list of coefficients to scale each selected neuron.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq,
                targetSeq: Int,
                neurons: [Int], coeffs: [Double],
                params: GrAI.Model.Params)
    {
        _targetSeq = targetSeq
        _neurons = neurons
        _coeffs = coeffs
        
        super.init(layerPrev: layerPrev,
                   nbNeurons: _neurons.count,
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
        _targetSeq = try values.decode(Int.self, forKey: Keys.targetSeq)
        _neurons = try values.decode([Int].self, forKey: .neurons)
        _coeffs = try values.decode([Double].self, forKey: .coeffs)
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
        try container.encode(_targetSeq, forKey: Keys.targetSeq)
        try container.encode(_neurons, forKey: Keys.neurons)
        try container.encode(_coeffs, forKey: Keys.coeffs)
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
            
        let layer = SelectNeuronsSeq(
            layerPrev: layerPrev,
            targetSeq: _targetSeq,
            neurons: _neurons,
            coeffs: _coeffs,
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
        _neuronsBuffer = nil
        _coeffsBuffer = nil
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if _neuronsBuffer == nil
        {
            _neuronsBuffer = MetalPrivateBuffer<UInt32>(
                nbNeurons, deviceID: deviceID
            )
            _coeffsBuffer = MetalPrivateBuffer<Float>(
                nbNeurons, deviceID: deviceID
            )
            
            let neuronsPtr = _neuronsBuffer.shared.buffer
            let coeffsPtr = _coeffsBuffer.shared.buffer
            
            for (num, neuron) in _neurons.enumerated()
            {
                neuronsPtr[num] = UInt32(neuron)
                coeffsPtr[num] = Float(_coeffs[num])
            }
            
            MetalKernel.get.upload([_neuronsBuffer])
            MetalKernel.get.upload([_coeffsBuffer])
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for depth in 0..<nbNeurons
            {
                neurons.get(depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronsPrev = layerPrev.neurons!
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(_targetSeq, _neurons[depth])!
                    neurons.get(depth)!.gc[batch][elem].out =
                        _coeffs[depth] * outPrev.gc[batch][elem].out
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
        try forwardGCCPU()
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
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    neurons.get(depth)!.v[elem].out =
                        _coeffs[depth] * neuronsPrev
                        .get(_targetSeq, _neurons[depth])!.v[elem].out
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
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pTargetSeq: [UInt32] = [UInt32(_targetSeq)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbNeuronsPrev: [UInt32] = [UInt32(layerPrev.nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(layerPrev.sequence)]
            
            let command = MetalKernel.get.createCommand(
                "selectNeuronsSeqForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pNbNeuronsPrev, atIndex: 2)
            command.setBytes(pTargetSeq, atIndex: 3)
            command.setBuffer(_neuronsBuffer.metal, atIndex: 4)
            command.setBuffer(_coeffsBuffer.metal, atIndex: 5)
            command.setBytes(pNbBatch, atIndex: 6)
            command.setBytes(pSequence, atIndex: 7)
            command.setBuffer(outs.metal, atIndex: 8)
            
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
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons!
            let sequence = layerPrev.sequence
            
            if layerPrev.dirty
            {
                for elem in 0..<batchSize {
                for depth in 0..<layerPrev.nbNeurons {
                for seq in 0..<sequence
                {
                    neuronsPrev.get(seq, depth)!.v[elem].delta = 0.0
                }}}
            }
            
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    neuronsPrev.get(_targetSeq, _neurons[depth])!
                        .v[elem].delta +=
                        _coeffs[depth] * neurons.get(depth)!.v[elem].delta
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
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            var command: MetalCommand
            if layerPrev.dirty
            {
                let nbElems = layerPrev.delta.nbElems
                let pNbElems: [UInt32] = [UInt32(nbElems)]
                
                command = MetalKernel.get.createCommand(
                    "reset", deviceID: deviceID
                )
                command.setBytes(pNbElems, atIndex: 0)
                command.setBuffer(layerPrev.delta.metal, atIndex: 1)
                
                command.dispatchThreads(nbElems)
                command.enqueue()
            }
            
            let pTargetSeq: [UInt32] = [UInt32(_targetSeq)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbNeuronsPrev: [UInt32] = [UInt32(layerPrev.nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(layerPrev.sequence)]
            
            command = MetalKernel.get.createCommand(
                "selectNeuronsSeqBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pNbNeuronsPrev, atIndex: 2)
            command.setBytes(pTargetSeq, atIndex: 3)
            command.setBuffer(_neuronsBuffer.metal, atIndex: 4)
            command.setBuffer(_coeffsBuffer.metal, atIndex: 5)
            command.setBytes(pNbBatch, atIndex: 6)
            command.setBytes(pSequence, atIndex: 7)
            command.setBuffer(layerPrev.delta.metal, atIndex: 8)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
