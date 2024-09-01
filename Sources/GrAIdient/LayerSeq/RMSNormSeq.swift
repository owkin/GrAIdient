//
// RMSNormSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 14/06/2024.
//

/// Layer with a sequential shape neural structure, an activation function and one layer normalization unit.
public class RMSNormSeq: ActivationSeq, LayerUpdate, LayerWithActivation
{
    /// Instance normalization by default or instance normalization in the CPU execution context.
    var _norm: LayerWeightsNormalization? = nil
    /// Instance normalization in the GPU execution context.
    var _normGPU: RMSNormalizationGPU? = nil
    
    /// Whether to add unit offset or not.
    var addUnitOffset: Bool
    
    /// Whether to compute weights' gradients or not.
    public var computeDeltaWeights: Bool = true
    
    /// Whether gradients of weights must be accumulated or not.
    public var accumulateDeltaWeights: Bool = false
    
    /// Weights in the CPU execution context.
    public var weightsCPU: [Float]
    {
        get {
            var weightsTmp = [Float]()
            if let norm = _norm
            {
                weightsTmp += norm.weights
            }
            return weightsTmp
        }
        set {
            if let norm = _norm
            {
                norm.weights = newValue
            }
        }
    }
    
    /// Weights in the GPU execution context.
    public var weightsGPU: [Float]
    {
        get {
            var weightsTmp = [Float]()
            if let norm = _normGPU
            {
                weightsTmp += norm.weights
            }
            else if let norm = _norm
            {
                weightsTmp += norm.weights
            }
            return weightsTmp
        }
        set {
            if let norm = _normGPU
            {
                norm.weights = newValue
            }
            else if let norm = _norm
            {
                norm.weights = newValue
            }
        }
    }
    
    /// Get instance normalization in the CPU execution context.
    var norm: RMSNormalization?
    {
        get {
            return _norm as? RMSNormalization
        }
    }
    
    /// Number of new weights due to this layer, estimated during the Gradient Checking.
    var nbLearnedGC: Int
    {
        get {
            return nbNeurons
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case norm
        case addUnitOffset
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - activation: The activation function.
    ///     - addUnitOffset: Whether to add unit offset or not.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq,
                activation: String?,
                addUnitOffset: Bool,
                params: GrAI.Model.Params)
    {
        self.addUnitOffset = addUnitOffset
        
        super.init(layerPrev: layerPrev,
                   sequence: layerPrev.sequence,
                   nbNeurons: layerPrev.nbNeurons,
                   activation: activation,
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
        addUnitOffset = try values.decode(Bool.self, forKey: .addUnitOffset)
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
        try container.encode(addUnitOffset, forKey: .addUnitOffset)
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
        let layerPrev = mapping[idPrev] as! LayerSeq
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = RMSNormSeq(
            layerPrev: layerPrev,
            activation: _activation?.name,
            addUnitOffset: addUnitOffset,
            params: params
        )
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
    /// Extract main operation of this layer without the activation part.
    ///
    /// This API will create a new layer in the same context as this.
    ///
    /// - Parameter inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new instance of `Layer`. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    public func removeActivation(inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = self.layerPrev as! LayerSeq
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = RMSNormSeq(
            layerPrev: layerPrev,
            activation: nil,
            addUnitOffset: addUnitOffset,
            params: params
        )
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
    /// Extract main operation of this layer without the activation part.
    ///
    /// - Parameter params: Contextual parameters linking to the model.
    ///
    /// - Returns: A new layer.
    ///
    public func removeActivation(params: GrAI.Model.Params) -> Layer
    {
        let layerPrev = self.layerPrev as! LayerSeq
        let layer = RMSNormSeq(
            layerPrev: layerPrev,
            activation: nil,
            addUnitOffset: addUnitOffset,
            params: params
        )
        // only one of them should be cloned
        if let norm = _normGPU
        {
            layer._norm = norm.clone()
        }
        else if let norm = _norm
        {
            layer._norm = norm.clone()
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
            _norm = RMSNormalization(norm: norm)
        }
        else if let norm = _norm
        {
            _norm = RMSNormalization(norm: norm)
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
            _normGPU = RMSNormalizationGPU(norm: norm)
        }
        else if let norm = _norm
        {
            _normGPU = RMSNormalizationGPU(norm: norm)
        }
        _normGPU?.initKernel(deviceID: deviceID)
        
        if !GrAI.Loop.gradientChecking
        {
            _norm = nil
        }
    }
    
    ///
    /// Initialize weights in the CPU execution context.
    ///
    /// We initialize batch normalization's weights.
    ///
    public func initWeightsCPU()
    {
        norm?.initWeights()
    }
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// We initialize batch normalization's weights.
    ///
    public func initWeightsGPU()
    {
        _normGPU?.initWeights()
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        try _forwardGCCPU()
        norm!.forwardGC(self)
        _activation?.forwardGC(self)
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    private func _forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            for seq in 0..<sequence {
            for depth in 0..<nbNeurons
            {
                neurons.get(seq, depth)!.initGC(
                    batchSize: batchSize, nbGC: newGC
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
            
            // Prepare GC for norm weights: Ɣ and β.
            for batch in 0..<batchSize {
            for seq in 0..<sequence {
            for elem in newGC-2*nbNeurons..<newGC
            {
                for depth in 0..<nbNeurons
                {
                    neurons.get(seq, depth)!.gc[batch][elem].out =
                        neuronsPrev.get(seq, depth)!.v[batch].out
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
        try _forwardGCGPU()
        norm!.forwardFlowGC(self)
        _activation?.forwardGC(self)
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    private func _forwardGCGPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            for seq in 0..<sequence {
            for depth in 0..<nbNeurons
            {
                neurons.get(seq, depth)!.initGC(
                    batchSize: batchSize, nbGC: newGC
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
            
            let outsPrevPtr = layerPrev.outs.download()
            
            // Prepare GC for norm weights: Ɣ and β.
            for batch in 0..<batchSize {
            for seq in 0..<sequence {
            for elem in newGC-2*nbNeurons..<newGC
            {
                for depth in 0..<nbNeurons
                {
                    let offset = depth + nbNeurons * seq +
                        sequence * nbNeurons * batch
                    
                    neurons.get(seq, depth)!.gc[batch][elem].out =
                        Double(outsPrevPtr[offset])
                }
            }}}
        }
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
            
            norm!.forward(self)
            _activation?.forwardCPU(self)
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
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems / coeff)
            command.enqueue()
            
            _normGPU!.forward(self)
            _activation?.forwardGPU(self)
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        _activation?.backwardCPU(self)
        norm!.backward(self)
        
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
        _activation?.backwardGPU(self)
        _normGPU!.backward(self)
        
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
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(layerPrev.delta.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems / coeff)
            command.enqueue()
            
            propagateDirty()
        }
    }
    
    /// Get the weights in the CPU execution context.
    public func collectWeightsCPU() -> [IWeightArrays]
    {
        var weights = [IWeightArrays]()
        if let norm = self.norm
        {
            weights += norm.collectWeights()
        }
        return weights
    }
    
    /// Get the weights in the GPU execution context.
    public func collectWeightsGPU() -> [IWeightBuffers]
    {
        return _normGPU!.collectWeights()
    }
    
    ///
    /// Get the outputs of Gradient Checking (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - batch: Index of sample in the mini batch.
    ///     - seq: Index of the sequence.
    ///     - elem: Weight estimation index during the Gradient Checking.
    /// - Returns: The outputs.
    ///
    func getOutsGC(batch: Int, seq: Int, elem: Int) -> [Double]
    {
        var outs = [Double](repeating: 0.0, count: nbNeurons)
        for depth in 0..<nbNeurons
        {
            outs[depth] = neurons.get(seq, depth)!.gc[batch][elem].out
        }
        return outs
    }
    
    ///
    /// Set the outputs of Gradient Checking (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - batch: Index sample in the mini batch.
    ///     - seq: Index of the sequence.
    ///     - elem: Weight estimation index during the Gradient Checking.
    ///     - outs: The outputs to set.
    ///
    func setOutsGC(batch: Int, seq: Int, elem: Int, outs: [Double])
    {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.gc[batch][elem].out = outs[depth]
        }
    }
    
    ///
    /// Get the outputs (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - batch: Index sample in the mini batch.
    ///     - seq: Index of the sequence.
    /// - Returns: The outputs.
    ///
    func getOuts(batch: Int, seq: Int) -> [Double]
    {
        var outs = [Double](repeating: 0.0, count: nbNeurons)
        for depth in 0..<nbNeurons
        {
            outs[depth] = neurons.get(seq, depth)!.v[batch].out
        }
        return outs
    }
    
    ///
    /// Set the outputs (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - batch: Index sample in the mini batch.
    ///     - seq: Index of the sequence.
    ///     - outs: The outputs to set.
    ///
    func setOuts(batch: Int, seq: Int, outs: [Double])
    {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.v[batch].out = outs[depth]
        }
    }
    
    ///
    /// Get the gradients (result of the backward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - batch: Index sample in the mini batch.
    ///     - seq: Index of the sequence.
    /// - Returns: The gradients.
    ///
    func getDelta(batch: Int, seq: Int) -> [Double]
    {
        var delta = [Double](repeating: 0.0, count: nbNeurons)
        for depth in 0..<nbNeurons
        {
            delta[depth] = neurons.get(seq, depth)!.v[batch].delta
        }
        return delta
    }
    
    ///
    /// Set the gradients (result of the backward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - batch: Index sample in the mini batch.
    ///     - seq: Index of the sequence.
    ///     - delta: The gradients to set.
    ///
    func setDelta(batch: Int, seq: Int, delta: [Double])
    {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.v[batch].delta = delta[depth]
        }
    }
}
