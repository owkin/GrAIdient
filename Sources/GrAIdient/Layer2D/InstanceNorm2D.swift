//
// InstanceNorm2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 17/02/2023.
//

/// Layer with a 2D shape neural structure, an activation function and instance normalization units.
public class InstanceNorm2D: Activation2D, LayerUpdate, LayerWithActivation
{
    /// Instance normalization by default or instance normalization in the CPU execution context.
    var _norm: LayerWeightsNormalization? = nil
    /// Instance normalization in the GPU execution context.
    var _normGPU: InstanceNormalizationGPU? = nil
    
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
    var norm: InstanceNormalization?
    {
        get {
            return _norm as? InstanceNormalization
        }
    }
    
    /// Number of new weights due to this layer, estimated during the Gradient Checking.
    var nbLearnedGC: Int
    {
        get {
            return 2 * nbChannels
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
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - activation: The activation function.
    ///     - params: Contextual parameters linking to the model.
    ///
    public override init(layerPrev: Layer2D, activation: String?,
                         params: GrAI.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbChannels: layerPrev.nbChannels,
                   height: layerPrev.height,
                   width: layerPrev.width,
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
        let layerPrev = mapping[idPrev] as! Layer2D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = InstanceNorm2D(
            layerPrev: layerPrev,
            activation: _activation?.name,
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
        let layerPrev = self.layerPrev as! Layer2D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = InstanceNorm2D(
            layerPrev: layerPrev,
            activation: nil,
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
        let layerPrev = self.layerPrev as! Layer2D
        let layer = InstanceNorm2D(
            layerPrev: layerPrev,
            activation: nil,
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
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            for j in 0..<nbChannels
            {
                neurons[j].initGC(batchSize: batchSize, nbGC: newGC)
            }
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbChannels {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.gc[batch][elem].out =
                        neuronsPrev[depth].get(i, j)!.gc[batch][elem].out
                }}}
            }}
            
            // Prepare GC for norm weights: Ɣ and β.
            for batch in 0..<batchSize {
            for elem in newGC-4*nbChannels..<newGC
            {
                for depth in 0..<nbChannels {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.gc[batch][elem].out =
                        neuronsPrev[depth].get(i, j)!.v[batch].out
                }}}
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
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            for j in 0..<nbChannels
            {
                neurons[j].initGC(batchSize: batchSize, nbGC: newGC)
            }
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbChannels {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.gc[batch][elem].out =
                        neuronsPrev[depth].get(i, j)!.gc[batch][elem].out
                }}}
            }}
            
            MetalKernel.get.download([layerPrev.outs])
            let outsPrevPtr = layerPrev.outs.shared.buffer
            
            // Prepare GC for norm weights: Ɣ and β.
            for batch in 0..<batchSize {
            for elem in newGC-4*nbChannels..<newGC
            {
                for depth in 0..<nbChannels
                {
                    let offsetStart =
                        (depth + nbChannels * batch) * height
                    
                    for i in 0..<height {
                    for j in 0..<width
                    {
                        let offset = j + (offsetStart + i) * width
                        
                        neurons[depth].get(i, j)!.gc[batch][elem].out =
                            Double(outsPrevPtr[offset])
                    }}
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
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for depth in 0..<nbChannels {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.v[elem].out =
                        neuronsPrev[depth].get(i, j)!.v[elem].out
                }}}
            }
            
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
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let nbElems = outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let kernel = nbElems % 4 == 0 ? "sum14" : "sum1"
            let coeff = nbElems % 4 == 0 ? 4 : 1
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal(), atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal(), atIndex: 2)
            
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
        
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for depth in 0..<nbChannels {
                for i in 0..<height {
                for j in 0..<width
                {
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta =
                            neurons[depth].get(i, j)!.v[elem].delta
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            neurons[depth].get(i, j)!.v[elem].delta
                    }
                }}}
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
        _activation?.backwardGPU(self)
        _normGPU!.backward(self)
        
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
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
            
            command.setBuffer(delta.metal(), atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(layerPrev.delta.metal(), atIndex: 2)
            
            command.dispatchThreads(nbElems / coeff)
            command.enqueue()
            
            propagateDirty()
        }
    }
    
    /// Get the weights in the CPU execution context.
    public func collectWeightsCPU() -> [IWeightArrays]
    {
        var weights = [WeightArrays]()
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
    ///     - depth: Channel index.
    ///     - batch: Index of sample in the mini batch.
    ///     - elem: Weight estimation index during the Gradient Checking.
    /// - Returns: The outputs.
    ///
    func getOutsGC(depth: Int, batch: Int, elem: Int) -> [Double]
    {
        var outs = [Double](repeating: 0.0, count: height * width)
        for i in 0..<height {
        for j in 0..<width
        {
            let offset = j + i * width
            outs[offset] = neurons[depth].get(i, j)!.gc[batch][elem].out
        }}
        return outs
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
    /// Get the outputs (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index sample in the mini batch.
    /// - Returns: The outputs.
    ///
    func getOuts(depth: Int, batch: Int) -> [Double]
    {
        var outs = [Double](repeating: 0.0, count: height * width)
        for i in 0..<height {
        for j in 0..<width
        {
            let offset = j + i * width
            outs[offset] = neurons[depth].get(i, j)!.v[batch].out
        }}
        return outs
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
    /// Set the gradients (result of the backward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - batch: Index sample in the mini batch.
    ///     - delta: The gradients to set.
    ///
    func setDelta(depth: Int, batch: Int, delta: [Double])
    {
        for i in 0..<height {
        for j in 0..<width
        {
            let offset = j + i * width
            neurons[depth].get(i, j)!.v[batch].delta = delta[offset]
        }}
    }
}
