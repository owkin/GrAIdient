//
// BN2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

/// Layer with a 2D shape neural structure, an activation function and batch normalization units.
public class BN2D: Activation2D, LayerUpdate, LayerWithActivation
{
    /// Batch normalization by default or batch normalization in the CPU execution context.
    var _bn: BatchNormalizationBase? = nil
    /// Batch normalization in the GPU execution context.
    var _bnGPU: BatchNormalizationGPU? = nil
    
    /// Whether to compute weights' gradients or not.
    public var computeDeltaWeights: Bool = true
    
    /// Whether gradients of weights must be accumulated or not.
    public var accumulateDeltaWeights: Bool = false
    
    /// Weights in the CPU execution context.
    public var weightsCPU: [Float]
    {
        get {
            var weightsTmp = [Float]()
            if let bn = _bn
            {
                weightsTmp += bn.weights
            }
            return weightsTmp
        }
        set {
            if let bn = _bn
            {
                bn.weights = newValue
            }
        }
    }
    
    /// Weights in the GPU execution context.
    public var weightsGPU: [Float]
    {
        get {
            var weightsTmp = [Float]()
            if let bn = _bnGPU
            {
                weightsTmp += bn.weights
            }
            else if let bn = _bn
            {
                weightsTmp += bn.weights
            }
            return weightsTmp
        }
        set {
            if let bn = _bnGPU
            {
                bn.weights = newValue
            }
            else if let bn = _bn
            {
                bn.weights = newValue
            }
        }
    }
    
    /// Stats in the CPU execution context.
    public var statsCPU: [Float]
    {
        get {
            var statsTmp = [Float]()
            if let bn = _bn
            {
                statsTmp += bn.stats
            }
            return statsTmp
        }
        set {
            if let bn = _bn
            {
                bn.stats = newValue
            }
        }
    }
    
    /// Stats in the GPU execution context.
    public var statsGPU: [Float]
    {
        get {
            var statsTmp = [Float]()
            if let bn = _bnGPU
            {
                statsTmp += bn.stats
            }
            else if let bn = _bn
            {
                statsTmp += bn.stats
            }
            return statsTmp
        }
        set {
            if let bn = _bnGPU
            {
                bn.stats = newValue
            }
            else if let bn = _bn
            {
                bn.stats = newValue
            }
        }
    }
    
    /// Get batch normalization in the CPU execution context.
    var bn: BatchNormalization?
    {
        get {
            return _bn as? BatchNormalization
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
        case BN = "BatchNormalization"
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
        
        _bn = BatchNormalizationBase(self)
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - activation: The activation function.
    ///     - bn: Whether to use batch normalization or not.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                nbChannels: Int, height: Int, width: Int,
                activation: String?, bn: Bool,
                params: GrAI.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbChannels: nbChannels,
                   height: height,
                   width: width,
                   activation: activation,
                   params: params)
        if bn
        {
            _bn = BatchNormalizationBase(self)
        }
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
        _bn = try values.decodeIfPresent(BatchNormalizationBase.self,
                                         forKey: .BN)
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
        if let bn = _bnGPU
        {
            try container.encode(bn, forKey: Keys.BN)
        }
        else if let bn = _bn
        {
            try container.encode(bn, forKey: Keys.BN)
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
            
        let layer = BN2D(
            layerPrev: layerPrev,
            activation: _activation?.name,
            params: params
        )
        if inPlace
        {
            layer._bn = _bn
            layer._bnGPU = _bnGPU
        }
        else
        {
            // only one of them should be cloned
            if let bn = _bnGPU
            {
                layer._bn = bn.clone()
            }
            else if let bn = _bn
            {
                layer._bn = bn.clone()
            }
        }
        return layer
    }
    
    ///
    /// Extract main operation of this layer.
    ///
    /// This API will create a new layer in the same context as this.
    ///
    /// - Parameter inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new instance of `Layer`. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    public func extractActivation(inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = self.layerPrev as! Layer2D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = BN2D(
            layerPrev: layerPrev,
            activation: nil,
            params: params
        )
        if inPlace
        {
            layer._bn = _bn
            layer._bnGPU = _bnGPU
        }
        else
        {
            // only one of them should be cloned
            if let bn = _bnGPU
            {
                layer._bn = bn.clone()
            }
            else if let bn = _bn
            {
                layer._bn = bn.clone()
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
    public func extractActivation(params: GrAI.Model.Params) -> Layer
    {
        let layerPrev = self.layerPrev as! Layer2D
        let layer = BN2D(
            layerPrev: layerPrev,
            activation: nil,
            params: params
        )
        // only one of them should be cloned
        if let bn = _bnGPU
        {
            layer._bn = bn.clone()
        }
        else if let bn = _bn
        {
            layer._bn = bn.clone()
        }
        return layer
    }
    
    ///
    /// Create the activation part ot this layer.
    ///
    /// This API will create a new layer in the same context as this.
    ///
    /// - Returns: A new activation layer.
    ///
    public func createActivation() -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = self.layerPrev as! Layer2D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        return Activation2D(
            layerPrev: layerPrev,
            activation: _activation!.name,
            params: params
        )
    }
    
    ///
    /// Create the activation part ot this layer.
    ///
    /// - Parameter params: Contextual parameters linking to the model.
    ///
    /// - Returns: A new activation layer.
    ///
    public func createActivation(params: GrAI.Model.Params) -> Layer
    {
        let layerPrev = self.layerPrev as! Layer2D
        return Activation2D(
            layerPrev: layerPrev,
            activation: _activation!.name,
            params: params
        )
    }
    
    ///
    /// Clean state resources in the CPU execution context.
    ///
    /// We reset batch normalization.
    ///
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        bn?.resetKernel()
    }
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We reset batch normalization.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _bnGPU?.resetKernel()
    }
    
    ///
    /// Initialize hard resources in the CPU execution context.
    ///
    /// We initialize batch normalization.
    ///
    public override func initKernelCPU()
    {
        super.initKernelCPU()
        
        if let bn = _bnGPU
        {
            _bn = BatchNormalization(bn: bn)
        }
        else if let bn = _bn
        {
            _bn = BatchNormalization(bn: bn)
        }
        bn?.initKernel()
        
        if !GrAI.Loop.gradientChecking
        {
            _bnGPU = nil
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
        
        if let bn = _bnGPU
        {
            _bnGPU = BatchNormalizationGPU(bn: bn)
        }
        else if let bn = _bn
        {
            _bnGPU = BatchNormalizationGPU(bn: bn)
        }
        _bnGPU?.initKernel(deviceID: deviceID)
        
        if !GrAI.Loop.gradientChecking
        {
            _bn = nil
        }
    }
    
    ///
    /// Initialize weights in the CPU execution context.
    ///
    /// We initialize batch normalization's weights.
    ///
    public func initWeightsCPU()
    {
        bn?.initWeights()
    }
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// We initialize batch normalization's weights.
    ///
    public func initWeightsGPU()
    {
        _bnGPU?.initWeights()
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        try _forwardGCCPU()
        bn!.forwardGC(self)
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
            
            // Prepare GC for BN weights: Ɣ and β.
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
        bn!.forwardFlowGC(self)
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
            
            // Prepare GC for BN weights: Ɣ and β.
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
            
            bn!.forward(self)
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
            
            let command = MetalKernel.get.createCommand(
                "sum1", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
            
            _bnGPU!.forward(self)
            _activation?.forwardGPU(self)
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        _activation?.backwardCPU(self)
        bn!.backward(self)
        
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
        _bnGPU!.backward(self)
        
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command: MetalCommand
            if layerPrev.dirty
            {
                command = MetalKernel.get.createCommand(
                    "sum1", deviceID: deviceID
                )
            }
            else
            {
                command = MetalKernel.get.createCommand(
                    "sum2", deviceID: deviceID
                )
            }
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(layerPrev.delta.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
            
            propagateDirty()
        }
    }
    
    /// Get the weights in the CPU execution context.
    public func collectWeightsCPU() -> [IWeightArrays]
    {
        var weights = [IWeightArrays]()
        if let bn = self.bn
        {
            weights += bn.collectWeights()
        }
        return weights
    }
    
    /// Get the weights in the GPU execution context.
    public func collectWeightsGPU() -> [IWeightBuffers]
    {
        return _bnGPU!.collectWeights()
    }
    
    ///
    /// Get the outputs of Gradient Checking (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - elem: Weight estimation index during the Gradient Checking.
    /// - Returns: The outputs.
    ///
    func getOutsGC(depth: Int, elem: Int) -> [Double]
    {
        var sorties = [Double](repeating: 0.0,
                               count: batchSize * height * width)
        
        for batch in 0..<batchSize
        {
            let offsetStart = batch * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                sorties[offset] =
                    neurons[depth].get(i, j)!.gc[batch][elem].out
            }}
        }
        
        return sorties
    }
    
    ///
    /// Set the outputs of Gradient Checking (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - elem: Weight estimation index during the Gradient Checking.
    ///     - outs: The outputs to set.
    ///
    func setOutsGC(depth: Int, elem: Int, outs: [Double])
    {
        for batch in 0..<batchSize
        {
            let offsetStart = batch * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                neurons[depth].get(i, j)!.gc[batch][elem].out = outs[offset]
            }}
        }
    }
    
    ///
    /// Get the outputs (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameter depth: Channel index.
    /// - Returns: The outputs.
    ///
    func getOuts(_ depth: Int) -> [Double]
    {
        var outs = [Double](repeating: 0.0,
                            count: batchSize * height * width)
        for elem in 0..<batchSize
        {
            let offsetStart = elem * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                outs[offset] = neurons[depth].get(i, j)!.v[elem].out
            }}
        }
        return outs
    }
    
    ///
    /// Set the outputs (result of the forward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - outs: The outputs to set.
    ///
    func setOuts(depth: Int, outs: [Double])
    {
        for elem in 0..<batchSize
        {
            let offsetStart = elem * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                neurons[depth].get(i, j)!.v[elem].out = outs[offset]
            }}
        }
    }
    
    ///
    /// Get the gradients (result of the backward pass) in the CPU execution context.
    ///
    /// - Parameter depth: Channel index.
    /// - Returns: The gradients.
    ///
    func getDelta(_ depth: Int) -> [Double]
    {
        var delta = [Double](repeating: 0.0,
                             count: batchSize * height * width)
        for elem in 0..<batchSize
        {
            let offsetStart = elem * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                delta[offset] = neurons[depth].get(i, j)!.v[elem].delta
            }}
        }
        return delta
    }
    
    ///
    /// Set the gradients (result of the backward pass) in the CPU execution context.
    ///
    /// - Parameters:
    ///     - depth: Channel index.
    ///     - outs: The gradients to set.
    ///
    func setDelta(depth: Int, delta: [Double])
    {
        for elem in 0..<batchSize
        {
            let offsetStart = elem * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                neurons[depth].get(i, j)!.v[elem].delta = delta[offset]
            }}
        }
    }
}
