//
// Constant2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 19/02/2023.
//

/// Layer with a 2D shape neural structure and weights.
public class Constant2D: Layer2D, LayerResize, LayerUpdate
{
    ///
    /// Grid of weights.
    /// Shape ~ (nbChannels,).
    ///
    var _wArrays: WeightArrays! = nil
    
    ///
    /// Buffer of weights.
    /// Shape ~ (nbChannels,).
    ///
    var _wBuffers: IWeightBuffers! = nil
    
    ///
    /// Buffer of gradients per sample.
    /// Shape ~ (batch, nbChannels).
    ///
    var _wDeltaWeights: FloatBuffer! = nil
    
    /// Whether to compute weights' gradients or not.
    public var computeDeltaWeights: Bool = true
    
    /// Whether gradients of weights must be accumulated or not.
    public var accumulateDeltaWeights: Bool = false
    
    /// Cache for weights before calling `initKernel` API.
    var _weightsList = [Float]()
    
    /// Weights in the CPU execution context.
    public var weightsCPU: [Float]
    {
        get {
            if _wArrays == nil
            {
                return _weightsList
            }
            
            var weightsTmp = [Float]()
            for depth in 0..<nbChannels
            {
                weightsTmp.append(Float(_wArrays.w[depth]))
            }
            return weightsTmp
        }
        set {
            _weightsList = newValue
        }
    }
    
    /// Weights in the GPU execution context.
    public var weightsGPU: [Float]
    {
        get {
            if _wBuffers == nil
            {
                return _weightsList
            }
            return _wBuffers.w.download()
        }
        set {
            _weightsList = newValue
        }
    }
    
    /// Number of new weights due to this layer, estimated during the Gradient Checking.
    var nbLearnedGC: Int
    {
        get {
            return nbChannels
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case weights
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - nbChannels: Number of channels.
    ///     - height: Height of the output grids.
    ///     - width: Width of the output grids.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(nbChannels: Int,
                height: Int,
                width: Int,
                params: GrAI.Model.Params)
    {
        super.init(layerPrev: nil,
                   nbChannels: nbChannels,
                   height: height,
                   width: width,
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
        try super.init(from: decoder)
        
        let weightsList = try values.decode([Float].self, forKey: .weights)
        self.weightsCPU = weightsList
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
        
        let weightsList: [Float]
        if GrAI.Opti.GPU
        {
            weightsList = self.weightsGPU
        }
        else
        {
            weightsList = self.weightsCPU
        }
        try container.encode(weightsList, forKey: .weights)
        
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
            
        let layer = Constant2D(
            nbChannels: nbChannels,
            height: height,
            width: width,
            params: params
        )
        if inPlace
        {
            layer._wArrays = _wArrays
            layer._wBuffers = _wBuffers
        }
        else
        {
            if GrAI.Opti.GPU
            {
                layer.weightsGPU = weightsGPU
            }
            else
            {
                layer.weightsCPU = weightsCPU
            }
        }
        return layer
    }
    
    ///
    /// Resize this layer.
    ///
    /// - Parameters:
    ///     - imageWidth: New size width.
    ///     - imageHeight: New size height.
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///     - inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new instance of `Layer`. When `inPlace` is false, `initKernel` is
    ///  necessary in order to recreate hard resources.
    ///
    public func resize(
        imageWidth: Int,
        imageHeight: Int,
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = Constant2D(
            nbChannels: nbChannels,
            height: imageHeight,
            width: imageWidth,
            params: params
        )
        if inPlace
        {
            layer._wArrays = _wArrays
            layer._wBuffers = _wBuffers
        }
        else
        {
            if GrAI.Opti.GPU
            {
                layer.weightsGPU = weightsGPU
            }
            else
            {
                layer.weightsCPU = weightsCPU
            }
        }
        return layer
    }
    
    ///
    /// Clean state resources in the CPU execution context.
    ///
    /// We first clean the neurons' state (forward and backward).
    /// We do not clean weights and biases but must reset their delta (dependent on batch size) and
    /// momentum state.
    ///
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        _wArrays?.reset()
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We first clean the neurons' state (forward and backward).
    /// We do not clean weights and biases but must reset their delta (dependent on batch size) and
    /// momentum state.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        
        _wDeltaWeights = nil
        _wBuffers?.reset()
    }
    
    ///
    /// Initialize weights in the CPU execution context.
    ///
    /// Their momentum and delta state are also reset.
    ///
    public func initWeightsCPU()
    {
        _wArrays = WeightArrays(nbChannels)
        
        if _weightsList.count == 0
        {
            for depth in 0..<nbChannels
            {
                _wArrays.w[depth] = 0.0
            }
        }
        else
        {
            for depth in 0..<nbChannels
            {
                _wArrays.w[depth] = Double(_weightsList[depth])
            }
            _weightsList = []
        }
    }
    
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// Their momentum and delta state are also reset.
    ///
    public func initWeightsGPU()
    {
        _wBuffers = WeightBuffers(
            nbElems: nbChannels,
            deviceID: deviceID
        )
        
        if _weightsList.count != 0
        {
            _wBuffers.w.initialize(array: &_weightsList)
        }
        else
        {
            _wBuffers.w.initialize()
        }
        
        _weightsList = []
        _wDeltaWeights = nil
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    /// We initialize the weights and biases' delta.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if computeDeltaWeights &&
           GrAI.Gradient.sample && _wDeltaWeights == nil
        {
            _wDeltaWeights = FloatBuffer(nbElems: 
                batchSize * nbChannels, deviceID: deviceID
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
        try checkStateCPU(batchSize: batchSize)
        
        let newGC = 2 * nbLearnedGC
        for depth in 0..<nbChannels {
        for i in 0..<height {
        for j in 0..<width
        {
            neurons[depth].get(i, j)!.initGC(batchSize: batchSize, nbGC: newGC)
        }}}
        
        for batch in 0..<batchSize {
        for DEPTH in 0..<nbChannels {
        for elem in 0...1
        {
            for depth in 0..<nbChannels {
            for i in 0..<height {
            for j in 0..<width
            {
                var tmp: Double = _wArrays.w[depth]
                if depth == DEPTH
                {
                    if elem % 2 == 0
                    {
                        tmp += Ɛ
                    }
                    else
                    {
                        tmp -= Ɛ
                    }
                }
                
                let offset = 2 * DEPTH + elem
                neurons[depth].get(i, j)!.gc[batch][offset].out = tmp
            }}}
        }}}
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        let newGC = 2 * nbLearnedGC
        for depth in 0..<nbChannels {
        for i in 0..<height {
        for j in 0..<width
        {
            neurons[depth].get(i, j)!.initGC(batchSize: batchSize, nbGC: newGC)
        }}}
        
        let weightsPtr = _wBuffers.w.download()
    
        for batch in 0..<batchSize {
        for DEPTH in 0..<nbChannels {
        for elem in 0...1
        {
            for depth in 0..<nbChannels {
            for i in 0..<height {
            for j in 0..<width
            {
                var tmp: Double = Double(weightsPtr[depth])
                if depth == DEPTH
                {
                    if elem % 2 == 0
                    {
                        tmp += Ɛ
                    }
                    else
                    {
                        tmp -= Ɛ
                    }
                }
                
                let offset = 2 * DEPTH + elem
                neurons[depth].get(i, j)!.gc[batch][offset].out = tmp
            }}}
        }}}
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
        for depth in 0..<nbChannels {
        for i in 0..<height {
        for j in 0..<width
        {
            neurons[depth].get(i, j)!.v[elem].out = _wArrays.w[depth]
        }}}}
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        try checkStateForwardGPU(batchSize: batchSize)
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        let command = MetalKernel.get.createCommand(
            "constant2DForward", deviceID: deviceID
        )
        command.setBuffer(_wBuffers.w.metal, atIndex: 0)
        command.setBytes(pNbChannels, atIndex: 1)
        command.setBytes(pDimensions, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBuffer(outs.metal, atIndex: 4)
        
        command.dispatchThreads(
            width: width * nbChannels,
            height: height * batchSize
        )
        command.enqueue()
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if computeDeltaWeights
        {
            // -----------------------------------------------------------------
            // Compute Gradients per batch
            // -----------------------------------------------------------------
            for depth in 0..<nbChannels
            {
                var tmp: Double = 0.0
                for elem in 0..<batchSize {
                for i in 0..<height {
                for j in 0..<width
                {
                    let deltaCur = neurons[depth].get(i, j)!.v[elem].delta
                    tmp += deltaCur
                }}}
                
                if accumulateDeltaWeights
                {
                    tmp += _wArrays.g[depth]
                }
                _wArrays.g[depth] = tmp
            }
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        if computeDeltaWeights
        {
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pAccumulate: [UInt32] = accumulateDeltaWeights ? [1] : [0]
            
            var command: MetalCommand
            if GrAI.Gradient.batch
            {
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "convBatchDerBiases", deviceID: deviceID
                )
                command.setBuffer(delta.metal, atIndex: 0)
                command.setBytes(pNbChannels, atIndex: 1)
                command.setBytes(pDimensions, atIndex: 2)
                command.setBytes(pNbBatch, atIndex: 3)
                command.setBytes(pAccumulate, atIndex: 4)
                command.setBuffer(_wBuffers.g.metal, atIndex: 5)
                
                command.dispatchThreads(nbChannels)
                command.enqueue()
            }
            else
            {
                // -------------------------------------------------------------
                // Compute Gradients per sample
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "convDerBiases", deviceID: deviceID
                )
                command.setBuffer(delta.metal, atIndex: 0)
                command.setBytes(pNbChannels, atIndex: 1)
                command.setBytes(pDimensions, atIndex: 2)
                command.setBytes(pNbBatch, atIndex: 3)
                command.setBuffer(_wDeltaWeights.metal, atIndex: 4)
                
                command.dispatchThreads(
                    width: nbChannels,
                    height: batchSize
                )
                command.enqueue()
                
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "reduceBiases", deviceID: deviceID
                )
                command.setBuffer(_wDeltaWeights.metal, atIndex: 0)
                command.setBytes(pNbChannels, atIndex: 1)
                command.setBytes(pNbBatch, atIndex: 2)
                command.setBytes(pAccumulate, atIndex: 3)
                command.setBuffer(_wBuffers.g.metal, atIndex: 4)
                
                command.dispatchThreads(nbChannels)
                command.enqueue()
            }
        }
    }
    
    /// Get the weights in the CPU execution context.
    public func collectWeightsCPU() -> [IWeightArrays]
    {
        return [_wArrays]
    }
    
    /// Get the weights in the GPU execution context.
    public func collectWeightsGPU() -> [IWeightBuffers]
    {
        return [_wBuffers]
    }
    
    ///
    /// Get the weights' gradients in the GPU execution context.
    ///
    /// Throw an error when layer has not been updated through backward pass or
    /// when gradients per sample have not been computed.
    ///
    /// - Parameter elem: The batch element to retrieve the outputs from.
    ///
    public func getDeltaWeightsGPU<T: BinaryFloatingPoint>(elem: Int) throws
        -> [T]
    {
        if dirty
        {
            throw UpdateError.Dirty
        }
        if !GrAI.Gradient.sample
        {
            throw UpdateError.PerSample
        }
        
        var deltaWeights = [T]()
        let deltaWeightsPtr = _wDeltaWeights.download()
        
        for depth in 0..<nbChannels
        {
            let offset = depth + nbChannels * elem
            
            deltaWeights.append(T(
                deltaWeightsPtr[offset]
            ))
        }
        return deltaWeights
    }
    
    ///
    /// Get the weights' gradients in the CPU execution context.
    ///
    /// Throw an error when layer has not been updated through backward pass.
    ///
    public func getDeltaWeightsCPU<T: BinaryFloatingPoint>() throws -> [T]
    {
        if dirty
        {
            throw UpdateError.Dirty
        }
        
        var deltaWeights = [T]()
        for depth in 0..<nbChannels
        {
            deltaWeights.append(T(_wArrays.g[depth]))
        }
        return deltaWeights
    }
    
    ///
    /// Get the weights' gradients in the GPU execution context.
    ///
    /// Throw an error when layer has not been updated through backward pass.
    ///
    public func getDeltaWeightsGPU<T: BinaryFloatingPoint>() throws -> [T]
    {
        if dirty
        {
            throw UpdateError.Dirty
        }
        
        var deltaWeights = [T]()
        let deltaWeightsPtr = _wBuffers.g.download()
        
        for i in 0..<_wBuffers.nbElems
        {
            deltaWeights.append(T(deltaWeightsPtr[i]))
        }
        return deltaWeights
    }
}
