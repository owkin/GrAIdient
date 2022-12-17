//
// Input1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 10/10/2022.
//

/// Arrays needed to update the inputs of a layer.
class InputArrays1D: InputArrays<Layer1D>, IWeightArrays
{
    /// Inputs array: the array to update.
    var w: [Double]
    {
        get {
            var cur = 0
            var outs = [Double](repeating: 0.0, count: nbElems)
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbNeurons
            {
                outs[cur] = _layer.neurons.get(depth)!.v[elem].out
                cur += 1
            }}
            return outs
        }
        set {
            var cur = 0
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbNeurons
            {
                _layer.neurons.get(depth)!.v[elem].out = newValue[cur]
                cur += 1
            }}
        }
    }
    
    /// Gradients array.
    var g: [Double]
    {
        get {
            var cur = 0
            var delta = [Double](repeating: 0.0, count: nbElems)
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbNeurons
            {
                delta[cur] = _layer.neurons.get(depth)!.v[elem].delta
                cur += 1
            }}
            return delta
        }
        set {
            var cur = 0
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbNeurons
            {
                _layer.neurons.get(depth)!.v[elem].delta = newValue[cur]
                cur += 1
            }}
        }
    }
}

/// GPU buffers needed to update the inputs of a layer.
class InputBuffers1D: InputBuffers<Layer1D>, IWeightBuffers
{
    /// Inputs buffer: the buffer to be update.
    var w: MetalBuffer<Float>
    {
        get {
            return _layer.outs
        }
    }
    
    /// Gradients buffer.
    var g: MetalBuffer<Float>
    {
        get {
            return _layer.delta
        }
    }
}

/// First layer with a 1D shape neural structure.
public class Input1D: LayerInput1D, LayerUpdate
{
    /// Grid of "weights".
    var _wArrays: InputArrays1D! = nil
    /// Buffer of "weights".
    var _wBuffers: InputBuffers1D! = nil
    
    /// Whether to compute weights' gradients or not.
    public var computeDeltaWeights: Bool = false
    
    /// Whether gradients of weights must be accumulated or not.
    public var accumulateDeltaWeights: Bool = false
    
    /// Weights in the CPU execution context.
    public var weightsCPU: [Float]
    {
        get { [] }
        set {}
    }
    /// Weights in the GPU execution context.
    public var weightsGPU: [Float]
    {
        get { [] }
        set {}
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - nbNeurons: Number of neurons.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(nbNeurons: Int, params: GrAI.Model.Params)
    {
        super.init(layerPrev: nil,
                   nbNeurons: nbNeurons,
                   params: params)
        computeDelta = false
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer1D, params: GrAI.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbNeurons: layerPrev.nbNeurons,
                   params: params)
        computeDelta = false
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    required init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
        computeDelta = false
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
        
        let layer: Input1D
        if idPrev > -1
        {
            layer = Input1D(
                layerPrev: mapping[idPrev] as! Layer1D,
                params: params
            )
        }
        else
        {
            layer = Input1D(nbNeurons: nbNeurons, params: params)
        }
        return layer
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean weights.
    ///
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        _wArrays = nil
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean weights.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _wBuffers = nil
    }
    
    ///
    /// Initialize state resources in the CPU execution context.
    ///
    /// We initialize the neurons' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        try super.checkStateCPU(batchSize: batchSize)
        
        if _wArrays == nil && computeDeltaWeights
        {
            _wArrays = InputArrays1D(
                layer: self,
                nbElems: nbNeurons * batchSize
            )
        }
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    /// We initialize the weights.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if _wBuffers == nil && computeDeltaWeights
        {
            _wBuffers = InputBuffers1D(
                layer: self,
                nbElems: nbNeurons * batchSize,
                deviceID: deviceID
            )
        }
    }
    
    /// Initialize weights in the CPU execution context.
    public func initWeightsCPU() {}
    /// Initialize weights in the GPU execution context.
    public func initWeightsGPU() {}
    
    ///
    /// API to set data in the CPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameter data: The data to set.
    ///
    public func setDataCPU<T: BinaryFloatingPoint>(_ data: [[T]]) throws
    {
        let batchSize = data.count
        try checkStateCPU(batchSize: batchSize)
        
        for (elem, sample) in data.enumerated()
        {
            if sample.count != nbNeurons
            {
                throw LayerError.DataSize
            }
            
            for (i, feature) in sample.enumerated() {
            if let neuron = neurons.get(i)
            {
                neuron.v[elem].out = Double(feature)
            }}
        }
    }
    
    ///
    /// API to set data in the GPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameter data: The data to set.
    ///
    public func setDataGPU<T: BinaryFloatingPoint>(_ data: [[T]]) throws
    {
        let batchSize = data.count
        try checkStateForwardGPU(batchSize: batchSize)
        
        // Wait for previous loop to end to avoid race condition with
        // didModifyRange in the following example:
        // FullyConnected.backwardWeightsGPU accesses layerPrev.outs.
        MetalKernel.get.download([outs])
        
        if batchSize * nbNeurons != data.count * data.first!.count
        {
            throw LayerError.DataSize
        }
        
        let outsPtr = outs.shared.buffer
        for elem in 0..<batchSize
        {
            for depth in 0..<nbNeurons
            {
                let offset = depth + nbNeurons * elem
                outsPtr[offset] = Float(data[elem][depth])
            }
        }
        MetalKernel.get.upload([outs])
    }
    
    ///
    /// API to set data in the GPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The data to set.
    ///     - batchSize: The batch size of data.
    ///
    public func setDataGPU(
        _ data: MetalPrivateBuffer<Float>,
        batchSize: Int) throws
    {
        if batchSize * nbNeurons != data.nbElems
        {
            throw LayerError.DataSize
        }
        try checkStateForwardGPU(batchSize: batchSize)
        
        outs = data
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D, computeForward
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    neurons.get(depth)?.v[elem].out =
                        neuronsPrev.get(depth)!.v[elem].out
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
        if let layerPrev = self.layerPrev as? Layer1D, computeForward
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
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    let delta = neurons.get(depth)!.v[elem].delta
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(depth)!.v[elem].delta = delta
                    }
                    else
                    {
                        neuronsPrev.get(depth)!.v[elem].delta += delta
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
    
    /// Get the "weights" in the CPU execution context.
    public func collectWeightsCPU() -> [IWeightArrays]
    {
        return [_wArrays]
    }

    /// Get the "weights" in the GPU execution context.
    public func collectWeightsGPU() -> [IWeightBuffers]
    {
        return [_wBuffers]
    }
}
