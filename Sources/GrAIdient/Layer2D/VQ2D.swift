//
// VQ2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 29/03/2023.
//

import Foundation

/// Layer with a 2D shape neural structure and weights.
public class VQ2D: Layer2D, LayerUpdate
{
    /// The number of vector approximations.
    let _K: Int
    
    /// Coefficient for commitment.
    var _beta: Double
    
    ///
    /// Indices of maximal elements.
    /// Shape ~ (batch, height, width).
    ///
    var _indices: MetalBuffer<Int32>! = nil
    
    ///
    /// Grid of weights.
    /// Shape ~ (K, nbChannels).
    ///
    var _wArrays: WeightGrids! = nil
    
    ///
    /// Buffer of weights.
    /// Shape ~ (K, nbChannels).
    ///
    var _wBuffers: IWeightBuffers! = nil
    
    ///
    /// Buffer of gradients per sample for biases.
    /// Shape ~ (batch, K, nbChannels).
    ///
    var _wDeltaWeights: MetalPrivateBuffer<Float>! = nil
    
    /// Whether to forward vectors in the dictionary.
    public var computeVQ = true
    
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
            for k in 0..<_K {
            for depth in 0..<nbChannels
            {
                weightsTmp.append(Float(_wArrays.w(k, depth)))
            }}
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
            
            var weightsTmp = [Float]()
            MetalKernel.get.download([_wBuffers.w_p!])
            weightsTmp += _wBuffers.w_p!.shared.array
        
            return weightsTmp
        }
        set {
            _weightsList = newValue
        }
    }
    
    /// Get the coefficient to apply during the weights initialization.
    var coeffInitWeights: Double
    {
        get {
            return sqrt(2.0 / Double(_K + nbChannels))
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case K
        case beta
        case weights
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - K: The number of vector approximations.
    ///     - beta: Coefficient for commitment.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                K: Int,
                beta: Double,
                params: GrAI.Model.Params)
    {
        _K = K
        _beta = beta
        super.init(
            layerPrev: layerPrev,
            nbChannels: layerPrev.nbChannels,
            height: layerPrev.height,
            width: layerPrev.width,
            params: params
        )
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
        
        _K = try values.decode(Int.self, forKey: .K)
        _beta = try values.decode(Double.self, forKey: .beta)
        
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
        
        try container.encode(_K, forKey: .K)
        try container.encode(_beta, forKey: .beta)
        
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
        let layerPrev = mapping[idPrev] as! Layer2D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = VQ2D(
            layerPrev: layerPrev, K: _K, beta: _beta, params: params
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
        _indices = nil
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
        
        _indices = nil
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
        _wArrays = WeightGrids(width: nbChannels, height: _K)
        
        if _weightsList.count == 0
        {
            let coeff = coeffInitWeights
            for k in 0..<_K {
            for depth in 0..<nbChannels
            {
                _wArrays.w(k, depth, coeff * Double.random(in: -1..<1))
            }}
        }
        else
        {
            for k in 0..<_K {
            for depth in 0..<nbChannels
            {
                let offset = depth + nbChannels * k
                _wArrays.w(k, depth, Double(_weightsList[offset]))
            }}
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
            nbElems: _K * nbChannels,
            deviceID: deviceID
        )
        
        let weightsPtr = _wBuffers.w_p!.shared.buffer
        if _weightsList.count == 0
        {
            let coeff = Float(coeffInitWeights)
            for elem in 0..<_K * nbChannels
            {
                weightsPtr[elem] = coeff * Float.random(in: -1..<1)
            }
        }
        else
        {
            for elem in 0..<_K * nbChannels
            {
                weightsPtr[elem] = _weightsList[elem]
            }
            _weightsList = []
        }
        
        MetalKernel.get.upload([_wBuffers.w_p!])
        _wDeltaWeights = nil
    }
    
    ///
    /// Initialize state resources in the CPU execution context.
    ///
    /// We initialize the neurons' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        try super.checkStateCPU(batchSize: batchSize)
        
        if _indices == nil
        {
            _indices = MetalSharedBuffer<Int32>(
                batchSize * height * width,
                deviceID: deviceID
            )
        }
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
            _wDeltaWeights = MetalPrivateBuffer<Float>(
                batchSize * _K * nbChannels, deviceID: deviceID
            )
        }
        
        if _indices == nil
        {
            _indices = MetalPrivateBuffer<Int32>(
                batchSize * height * width,
                deviceID: deviceID
            )
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
            let indicesPtr = (_indices as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                var minIndex = -1
                var minValue: Double? = nil
                
                for k in 0..<_K
                {
                    var value: Double = 0.0
                    for depth in 0..<nbChannels
                    {
                        let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                        let vq = _wArrays.w(k, depth)
                        value += pow(outPrev - vq, 2.0)
                    }
                    value = sqrt(value)
                    
                    if minValue == nil || value < minValue!
                    {
                        minValue = value
                        minIndex = k
                    }
                }
                
                if minIndex < 0
                {
                    fatalError("'minIndex' is negative.")
                }
                
                if computeVQ
                {
                    for depth in 0..<nbChannels
                    {
                        neurons[depth].get(i, j)!.v[elem].out =
                            _wArrays.w(minIndex, depth)
                    }
                }
                else
                {
                    for depth in 0..<nbChannels
                    {
                        neurons[depth].get(i, j)!.v[elem].out =
                            neuronsPrev[depth].get(i, j)!.v[elem].out
                    }
                }
                
                indicesPtr[j + (elem * height + i) * width] = Int32(minIndex)
            }}}
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
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pK: [UInt32] = [UInt32(_K)]
            let pComputeVQ: [UInt32] = computeVQ ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "vq2DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(_wBuffers.w.metal, atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pDimensions, atIndex: 3)
            command.setBytes(pK, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pComputeVQ, atIndex: 6)
            command.setBuffer(outs.metal, atIndex: 7)
            command.setBuffer(_indices.metal, atIndex: 8)
            
            command.dispatchThreads(
                width: height * width,
                height: batchSize
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        _backwardCPU()
        _backwardWeightsCPU()
    }
    
    private func _backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            let indicesPtr = (_indices as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                let minIndex = Int(indicesPtr[j + (elem * height + i) * width])
                for depth in 0..<nbChannels
                {
                    let vq = _wArrays.w(minIndex, depth)
                    let deltaCur = neurons[depth].get(i, j)!.v[elem].delta
                    let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta =
                            deltaCur
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            deltaCur
                    }
                    
                    if computeVQ
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            _beta * (outPrev - vq)
                    }
                }
            }}}
            propagateDirty()
        }
    }
    
    private func _backwardWeightsCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeDeltaWeights
        {
            let neuronsPrev = layerPrev.neurons
            let coeff = batchSize * height * width
            let indicesPtr = (_indices as! MetalSharedBuffer<Int32>).buffer
            
            if !accumulateDeltaWeights
            {
                for k in 0..<_K {
                for depth in 0..<nbChannels
                {
                    _wArrays.g(k, depth, 0.0)
                }}
            }
            
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                let minIndex =
                    Int(indicesPtr[j + (elem * height + i) * width])
                for depth in 0..<nbChannels
                {
                    let vq = _wArrays.w(minIndex, depth)
                    let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                    
                    let g = _wArrays.g(minIndex, depth)
                    _wArrays.g(
                        minIndex, depth,
                        g + 1.0 / Double(coeff) * (vq - outPrev)
                    )
                }
            }}}
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        try _backwardGPU()
        _backwardWeightsGPU()
    }
    
    private func _backwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pK: [UInt32] = [UInt32(_K)]
            let pBeta: [Float] = [Float(_beta)]
            let pComputeVQ: [UInt32] = computeVQ ? [1] : [0]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "vq2DBackward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBuffer(_wBuffers.w.metal, atIndex: 2)
            command.setBuffer(_indices.metal, atIndex: 3)
            command.setBytes(pNbChannels, atIndex: 4)
            command.setBytes(pDimensions, atIndex: 5)
            command.setBytes(pK, atIndex: 6)
            command.setBytes(pBeta, atIndex: 7)
            command.setBytes(pNbBatch, atIndex: 8)
            command.setBytes(pComputeVQ, atIndex: 9)
            command.setBytes(pDirty, atIndex: 10)
            command.setBuffer(layerPrev.delta.metal, atIndex: 11)
            
            command.dispatchThreads(
                width: nbChannels * width,
                height: batchSize * height
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
    
    private func _backwardWeightsGPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeDeltaWeights
        {
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pK: [UInt32] = [UInt32(_K)]
            let pAccumulate: [UInt32] = accumulateDeltaWeights ? [1] : [0]
            
            var command: MetalCommand
            if GrAI.Gradient.batch
            {
                if !accumulateDeltaWeights
                {
                    let nbElems = _wBuffers.g.nbElems
                    let pNbElems: [UInt32] = [UInt32(nbElems)]
                    
                    command = MetalKernel.get.createCommand(
                        "reset", deviceID: deviceID
                    )
                    command.setBytes(pNbElems, atIndex: 0)
                    command.setBuffer(_wBuffers.g.metal, atIndex: 1)
                    
                    command.dispatchThreads(nbElems)
                    command.enqueue()
                }
                
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "vq2DBatchDerWeights", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(_wBuffers.w.metal, atIndex: 1)
                command.setBuffer(_indices.metal, atIndex: 2)
                command.setBytes(pNbChannels, atIndex: 3)
                command.setBytes(pDimensions, atIndex: 4)
                command.setBytes(pK, atIndex: 5)
                command.setBytes(pNbBatch, atIndex: 6)
                command.setBuffer(_wBuffers.g.metal, atIndex: 7)
                
                command.dispatchThreads(width: nbChannels, height: _K)
                command.enqueue()
            }
            else
            {
                let nbElems = _wDeltaWeights.nbElems
                let pNbElems: [UInt32] = [UInt32(nbElems)]
                
                command = MetalKernel.get.createCommand(
                    "reset", deviceID: deviceID
                )
                command.setBytes(pNbElems, atIndex: 0)
                command.setBuffer(_wDeltaWeights.metal, atIndex: 1)
                
                command.dispatchThreads(nbElems)
                command.enqueue()
                
                // -------------------------------------------------------------
                // Compute Gradients per sample
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "vq2DDerWeights", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(_wBuffers.w.metal, atIndex: 1)
                command.setBuffer(_indices.metal, atIndex: 2)
                command.setBytes(pNbChannels, atIndex: 3)
                command.setBytes(pDimensions, atIndex: 4)
                command.setBytes(pK, atIndex: 5)
                command.setBytes(pNbBatch, atIndex: 6)
                command.setBuffer(_wDeltaWeights.metal, atIndex: 7)
                
                command.dispatchThreads(
                    width: nbChannels,
                    height: batchSize * _K
                )
                command.enqueue()
                
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "vq2DReduceWeights", deviceID: deviceID
                )
                command.setBuffer(_wDeltaWeights.metal, atIndex: 0)
                command.setBytes(pNbChannels, atIndex: 1)
                command.setBytes(pK, atIndex: 2)
                command.setBytes(pNbBatch, atIndex: 3)
                command.setBytes(pAccumulate, atIndex: 4)
                command.setBuffer(_wBuffers.g.metal, atIndex: 5)
                
                command.dispatchThreads(width: nbChannels, height: _K)
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
}
