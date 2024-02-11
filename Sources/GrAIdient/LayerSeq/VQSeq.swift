//
// VQSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 18/06/2023.
//

import Foundation
import MetalKit

/// Layer with a sequential shape neural structure and weights.
public class VQSeq: LayerSeq, LayerWeightInit
{
    /// The number of vector approximations.
    public let K: Int
    
    /// Coefficient to be applied to the loss computation.
    public var coeff: Double = 1.0
    /// Coefficient for commitment.
    public var beta: Double = 1.0
    
    ///
    /// Loss buffer in the GPU execution context.
    /// Shape ~ (batch,).
    ///
    public internal(set) var loss: MetalSharedBuffer<Float>! = nil
    ///
    /// Indices of maximal elements.
    /// Shape ~ (batch, seq).
    ///
    public var indices: MetalBuffer<Int32>! = nil
    
    ///
    /// Grid of weights.
    /// Shape ~ (K, nbNeurons).
    ///
    var _wArrays: WeightGrids! = nil
    
    ///
    /// Buffer of weights.
    /// Shape ~ (K, nbNeurons).
    ///
    var _wBuffers: IWeightBuffers! = nil
    
    ///
    /// Buffer of gradients per sample for biases.
    /// Shape ~ (batch, K, nbNeurons).
    ///
    var _wDeltaWeights: MetalPrivateBuffer<Float>! = nil
    
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
            for k in 0..<K {
            for depth in 0..<nbNeurons
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
    
    /// Method used to initialize weights values.
    public var weightInitClass: WeightInitClass = .XavierUniform
    
    /// Get the number of input and output connections.
    public var connectivityIO: (Int, Int)
    {
        get {
            return (nbNeurons, K)
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case K
        case coeff
        case beta
        case weights
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - K: The number of vector approximations.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq,
                K: Int,
                params: GrAI.Model.Params)
    {
        self.K = K
        super.init(layerPrev: layerPrev,
                   sequence: layerPrev.sequence,
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
        
        K = try values.decode(Int.self, forKey: .K)
        coeff = try Double(values.decode(Float.self, forKey: .coeff))
        beta = try Double(values.decode(Float.self, forKey: .beta))
        
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
        
        try container.encode(K, forKey: .K)
        try container.encode(Float(coeff), forKey: .coeff)
        try container.encode(Float(beta), forKey: .beta)
        
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
        let layerPrev = mapping[idPrev] as! LayerSeq
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = VQSeq(
            layerPrev: layerPrev, K: K, params: params
        )
        layer.coeff = coeff
        layer.beta = beta
        
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
        indices = nil
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
        
        indices = nil
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
        if _weightsList.count == 0
        {
            _weightsList = generateWeightsList()
        }
        
        _wArrays = WeightGrids(width: nbNeurons, height: K)
        
        for k in 0..<K {
        for depth in 0..<nbNeurons
        {
            let offset = depth + nbNeurons * k
            _wArrays.w(k, depth, Double(_weightsList[offset]))
        }}
        _weightsList = []
    }
    
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// Their momentum and delta state are also reset.
    ///
    public func initWeightsGPU()
    {
        if _weightsList.count == 0
        {
            _weightsList = generateWeightsList()
        }
        
        _wBuffers = WeightBuffers(
            nbElems: K * nbNeurons,
            deviceID: deviceID
        )
        
        let weightsPtr = _wBuffers.w_p!.shared.buffer
        for elem in 0..<K * nbNeurons
        {
            weightsPtr[elem] = _weightsList[elem]
        }
        _weightsList = []
        
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
        
        if indices == nil
        {
            indices = MetalSharedBuffer<Int32>(
                batchSize * sequence,
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
                batchSize * K * nbNeurons, deviceID: deviceID
            )
        }
        
        if indices == nil
        {
            indices = MetalPrivateBuffer<Int32>(
                batchSize * sequence,
                deviceID: deviceID
            )
        }
    }
    
    ///
    /// Setup loss state  in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter batchSize: The batch size of data.
    ///
    public func checkLossGPU(batchSize: Int) throws
    {
        if loss == nil
        {
            loss = MetalSharedBuffer<Float>(batchSize, deviceID: deviceID)
        }
        else if batchSize <= 0 || batchSize > loss.nbElems
        {
            throw LayerError.BatchSize
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
            let indicesPtr = (indices as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                var minIndex = -1
                var minValue: Double? = nil
                
                for k in 0..<K
                {
                    var value: Double = 0.0
                    for depth in 0..<nbNeurons
                    {
                        let outPrev = neuronsPrev.get(seq, depth)!.v[elem].out
                        let vq = _wArrays.w(k, depth)
                        value += pow(outPrev - vq, 2.0)
                    }
                    
                    if minValue == nil || value < minValue!
                    {
                        minValue = value
                        minIndex = k
                    }
                }
                
                if minIndex < 0
                {
                    throw VQError.IndexValue
                }
                
                for depth in 0..<nbNeurons
                {
                    neurons.get(seq, depth)!.v[elem].out =
                        _wArrays.w(minIndex, depth)
                }
                indicesPtr[seq + elem * sequence] = Int32(minIndex)
            }}
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
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pK: [UInt32] = [UInt32(K)]
            
            let command = MetalKernel.get.createCommand(
                "vqSeqForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(_wBuffers.w.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pK, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            command.setBuffer(indices.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: sequence,
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
    
    fileprivate func _backwardCPU()
    {
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons!
            let indicesPtr = (indices as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                let minIndex = Int(indicesPtr[seq + elem * sequence])
                if minIndex >= 0 {
                for depth in 0..<nbNeurons
                {
                    let vq = _wArrays.w(minIndex, depth)
                    let deltaCur = neurons.get(seq, depth)!.v[elem].delta
                    let outPrev = neuronsPrev.get(seq, depth)!.v[elem].out
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(seq, depth)!.v[elem].delta = deltaCur
                    }
                    else
                    {
                        neuronsPrev.get(seq, depth)!.v[elem].delta += deltaCur
                    }
                    
                    // Commitment term.
                    neuronsPrev.get(seq, depth)!.v[elem].delta +=
                        beta / Double(batchSize * sequence) *
                        2.0 * (outPrev - vq)
                }}
                else if layerPrev.dirty {
                for depth in 0..<nbNeurons
                {
                    neuronsPrev.get(seq, depth)!.v[elem].delta = 0.0
                }}
            }}
            propagateDirty()
        }
    }
    
    fileprivate func _backwardWeightsCPU()
    {
        if let layerPrev = self.layerPrev as? LayerSeq, computeDeltaWeights
        {
            let neuronsPrev = layerPrev.neurons!
            let indicesPtr = (indices as! MetalSharedBuffer<Int32>).buffer
            
            if !accumulateDeltaWeights
            {
                for k in 0..<K {
                for depth in 0..<nbNeurons
                {
                    _wArrays.g(k, depth, 0.0)
                }}
            }
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                let minIndex = Int(indicesPtr[seq + elem * sequence])
                if minIndex >= 0 {
                for depth in 0..<nbNeurons
                {
                    let vq = _wArrays.w(minIndex, depth)
                    let outPrev = neuronsPrev.get(seq, depth)!.v[elem].out
                    
                    let g = _wArrays.g(minIndex, depth)
                    _wArrays.g(
                        minIndex, depth,
                        g + coeff / Double(batchSize * sequence) *
                        2.0 * (vq - outPrev)
                    )
                }}
            }}
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
    
    fileprivate func _backwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pK: [UInt32] = [UInt32(K)]
            let pBeta: [Float] = [Float(beta)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "vqSeqBackward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBuffer(_wBuffers.w.metal, atIndex: 2)
            command.setBuffer(indices.metal, atIndex: 3)
            command.setBytes(pNbNeurons, atIndex: 4)
            command.setBytes(pK, atIndex: 5)
            command.setBytes(pBeta, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pSequence, atIndex: 8)
            command.setBytes(pDirty, atIndex: 9)
            command.setBuffer(layerPrev.delta.metal, atIndex: 10)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * sequence
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
    
    fileprivate func _backwardWeightsGPU()
    {
        if let layerPrev = self.layerPrev as? LayerSeq, computeDeltaWeights
        {
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pK: [UInt32] = [UInt32(K)]
            let pCoeff: [Float] = [Float(coeff)]
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
                    "vqSeqBatchDerWeights", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(_wBuffers.w.metal, atIndex: 1)
                command.setBuffer(indices.metal, atIndex: 2)
                command.setBytes(pNbNeurons, atIndex: 3)
                command.setBytes(pK, atIndex: 4)
                command.setBytes(pCoeff, atIndex: 5)
                command.setBytes(pNbBatch, atIndex: 6)
                command.setBytes(pSequence, atIndex: 7)
                command.setBuffer(_wBuffers.g.metal, atIndex: 8)
                
                command.dispatchThreads(width: nbNeurons, height: K)
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
                    "vqSeqDerWeights", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(_wBuffers.w.metal, atIndex: 1)
                command.setBuffer(indices.metal, atIndex: 2)
                command.setBytes(pNbNeurons, atIndex: 3)
                command.setBytes(pK, atIndex: 4)
                command.setBytes(pCoeff, atIndex: 5)
                command.setBytes(pNbBatch, atIndex: 6)
                command.setBytes(pSequence, atIndex: 7)
                command.setBuffer(_wDeltaWeights.metal, atIndex: 8)
                
                command.dispatchThreads(
                    width: nbNeurons,
                    height: batchSize * K
                )
                command.enqueue()
                
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "vq2DReduceWeights", deviceID: deviceID
                ) // vqSeq and vq2D do the same reduction.
                command.setBuffer(_wDeltaWeights.metal, atIndex: 0)
                command.setBytes(pNbNeurons, atIndex: 1)
                command.setBytes(pK, atIndex: 2)
                command.setBytes(pNbBatch, atIndex: 3)
                command.setBytes(pAccumulate, atIndex: 4)
                command.setBuffer(_wBuffers.g.metal, atIndex: 5)
                
                command.dispatchThreads(width: nbNeurons, height: K)
                command.enqueue()
            }
        }
    }
    
    ///
    /// Get loss in the CPU execution context.
    ///
    /// - Returns: The loss value.
    ///
    public func getLossCPU<T: BinaryFloatingPoint>() -> T
    {
        var losses = [T](repeating: 0.0, count: batchSize)
        
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            let neuronsPrev = layerPrev.neurons!
            let indicesPtr = (indices as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                let minIndex = Int(indicesPtr[seq + elem * sequence])
                if minIndex >= 0
                {
                    var value: Double = 0.0
                    for depth in 0..<nbNeurons
                    {
                        let outPrev = neuronsPrev.get(seq, depth)!.v[elem].out
                        let vq = neurons.get(seq, depth)!.v[elem].out
                        value += pow(outPrev - vq, 2.0)
                    }
                    losses[elem] += T(value)
                }
            }}
        }
        return T(coeff) / T(batchSize * sequence) *
            losses.reduce(0, +)
    }
    
    ///
    /// Get loss in the GPU execution context.
    ///
    /// - Returns: The loss value.
    ///
    public func getLossGPU<T: BinaryFloatingPoint>() throws -> T
    {
        try checkLossGPU(batchSize: batchSize)
        
        let layerPrev = self.layerPrev as! LayerSeq
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let command = MetalKernel.get.createCommand(
            "vqSeqLoss", deviceID: deviceID
        )
        command.setBuffer(layerPrev.outs.metal, atIndex: 0)
        command.setBuffer(outs.metal, atIndex: 1)
        command.setBuffer(indices.metal, atIndex: 2)
        command.setBytes(pNbNeurons, atIndex: 3)
        command.setBytes(pNbBatch, atIndex: 4)
        command.setBytes(pSequence, atIndex: 5)
        command.setBuffer(loss.metal, atIndex: 6)
        
        command.dispatchThreads(batchSize)
        command.enqueue()
        
        MetalKernel.get.download([loss])
        var loss: Float = 0.0
        let lossPtr = self.loss.buffer
        for i in 0..<batchSize
        {
            loss += lossPtr[i]
        }
        
        return T(coeff) * T(loss) / T(batchSize * sequence)
    }
    
    /// Compute the derivative of the loss in the CPU execution context.
    public func lossDerivativeCPU() throws
    {
        if dirty
        {
            for elem in 0..<batchSize {
            for seq in 0..<sequence {
            for depth in 0..<nbNeurons 
            {
                neurons.get(seq, depth)!.v[elem].delta = 0.0
            }}}
        }
        else
        {
            throw VQError.RedundantLoss
        }
        
        backwardCPU()
        dirty = false
    }
    
    /// Compute the derivative of the loss in the GPU execution context.
    public func lossDerivativeGPU() throws
    {
        if dirty
        {
            try checkStateBackwardGPU(batchSize: batchSize)
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command = MetalKernel.get.createCommand(
                "reset", deviceID: deviceID
            )
            command.setBytes(pNbElems, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            
            command.dispatchThreads(nbElems)
            command.enqueue()
        }
        else
        {
            throw VQError.RedundantLoss
        }
        
        try backwardGPU()
        dirty = false
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

/// Layer with a sequential shape neural structure and weights.
public class VQGradSeq: VQSeq
{
    /// Scale coefficient for taking into account pixels with high magnitude of gradient norm.
    public var magnitudeCoeff: Double = 2.0
    
    /// Number of threads per thread group in the GPU execution context.
    private let _threadsPerThreadgroup = 64
    
    /// Layer computing a map of maximal activations with respect to the loss.
    let _layerCAM: LayerCAMSeq
    
    ///
    /// Maximal CAM elements.
    /// Shape ~ (batch, nbThreadgroups).
    ///
    private var _camMax: MetalPrivateBuffer<Float>! = nil
    
    /// Number of thread groups in the GPU execution context.
    var nbThreadgroups: Int
    {
        get {
            let value = Double(sequence) /
                        Double(_threadsPerThreadgroup)
            return Int(ceil(value))
        }
    }
    
    /// Whether to take positive or negative part of gradients.
    public var keepPositive: Bool
    {
        get {
            return _layerCAM.keepPositive
        }
        set {
            _layerCAM.keepPositive = newValue
        }
    }
    
    /// GPU device on which model is executed.
    public override var deviceID: Int
    {
        get {
            return super.deviceID
        }
        set {
            super.batchSize = newValue
            _layerCAM.batchSize = newValue
        }
    }
    
    /// Batch size of data.
    public override var batchSize: Int
    {
        get {
            return super.batchSize
        }
        set {
            super.batchSize = newValue
            _layerCAM.batchSize = newValue
        }
    }
    
    /// Running phase of a model: Training or Inference.
    public override var phase: Phase?
    {
        get {
            return super.phase
        }
        set {
            super.phase = newValue
            _layerCAM.phase = newValue
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case magnitudeCoeff
        case layerCAM
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - K: The number of vector approximations.
    ///     - params: Contextual parameters linking to the model.
    ///
    public override init(layerPrev: LayerSeq,
                         K: Int,
                         params: GrAI.Model.Params)
    {
        var paramsHidden = GrAI.Model.Params(params: params)
        paramsHidden.hidden = true
        
        _layerCAM = try! LayerCAMSeq(layerPrev: layerPrev, params: paramsHidden)
        
        super.init(layerPrev: layerPrev, K: K, params: params)
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
        let magnitudeCoeff = try container.decode(
            Float.self, forKey: .magnitudeCoeff
        )
        self.magnitudeCoeff = Double(magnitudeCoeff)
        _layerCAM = try container.decode(LayerCAMSeq.self, forKey: .layerCAM)
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
        try container.encode(Float(magnitudeCoeff), forKey: .magnitudeCoeff)
        try container.encode(_layerCAM, forKey: .layerCAM)
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
            
        let layer = VQGradSeq(
            layerPrev: layerPrev, K: K, params: params
        )
        layer.magnitudeCoeff = magnitudeCoeff
        layer.coeff = coeff
        layer.beta = beta
        
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
    /// Find the `layerPrev` associated to the layer's `idPrev`.
    ///
    /// - Parameter layers: The potential layers where to find the layer's `idPrev`.
    ///
    public override func initLinks(_ layers: [Layer])
    {
        super.initLinks(layers)
        _layerCAM.initLinks(layers)
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
        _layerCAM.resetKernelGPU()
        _camMax = nil
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
        
        if _camMax == nil
        {
            _camMax = MetalPrivateBuffer<Float>(
                batchSize * nbThreadgroups,
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
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            if layerPrev.dirty
            {
                throw UpdateError.Dirty
            }
            
            try _layerCAM.forwardCPU()
            let neuronsCAM = _layerCAM.neurons!
            
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons!
            let indicesPtr = (indices as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize
            {
                var camMax: Double = 0.0
                for seq in 0..<sequence
                {
                    let cam: Double = neuronsCAM.get(seq, 0)!.v[elem].out
                    camMax = max(cam, camMax)
                }
                
                for seq in 0..<sequence
                {
                    let cam: Double = neuronsCAM.get(seq, 0)!.v[elem].out
                    if cam / camMax >= camMax / magnitudeCoeff
                    {
                        var minIndex = -1
                        var minValue: Double? = nil
                        
                        for k in 0..<K
                        {
                            var value: Double = 0.0
                            for depth in 0..<nbNeurons
                            {
                                let outPrev =
                                    neuronsPrev.get(seq, depth)!.v[elem].out
                                let vq = _wArrays.w(k, depth)
                                value += pow(outPrev - vq, 2.0)
                            }
                            
                            if minValue == nil || value < minValue!
                            {
                                minValue = value
                                minIndex = k
                            }
                        }
                        
                        if minIndex < 0
                        {
                            throw VQError.IndexValue
                        }
                        
                        for depth in 0..<nbNeurons
                        {
                            neurons.get(seq, depth)!.v[elem].out =
                                _wArrays.w(minIndex, depth)
                        }
                        indicesPtr[seq + elem * sequence] = Int32(minIndex)
                    }
                    else
                    {
                        indicesPtr[seq + elem * sequence] = -1
                    }
                }
            }
        }
    }
    
    ///
    /// Compute the squared norm in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    private func _computeLayerCAMMaxGPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            if layerPrev.dirty
            {
                throw UpdateError.Dirty
            }
            
            // -----------------------------------------------------------------
            // Begin the reduction that is specific to the gradient norm max.
            // -----------------------------------------------------------------
            
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pNbThreadgroups: [UInt32] = [UInt32(nbThreadgroups)]
            
            let command = MetalKernel.get.createCommand(
                "vqLayerCAMMaxSeq", deviceID: deviceID
            )
            command.setBuffer(_layerCAM.outs.metal, atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pNbThreadgroups, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pSequence, atIndex: 4)
            command.setBuffer(_camMax.metal, atIndex: 5)
            
            let threadsPerThreadgroup = MTLSizeMake(
                _threadsPerThreadgroup, 1, 1
            )
            let threadsPerGrid = MTLSize(
                width: sequence,
                height: batchSize,
                depth: 1
            )
            command.dispatchThreads(
                threadsPerGrid: threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            command.enqueue()
            
            // Continue the reduction in a more generic way.
            reduceMax(
                inBuffer: _camMax.metal,
                outBuffer: _camMax.metal,
                dim1: nbThreadgroups, dim2: batchSize,
                deviceID: deviceID
            )
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
            if layerPrev.dirty
            {
                throw UpdateError.Dirty
            }
            
            try _layerCAM.forwardGPU()
            try _computeLayerCAMMaxGPU()
            
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pK: [UInt32] = [UInt32(K)]
            let pMagnitudeCoeff: [Float] = [Float(magnitudeCoeff)]
            
            let command = MetalKernel.get.createCommand(
                "vqGradSeqForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(_layerCAM.outs.metal, atIndex: 1)
            command.setBuffer(_camMax.metal, atIndex: 2)
            command.setBuffer(_wBuffers.w.metal, atIndex: 3)
            command.setBytes(pNbNeurons, atIndex: 4)
            command.setBytes(pK, atIndex: 5)
            command.setBytes(pMagnitudeCoeff, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pSequence, atIndex: 8)
            command.setBuffer(outs.metal, atIndex: 9)
            command.setBuffer(indices.metal, atIndex: 10)
            
            command.dispatchThreads(
                width: sequence,
                height: batchSize
            )
            command.enqueue()
        }
    }
    
    override func _backwardCPU() {}
    
    override func _backwardGPU() throws {}
}
