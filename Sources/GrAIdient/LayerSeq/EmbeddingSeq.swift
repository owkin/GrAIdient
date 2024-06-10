//
// EmbeddingSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 04/06/2024.
//

import Foundation

/// Input layer with a sequential shape neural structure and weights.
class EmbeddingSeq: LayerSeq, LayerWeightInit
{
    /// Size of vocabulary.
    public var vocabularySize: Int
    
    ///
    /// Input buffer.
    /// Shape ~ (batch, seq).
    ///
    public var ins: MetalBuffer<Int32>! = nil
    
    ///
    /// Grid of weights.
    /// Shape ~ (vocabularySize, nbNeurons).
    ///
    var _wArrays: WeightGrids! = nil
    
    ///
    /// Buffer of weights.
    /// Shape ~ (vocabularySize, nbNeurons).
    ///
    var _wBuffers: IWeightBuffers! = nil
    
    ///
    /// Buffer of gradients per sample.
    /// Shape ~ (batch, vocabularySize, nbNeurons).
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
            for index in 0..<vocabularySize {
            for depth in 0..<nbNeurons
            {
                weightsTmp.append(Float(_wArrays.w(index, depth)))
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
            return _wBuffers.w.download()
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
            return (nbNeurons, vocabularySize)
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case vocabularySize
        case weights
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - sequence: Length of the sequence.
    ///     - vocabularySize: Vocabulary size.
    ///     - nbNeurons: Number of neurons.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(sequence: Int,
                vocabularySize: Int,
                nbNeurons: Int,
                params: GrAI.Model.Params)
    {
        self.vocabularySize = vocabularySize
        super.init(layerPrev: nil,
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
        let values = try decoder.container(keyedBy: Keys.self)
        vocabularySize = try values.decode(Int.self, forKey: .vocabularySize)
        
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
        
        try container.encode(vocabularySize, forKey: .vocabularySize)
        
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
        if idPrev > -1
        {
            fatalError("EmbeddingSeq must be the first layer.")
        }
        
        let context = ModelContext(name: "", curID: 0)
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = EmbeddingSeq(
            sequence: sequence,
            vocabularySize: vocabularySize,
            nbNeurons: nbNeurons,
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
        ins = nil
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
        
        ins = nil
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
        
        _wArrays = WeightGrids(width: nbNeurons, height: vocabularySize)
        
        for index in 0..<vocabularySize {
        for depth in 0..<nbNeurons
        {
            let offset = depth + nbNeurons * index
            _wArrays.w(index, depth, Double(_weightsList[offset]))
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
        _wBuffers = WeightBuffers(
            nbElems: vocabularySize * nbNeurons,
            deviceID: deviceID
        )
        
        if _weightsList.count == 0
        {
            generateWeightsList(out: _wBuffers.w, deviceID: deviceID)
        }
        else
        {
            _wBuffers.w.initialize(array: &_weightsList)
        }
        
        _weightsList = []
        _wDeltaWeights = nil
    }
    
    ///
    /// Check and setup input in the CPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The input data.
    ///     - batchSize: The batch size of data.
    ///     - sequence: Length of the sequence.
    ///
    public func checkInputCPU(
        _ data: [[Int]],
        batchSize: Int,
        sequence: Int) throws
    {
        if data.count != batchSize || data.first!.count != sequence
        {
            throw LayerError.DataSize
        }
        
        if ins == nil
        {
            ins = MetalSharedBuffer<Int32>(
                batchSize * sequence, deviceID: deviceID
            )
        }
        else if batchSize <= 0 || batchSize > ins.nbElems / sequence
        {
            throw LayerError.BatchSize
        }
        
        var dataFlat = data.flatMap { $0.map { Int32($0)} }
        let ins_s = ins as! MetalSharedBuffer<Int32>
        copyArrayToBuffer(
            array: &dataFlat,
            buffer: ins_s.buffer,
            start: 0,
            nbElems: batchSize * sequence
        )
    }
    
    ///
    /// Check and setup input in the GPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The input data.
    ///     - batchSize: The batch size of data.
    ///     - sequence: Length of the sequence.
    ///
    public func checkInputGPU(
        _ data: [[Int]],
        batchSize: Int,
        sequence: Int) throws
    {
        if data.count != batchSize || data.first!.count != sequence
        {
            throw LayerError.DataSize
        }
        
        if ins == nil
        {
            ins = MetalPrivateBuffer<Int32>(
                batchSize * sequence, deviceID: deviceID
            )
        }
        else if batchSize <= 0 || batchSize > ins.nbElems / sequence
        {
            throw LayerError.BatchSize
        }
        
        // Wait for previous loop to end to avoid race condition.
        _ = ins.download()
        
        var dataFlat = data.flatMap { $0.map { Int32($0)} }
        let ins_s = ins as! MetalPrivateBuffer<Int32>
        copyArrayToBuffer(
            array: &dataFlat,
            buffer: ins_s.shared.buffer,
            start: 0,
            nbElems: batchSize * sequence
        )
        ins.upload()
    }
    
    ///
    /// API to set data in the CPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The data to set.
    ///     - batchSize: The batch size of data.
    ///     - sequence: Length of the sequence.
    ///
    public func setDataCPU(
        _ data: [[Int]],
        batchSize: Int,
        sequence: Int) throws
    {
        try checkInputCPU(
            data,
            batchSize: batchSize,
            sequence: sequence
        )
    }
    
    ///
    /// API to set data in the GPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The data to set.
    ///     - batchSize: The batch size of data.
    ///     - sequence: Length of the sequence.
    ///
    public func setDataGPU(
        _ data: [[Int]],
        batchSize: Int,
        sequence: Int) throws
    {
        try checkInputGPU(
            data,
            batchSize: batchSize,
            sequence: sequence
        )
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
                batchSize * vocabularySize * nbNeurons, deviceID: deviceID
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
        
        var loss: Float = 0.0
        let lossPtr = self.loss.download()
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
