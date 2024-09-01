//
// EmbeddingSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 04/06/2024.
//

import Foundation

/// Input layer with a sequential shape neural structure and weights.
public class EmbeddingSeq: LayerSeq, LayerWeightInit
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
    
    /// Number of new weights due to this layer, estimated during the Gradient Checking.
    var nbLearnedGC: Int
    {
        get {
            return nbNeurons * vocabularySize
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
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        let newGC = 2 * nbLearnedGC
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.initGC(batchSize: batchSize, nbGC: newGC)
        }}
        
        let insPtr = (ins as! MetalSharedBuffer<Int32>).buffer
        
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        for EMBEDDING in 0..<vocabularySize {
        for DEPTH in 0..<nbNeurons {
        for elem in 0...1
        {
            let index = Int(insPtr[sequence * batch + seq])
            for depth in 0..<nbNeurons
            {
                var w = _wArrays.w(index, depth)
                if EMBEDDING == index && DEPTH == depth
                {
                    if elem % 2 == 0
                    {
                        w += Ɛ
                    }
                    else
                    {
                        w -= Ɛ
                    }
                }
                
                let offset = 2 * (DEPTH + nbNeurons * EMBEDDING) + elem
                neurons.get(seq, depth)!.gc[batch][offset].out = w
            }
        }}}}}
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
        for seq in 0..<sequence {
        for depth in 0..<nbNeurons
        {
            neurons.get(seq, depth)!.initGC(batchSize: batchSize, nbGC: newGC)
        }}
        
        let insPtr = ins.download()
        let weightsPtr = _wBuffers.w.download()
        
        for batch in 0..<batchSize {
        for seq in 0..<sequence {
        for EMBEDDING in 0..<vocabularySize {
        for DEPTH in 0..<nbNeurons {
        for elem in 0...1
        {
            let index = Int(insPtr[sequence * batch + seq])
            for depth in 0..<nbNeurons
            {
                let offsetWeights = depth + nbNeurons * index
                var w = Double(weightsPtr[offsetWeights])
                
                if EMBEDDING == index && DEPTH == depth
                {
                    if elem % 2 == 0
                    {
                        w += Ɛ
                    }
                    else
                    {
                        w -= Ɛ
                    }
                }
                
                let offset = 2 * (DEPTH + nbNeurons * EMBEDDING) + elem
                neurons.get(seq, depth)!.gc[batch][offset].out = w
            }
        }}}}}
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        let insPtr = (ins as! MetalSharedBuffer<Int32>).buffer
        
        for elem in 0..<batchSize {
        for seq in 0..<sequence
        {
            let index = Int(insPtr[sequence * elem + seq])
            for depth in 0..<nbNeurons
            {
                neurons.get(seq, depth)!.v[elem].out =
                    _wArrays.w(index, depth)
            }
        }}
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        try checkStateForwardGPU(batchSize: batchSize)
        
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let command = MetalKernel.get.createCommand(
            "embeddingSeqForward", deviceID: deviceID
        )
        command.setBuffer(ins.metal, atIndex: 0)
        command.setBuffer(_wBuffers.w.metal, atIndex: 1)
        command.setBytes(pNbNeurons, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pSequence, atIndex: 4)
        command.setBuffer(outs.metal, atIndex: 5)
        
        command.dispatchThreads(
            width: sequence,
            height: batchSize
        )
        command.enqueue()
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        _backwardWeightsCPU()
    }
    
    fileprivate func _backwardWeightsCPU()
    {
        let insPtr = (ins as! MetalSharedBuffer<Int32>).buffer
        
        if !accumulateDeltaWeights
        {
            for index in 0..<vocabularySize {
            for depth in 0..<nbNeurons
            {
                _wArrays.g(index, depth, 0.0)
            }}
        }
        
        for elem in 0..<batchSize {
        for seq in 0..<sequence
        {
            let index = Int(insPtr[sequence * elem + seq])
            if index < 0 || index >= vocabularySize
            {
                fatalError("Index \(index) is out of range.")
            }
            for depth in 0..<nbNeurons
            {
                let g = _wArrays.g(index, depth)
                let deltaCur = neurons.get(seq, depth)!.v[elem].delta
                
                _wArrays.g(
                    index, depth, g + deltaCur
                )
            }
        }}
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        _backwardWeightsGPU()
    }
    
    fileprivate func _backwardWeightsGPU()
    {
        let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        let pVocabularySize: [UInt32] = [UInt32(vocabularySize)]
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
                "embeddingSeqBatchDerWeights", deviceID: deviceID
            )
            command.setBuffer(ins.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pVocabularySize, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBuffer(_wBuffers.g.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons, height: vocabularySize
            )
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
                "embeddingSeqDerWeights", deviceID: deviceID
            )
            command.setBuffer(ins.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pVocabularySize, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBuffer(_wDeltaWeights.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * vocabularySize
            )
            command.enqueue()
            
            // -------------------------------------------------------------
            // Compute Gradients per batch
            // -------------------------------------------------------------
            command = MetalKernel.get.createCommand(
                "vq2DReduceWeights", deviceID: deviceID
            ) // embeddingSeq and vq2D do the same reduction.
            command.setBuffer(_wDeltaWeights.metal, atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pVocabularySize, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pAccumulate, atIndex: 4)
            command.setBuffer(_wBuffers.g.metal, atIndex: 5)
            
            command.dispatchThreads(
                width: nbNeurons, height: vocabularySize
            )
            command.enqueue()
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
