//
// FullyConnectedPatch.swift
// GrAIdient
//
// Created by Jean-François Reboud on 21/02/2023.
//

import Foundation

///
/// Layer with a sequential shape neural structure, weights and biases and an activation function.
///
/// This layer transforms a 2D layer into a sequential layer, applying fully connected operation
/// on patches of the 2D layer.
///
public class FullyConnectedPatch: ActivationSeq,
                                  LayerWithActivation, LayerWeightInit
{
    /// Size (height, width) of a patch.
    let _patch: Int
    
    ///
    /// Grid of weights.
    /// Shape ~ (nbNeurons, nbNeuronsPrev x patch x patch).
    ///
    var _wArrays: WeightGrids! = nil
    ///
    /// Array of biases.
    /// Shape ~ (nbNeurons,).
    ///
    var _bArrays: WeightArrays! = nil
    
    ///
    /// Buffer of weights.
    /// Shape ~ (nbNeurons, nbNeuronsPrev x patch x patch).
    ///
    var _wBuffers: IWeightBuffers! = nil
    ///
    /// Buffer of biases.
    /// Shape ~ (nbNeurons,).
    ///
    var _bBuffers: IWeightBuffers! = nil
    
    ///
    /// Buffer of gradients per sample for weights.
    /// Shape ~ (batch, nbNeurons, nbNeuronsPrev x patch x patch).
    ///
    var _wDeltaWeights: MetalPrivateBuffer<Float>! = nil
    ///
    /// Buffer of gradients per sample for biases.
    /// Shape ~ (batch, nbNeurons).
    ///
    var _bDeltaWeights: MetalPrivateBuffer<Float>! = nil
    
    /// Whether to compute weights' gradients or not.
    public var computeDeltaWeights: Bool = true
    
    /// Whether gradients of weights must be accumulated or not.
    public var accumulateDeltaWeights: Bool = false
    
    /// Height of the weight's grid shape.
    public let weightHeight: Int
    /// Width of the weight's grid shape.
    public let weightWidth: Int
    
    /// Whether to update biases or not.
    var _updateBiases: Bool = true
    
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
            for i in 0..<weightHeight {
            for j in 0..<weightWidth
            {
                weightsTmp.append(Float(_wArrays.w(i, j)))
            }}
            
            if _updateBiases {
            for depth in 0..<weightHeight
            {
                weightsTmp.append(Float(_bArrays.w[depth]))
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
            
            if _updateBiases
            {
                MetalKernel.get.download([_bBuffers.w_p!])
                weightsTmp += _bBuffers.w_p!.shared.array
            }
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
            return (weightWidth, weightHeight)
        }
    }
    
    /// Number of new weights due to this layer, estimated during the Gradient Checking.
    var nbLearnedGC: Int
    {
        get {
            if !_updateBiases
            {
                return nbNeurons * weightWidth
            }
            else
            {
                return nbNeurons * (weightWidth + 1)
            }
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case patch
        case weightWidth
        case weightHeight
        case weights
        case updateBiases
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - patch: Size of a patch.
    ///     - nbNeurons: Number of neurons.
    ///     - activation: The activation function.
    ///     - biases: Whether to update biases or not.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                patch: Int, nbNeurons: Int, activation: String?, biases: Bool,
                params: GrAI.Model.Params) throws
    {
        if layerPrev.height % patch != 0 || layerPrev.width % patch != 0
        {
            throw LayerError.Init(message:
                "Previous layer's size is not divisible by patch \(patch)."
            )
        }
        
        _patch = patch
        let nbPatches = (layerPrev.height / patch) * (layerPrev.width / patch)
        
        weightWidth = layerPrev.nbChannels * patch * patch
        weightHeight = nbNeurons
        _updateBiases = biases
        
        super.init(layerPrev: layerPrev,
                   sequence: nbPatches,
                   nbNeurons: nbNeurons,
                   activation: activation,
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
        _patch = try values.decode(Int.self, forKey: .patch)
        _updateBiases = try values.decode(Bool.self, forKey: .updateBiases)
        weightWidth = try values.decode(Int.self, forKey: .weightWidth)
        weightHeight = try values.decode(Int.self, forKey: .weightHeight)
        
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
        
        try container.encode(_patch, forKey: .patch)
        try container.encode(_updateBiases, forKey: .updateBiases)
        try container.encode(weightWidth, forKey: .weightWidth)
        try container.encode(weightHeight, forKey: .weightHeight)
        
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
            
        let layer = try! FullyConnectedPatch(
            layerPrev: layerPrev,
            patch: _patch,
            nbNeurons: nbNeurons,
            activation: _activation?.name,
            biases: _updateBiases,
            params: params
        )
        if inPlace
        {
            layer._wArrays = _wArrays
            layer._bArrays = _bArrays
            layer._wBuffers = _wBuffers
            layer._bBuffers = _bBuffers
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
        
        let layer = try! FullyConnectedPatch(
            layerPrev: layerPrev,
            patch: _patch,
            nbNeurons: nbNeurons,
            activation: nil,
            biases: _updateBiases,
            params: params
        )
        if inPlace
        {
            layer._wArrays = _wArrays
            layer._bArrays = _bArrays
            layer._wBuffers = _wBuffers
            layer._bBuffers = _bBuffers
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
    /// Extract main operation of this layer without the activation part.
    ///
    /// - Parameter params: Contextual parameters linking to the model.
    ///
    /// - Returns: A new layer.
    ///
    public func removeActivation(params: GrAI.Model.Params) -> Layer
    {
        let layerPrev = self.layerPrev as! Layer2D
        let layer = try! FullyConnectedPatch(
            layerPrev: layerPrev,
            patch: _patch,
            nbNeurons: nbNeurons,
            activation: nil,
            biases: _updateBiases,
            params: params
        )
        if GrAI.Opti.GPU
        {
            layer.weightsGPU = weightsGPU
        }
        else
        {
            layer.weightsCPU = weightsCPU
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
        _bArrays?.reset()
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
        _bDeltaWeights = nil
        
        _wBuffers?.reset()
        _bBuffers?.reset()
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
            _weightsList += [Float](repeating: 0.0, count: weightHeight)
        }
        
        _wArrays = WeightGrids(width: weightWidth, height: weightHeight)
        _bArrays = WeightArrays(weightHeight)
        
        for i in 0..<weightHeight {
        for j in 0..<weightWidth
        {
            let offset = j + weightWidth * i
            _wArrays.w(i, j, Double(_weightsList[offset]))
        }}
    
        // In both cases, biases may have been set by caller or by ourselves.
        if _updateBiases
        {
            let offset = weightHeight * weightWidth
            for depth in 0..<weightHeight
            {
                _bArrays.w[depth] = Double(_weightsList[offset + depth])
            }
        }
        else
        {
            for depth in 0..<weightHeight
            {
                _bArrays.w[depth] = 0.0
            }
        }
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
            _weightsList += [Float](repeating: 0.0, count: weightHeight)
        }
        
        _wBuffers = WeightBuffers(
            nbElems: weightHeight * weightWidth,
            deviceID: deviceID
        )
        _bBuffers = WeightBuffers(
            nbElems: weightHeight,
            deviceID: deviceID
        )
        
        let weightsPtr = _wBuffers.w_p!.shared.buffer
        let biasesPtr = _bBuffers.w_p!.shared.buffer
    
        for elem in 0..<weightHeight * weightWidth
        {
            weightsPtr[elem] = _weightsList[elem]
        }
        
        // In both cases, biases may have been set by caller or by ourselves.
        if _updateBiases
        {
            let offset = weightHeight * weightWidth
            for depth in 0..<weightHeight
            {
                biasesPtr[depth] = _weightsList[offset + depth]
            }
        }
        else
        {
            for depth in 0..<weightHeight
            {
                biasesPtr[depth] = 0.0
            }
        }
        _weightsList = []
        
        MetalKernel.get.upload([_wBuffers.w_p!, _bBuffers.w_p!])
        
        _wDeltaWeights = nil
        _bDeltaWeights = nil
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
                batchSize * sequence * nbNeurons * weightWidth,
                deviceID: deviceID
            )
            
            if _updateBiases
            {
                _bDeltaWeights = MetalPrivateBuffer<Float>(
                    batchSize * sequence * nbNeurons, deviceID: deviceID
                )
            }
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        try _forwardGCCPU()
        _activation?.forwardGC(self)
    }
    
    private func _forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
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
            
            let nbSeqPerCol = layerPrev.width / _patch
            let neuronsPrev = layerPrev.neurons
            let nbChannelsPrev = layerPrev.nbChannels
            
            for elem in 0..<nbGC {
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                let seqI = seq / nbSeqPerCol
                let seqJ = seq % nbSeqPerCol
                
                let iStart = seqI * _patch
                let jStart = seqJ * _patch
                
                for depth in 0..<nbNeurons
                {
                    var tmp: Double = _bArrays.w[depth]
                    for depthPrev in 0..<nbChannelsPrev {
                    for i in 0..<_patch {
                    for j in 0..<_patch
                    {
                        let offsetWeight = j +
                            i * _patch + depthPrev * _patch * _patch
                        
                        let w = _wArrays.w(depth, offsetWeight)
                        let outPrev = neuronsPrev[depthPrev]
                            .get(iStart+i, jStart+j)!.gc[batch][elem].out
                        tmp += w * outPrev
                    }}}
                    neurons.get(seq, depth)!.gc[batch][elem].out = tmp
                }
            }}}
            
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                let seqI = seq / nbSeqPerCol
                let seqJ = seq % nbSeqPerCol
                
                let iStart = seqI * _patch
                let jStart = seqJ * _patch
                
                for DEPTH in 0..<nbNeurons {
                for DEPTHPREV in 0..<weightWidth {
                for elem in 0...1
                {
                    for depth in 0..<nbNeurons
                    {
                        var tmp: Double = _bArrays.w[depth]
                        for depthPrev in 0..<nbChannelsPrev {
                        for i in 0..<_patch {
                        for j in 0..<_patch
                        {
                            let offsetWeight = j +
                                i * _patch + depthPrev * _patch * _patch
                
                            var w = _wArrays.w(depth, offsetWeight)
                            if depth == DEPTH && offsetWeight == DEPTHPREV
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
                            
                            let outPrev = neuronsPrev[depthPrev]
                                .get(iStart+i, jStart+j)!.v[batch].out
                            tmp += w * outPrev
                        }}}
                        
                        let offset = nbGC +
                            2 * (DEPTHPREV + weightWidth * DEPTH) + elem
                        neurons.get(seq, depth)!.gc[batch][offset].out = tmp
                    }
                }}}
            }}
            
            if _updateBiases {
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                let seqI = seq / nbSeqPerCol
                let seqJ = seq % nbSeqPerCol
                
                let iStart = seqI * _patch
                let jStart = seqJ * _patch
                
                for DEPTH in 0..<nbNeurons {
                for elem in 0...1
                {
                    for depth in 0..<nbNeurons
                    {
                        var tmp: Double = _bArrays.w[depth]
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
                        
                        for depthPrev in 0..<nbChannelsPrev {
                        for i in 0..<_patch {
                        for j in 0..<_patch
                        {
                            let offsetWeight = j +
                                i * _patch + depthPrev * _patch * _patch
                            
                            let w = _wArrays.w(depth, offsetWeight)
                            let outPrev = neuronsPrev[depthPrev]
                                .get(iStart+i, jStart+j)!.v[batch].out
                            tmp += w * outPrev
                        }}}
                        
                        let offset = nbGC +
                            2 * nbNeurons * weightWidth +
                            2 * DEPTH + elem
                        neurons.get(seq, depth)!.gc[batch][offset].out = tmp
                    }
                }}
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
        _activation?.forwardGC(self)
    }
    
    private func _forwardGCGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
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
            
            MetalKernel.get.download([_wBuffers.w_p!, _bBuffers.w_p!])
            MetalKernel.get.download([layerPrev.outs])
            
            let weightsPtr = _wBuffers.w_p!.shared.buffer
            let biasesPtr = _bBuffers.w_p!.shared.buffer
            
            let nbSeqPerCol = layerPrev.width / _patch
            let neuronsPrev = layerPrev.neurons
            let nbChannelsPrev = layerPrev.nbChannels
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            for elem in 0..<nbGC {
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                let seqI = seq / nbSeqPerCol
                let seqJ = seq % nbSeqPerCol
                
                let iStart = seqI * _patch
                let jStart = seqJ * _patch
                
                for depth in 0..<nbNeurons
                {
                    var tmp: Double = Double(biasesPtr[depth])
                    for depthPrev in 0..<nbChannelsPrev {
                    for i in 0..<_patch {
                    for j in 0..<_patch
                    {
                        let offsetWeight = j +
                            i * _patch + depthPrev * _patch * _patch
                        let offsetWeights = offsetWeight + weightWidth * depth
                        
                        let w = Double(weightsPtr[offsetWeights])
                        let outPrev = neuronsPrev[depthPrev]
                            .get(iStart+i, jStart+j)!.gc[batch][elem].out
                        tmp += w * outPrev
                    }}}
                    neurons.get(seq, depth)!.gc[batch][elem].out = tmp
                }
            }}}
            
            let outsPrevPtr = layerPrev.outs.shared.buffer
            
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                let seqI = seq / nbSeqPerCol
                let seqJ = seq % nbSeqPerCol
                
                let iStart = seqI * _patch
                let jStart = seqJ * _patch
                
                for DEPTH in 0..<nbNeurons {
                for DEPTHPREV in 0..<weightWidth {
                for elem in 0...1
                {
                    for depth in 0..<nbNeurons
                    {
                        var tmp: Double = Double(biasesPtr[depth])
                        for depthPrev in 0..<nbChannelsPrev {
                        for i in 0..<_patch {
                        for j in 0..<_patch
                        {
                            let offsetWeight = j +
                                i * _patch + depthPrev * _patch * _patch
                            let offsetWeights = offsetWeight +
                                weightWidth * depth
                            let offsetStartPrev =
                                (depthPrev + nbChannelsPrev*batch) * heightPrev
                            let offsetPrev = jStart+j +
                                (offsetStartPrev + iStart+i) * widthPrev
                
                            var w = Double(weightsPtr[offsetWeights])
                            if depth == DEPTH && offsetWeight == DEPTHPREV
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
                            
                            let outPrev = Double(outsPrevPtr[offsetPrev])
                            tmp += w * outPrev
                        }}}
                        
                        let offset = nbGC +
                            2 * (DEPTHPREV + weightWidth * DEPTH) + elem
                        neurons.get(seq, depth)!.gc[batch][offset].out = tmp
                    }
                }}}
            }}
            
            if _updateBiases {
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                let seqI = seq / nbSeqPerCol
                let seqJ = seq % nbSeqPerCol
                
                let iStart = seqI * _patch
                let jStart = seqJ * _patch
                
                for DEPTH in 0..<nbNeurons {
                for elem in 0...1
                {
                    for depth in 0..<nbNeurons
                    {
                        var tmp: Double = Double(biasesPtr[depth])
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
                        
                        for depthPrev in 0..<nbChannelsPrev {
                        for i in 0..<_patch {
                        for j in 0..<_patch
                        {
                            let offsetWeight = j +
                                i * _patch + depthPrev * _patch * _patch
                            let offsetWeights = offsetWeight +
                                weightWidth * depth
                            let offsetStartPrev =
                                (depthPrev + nbChannelsPrev*batch) * heightPrev
                            let offsetPrev = jStart+j +
                                (offsetStartPrev + iStart+i) * widthPrev
                            
                            let w = Double(weightsPtr[offsetWeights])
                            let outPrev = Double(outsPrevPtr[offsetPrev])
                            tmp += w * outPrev
                        }}}
                        
                        let offset = nbGC +
                            2 * nbNeurons * weightWidth +
                            2 * DEPTH + elem
                        neurons.get(seq, depth)!.gc[batch][offset].out = tmp
                    }
                }}
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
        try _forwardCPU()
        _activation?.forwardCPU(self)
    }
    
    private func _forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbSeqPerCol = layerPrev.width / _patch
            let neuronsPrev = layerPrev.neurons
            let nbChannelsPrev = layerPrev.nbChannels
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                let seqI = seq / nbSeqPerCol
                let seqJ = seq % nbSeqPerCol
                
                let iStart = seqI * _patch
                let jStart = seqJ * _patch
                
                for depth in 0..<nbNeurons
                {
                    var tmp: Double = _bArrays.w[depth]
                    for depthPrev in 0..<nbChannelsPrev {
                    for i in 0..<_patch {
                    for j in 0..<_patch
                    {
                        let offsetWeight = j +
                            i * _patch + depthPrev * _patch * _patch
                        
                        let w = _wArrays.w(depth, offsetWeight)
                        let outPrev = neuronsPrev[depthPrev]
                            .get(iStart+i, jStart+j)!.v[elem].out
                        tmp += w * outPrev
                    }}}
                    
                    neurons.get(seq, depth)!.v[elem].out = tmp
                }
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
        try _forwardGPU()
        _activation?.forwardGPU(self)
    }
    
    private func _forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbChannelsPrev: [UInt32] = [UInt32(layerPrev.nbChannels)]
            let pDimensionsPrev: [UInt32] = [UInt32(layerPrev.width),
                                             UInt32(layerPrev.height)]
            let pPatch: [UInt32] = [UInt32(_patch)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            
            let command = MetalKernel.get.createCommand(
                "flPatchForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(_wBuffers.w.metal, atIndex: 1)
            command.setBuffer(_bBuffers.w.metal, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbChannelsPrev, atIndex: 4)
            command.setBytes(pDimensionsPrev, atIndex: 5)
            command.setBytes(pPatch, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pSequence, atIndex: 8)
            command.setBuffer(outs.metal, atIndex: 9)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize * sequence
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        _activation?.backwardCPU(self)
        
        _backwardCPU()
        _backwardWeightsCPU()
    }
    
    private func _backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let nbSeqPerCol = layerPrev.width / _patch
            let neuronsPrev = layerPrev.neurons
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                let seqI = seq / nbSeqPerCol
                let seqJ = seq % nbSeqPerCol
                
                let iStart = seqI * _patch
                let jStart = seqJ * _patch
                
                for offsetWeight in 0..<weightWidth
                {
                    var res = offsetWeight
                    let depthPrev = res / (_patch * _patch)
                    res -= depthPrev * _patch * _patch
                    let i = res / _patch
                    res -= i * _patch
                    let j = res
                    
                    var tmp: Double = 0.0
                    for depth in 0..<nbNeurons
                    {
                        let w = _wArrays.w(depth, offsetWeight)
                        let deltaCur = neurons.get(seq, depth)!.v[elem].delta
                        tmp += w * deltaCur
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev[depthPrev]
                            .get(iStart+i, jStart+j)!.v[elem].delta = tmp
                    }
                    else
                    {
                        neuronsPrev[depthPrev]
                            .get(iStart+i, jStart+j)!.v[elem].delta += tmp
                    }
                }
            }}
            propagateDirty()
        }
    }
    
    private func _backwardWeightsCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeDeltaWeights
        {
            let nbSeqPerCol = layerPrev.width / _patch
            let neuronsPrev = layerPrev.neurons
            
            // -----------------------------------------------------------------
            // Compute Gradients per batch
            // -----------------------------------------------------------------
            for depth in 0..<nbNeurons {
            for offsetWeight in 0..<weightWidth
            {
                var res = offsetWeight
                let depthPrev = res / (_patch * _patch)
                res -= depthPrev * _patch * _patch
                let i = res / _patch
                res -= i * _patch
                let j = res
                
                var tmp: Double = 0.0
                for elem in 0..<batchSize {
                for seq in 0..<sequence
                {
                    let seqI = seq / nbSeqPerCol
                    let seqJ = seq % nbSeqPerCol
                    
                    let iStart = seqI * _patch
                    let jStart = seqJ * _patch
                    
                    let deltaCur = neurons.get(seq, depth)!.v[elem].delta
                    let outPrev = neuronsPrev[depthPrev]
                        .get(iStart + i, jStart + j)!.v[elem].out
                    tmp += outPrev * deltaCur
                }}
                
                if accumulateDeltaWeights
                {
                    tmp += _wArrays.g(depth, offsetWeight)
                }
                _wArrays.g(depth, offsetWeight, tmp)
            }}
            
            if _updateBiases
            {
                for depth in 0..<nbNeurons
                {
                    var tmp: Double = 0.0
                    for elem in 0..<batchSize {
                    for seq in 0..<sequence
                    {
                        let deltaCur = neurons.get(seq, depth)!.v[elem].delta
                        tmp += deltaCur
                    }}
                    
                    if accumulateDeltaWeights
                    {
                        tmp += _bArrays.g[depth]
                    }
                    _bArrays.g[depth] = tmp
                }
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
        _activation?.backwardGPU(self)
        
        try _backwardGPU()
        _backwardWeightsGPU()
    }
    
    ///
    /// Apply a backward guided pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    /// - Parameter positive:
    ///     if positive, negative gradients are reset
    ///     if not, positive gradients are reset
    ///
    public override func backwardGuidedGPU(positive: Bool) throws
    {
        _activation?.backwardGPU(self)
        
        let nbElems = delta.nbElems
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        
        let command: MetalCommand
        if positive
        {
            command = MetalKernel.get.createCommand(
                "resetNeg", deviceID: deviceID
            )
        }
        else
        {
            command = MetalKernel.get.createCommand(
                "resetPos", deviceID: deviceID
            )
        }
        
        command.setBytes(pNbElems, atIndex: 0)
        command.setBuffer(delta.metal, atIndex: 1)
        
        command.dispatchThreads(nbElems)
        command.enqueue()
        
        try backwardGPU()
        _backwardWeightsGPU()
    }
    
    private func _backwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbChannelsPrev: [UInt32] = [UInt32(layerPrev.nbChannels)]
            let pDimensionsPrev: [UInt32] = [UInt32(layerPrev.width),
                                             UInt32(layerPrev.height)]
            let pPatch: [UInt32] = [UInt32(_patch)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "flPatchBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(_wBuffers.w.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbChannelsPrev, atIndex: 3)
            command.setBytes(pDimensionsPrev, atIndex: 4)
            command.setBytes(pPatch, atIndex: 5)
            command.setBytes(pNbBatch, atIndex: 6)
            command.setBytes(pSequence, atIndex: 7)
            command.setBytes(pDirty, atIndex: 8)
            command.setBuffer(layerPrev.delta.metal, atIndex: 9)
            
            command.dispatchThreads(
                width: weightWidth,
                height: batchSize * sequence
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
    
    private func _backwardWeightsGPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeDeltaWeights
        {
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbChannelsPrev: [UInt32] = [UInt32(layerPrev.nbChannels)]
            let pDimensionsPrev: [UInt32] = [UInt32(layerPrev.width),
                                             UInt32(layerPrev.height)]
            let pPatch: [UInt32] = [UInt32(_patch)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pAccumulate: [UInt32] = accumulateDeltaWeights ? [1] : [0]
            
            var command: MetalCommand
            if GrAI.Gradient.batch
            {
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "flPatchBatchDerWeights", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(delta.metal, atIndex: 1)
                command.setBytes(pNbNeurons, atIndex: 2)
                command.setBytes(pNbChannelsPrev, atIndex: 3)
                command.setBytes(pDimensionsPrev, atIndex: 4)
                command.setBytes(pPatch, atIndex: 5)
                command.setBytes(pNbBatch, atIndex: 6)
                command.setBytes(pSequence, atIndex: 7)
                command.setBytes(pAccumulate, atIndex: 8)
                command.setBuffer(_wBuffers.g.metal, atIndex: 9)
                
                command.dispatchThreads(
                    width: nbNeurons,
                    height: weightWidth
                )
                command.enqueue()
                
                if _updateBiases
                {
                    let kernel = nbNeurons % 4 == 0 ?
                        "flPatchBatch4DerBiases" : "flPatchBatchDerBiases"
                    let coeff = nbNeurons % 4 == 0 ? 4 : 1
                    command = MetalKernel.get.createCommand(
                        kernel, deviceID: deviceID
                    )
                    command.setBuffer(delta.metal, atIndex: 0)
                    command.setBytes(pNbNeurons, atIndex: 1)
                    command.setBytes(pNbBatch, atIndex: 2)
                    command.setBytes(pSequence, atIndex: 3)
                    command.setBytes(pAccumulate, atIndex: 4)
                    command.setBuffer(_bBuffers.g.metal, atIndex: 5)
                    
                    command.dispatchThreads(nbNeurons / coeff)
                    command.enqueue()
                }
            }
            else
            {
                // -------------------------------------------------------------
                // Compute Gradients per sample
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "flPatchDerWeights", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(delta.metal, atIndex: 1)
                command.setBytes(pNbNeurons, atIndex: 2)
                command.setBytes(pNbChannelsPrev, atIndex: 3)
                command.setBytes(pDimensionsPrev, atIndex: 4)
                command.setBytes(pPatch, atIndex: 5)
                command.setBytes(pNbBatch, atIndex: 6)
                command.setBytes(pSequence, atIndex: 7)
                command.setBuffer(_wDeltaWeights.metal, atIndex: 8)
                
                command.dispatchThreads(
                    width: nbNeurons * batchSize,
                    height: weightWidth
                )
                command.enqueue()
                
                if _updateBiases
                {
                    command = MetalKernel.get.createCommand(
                        "flPatchDerBiases", deviceID: deviceID
                    )
                    command.setBuffer(delta.metal, atIndex: 0)
                    command.setBytes(pNbNeurons, atIndex: 1)
                    command.setBytes(pNbBatch, atIndex: 2)
                    command.setBytes(pSequence, atIndex: 3)
                    command.setBuffer(_bDeltaWeights.metal, atIndex: 4)
                    
                    command.dispatchThreads(
                        width: nbNeurons,
                        height: batchSize
                    )
                    command.enqueue()
                }
                
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "flPatchReduceWeights", deviceID: deviceID
                )
                command.setBuffer(_wDeltaWeights.metal, atIndex: 0)
                command.setBytes(pNbNeurons, atIndex: 1)
                command.setBytes(pNbChannelsPrev, atIndex: 2)
                command.setBytes(pPatch, atIndex: 3)
                command.setBytes(pNbBatch, atIndex: 4)
                command.setBytes(pAccumulate, atIndex: 5)
                command.setBuffer(_wBuffers.g.metal, atIndex: 6)
                
                command.dispatchThreads(
                    width: nbNeurons,
                    height: weightWidth
                )
                command.enqueue()
                
                if _updateBiases
                {
                    command = MetalKernel.get.createCommand(
                        "reduceBiases", deviceID: deviceID
                    )
                    command.setBuffer(_bDeltaWeights.metal, atIndex: 0)
                    command.setBytes(pNbNeurons, atIndex: 1)
                    command.setBytes(pNbBatch, atIndex: 2)
                    command.setBytes(pAccumulate, atIndex: 3)
                    command.setBuffer(_bBuffers.g.metal, atIndex: 4)
                    
                    command.dispatchThreads(nbNeurons)
                    command.enqueue()
                }
            }
        }
    }
    
    /// Get the weights in the CPU execution context.
    public func collectWeightsCPU() -> [IWeightArrays]
    {
        var weights = [IWeightArrays]()
        weights.append(_wArrays)
        if _updateBiases
        {
            weights.append(_bArrays)
        }
        return weights
    }
    
    /// Get the weights in the GPU execution context.
    public func collectWeightsGPU() -> [IWeightBuffers]
    {
        var weights = [IWeightBuffers]()
        weights.append(_wBuffers)
        if _updateBiases
        {
            weights.append(_bBuffers)
        }
        return weights
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
        MetalKernel.get.download([_wDeltaWeights])
        var deltaWeightsPtr = _wDeltaWeights.shared.buffer
        
        let offsetStart = elem * nbNeurons * weightWidth
        for depth in 0..<nbNeurons {
        for depthPrev in 0..<weightWidth
        {
            let offset = offsetStart + depthPrev + weightWidth * depth
            
            deltaWeights.append(T(
                deltaWeightsPtr[offset]
            ))
        }}
        
        if _updateBiases
        {
            MetalKernel.get.download([_bDeltaWeights])
            deltaWeightsPtr = _bDeltaWeights.shared.buffer
            
            for depth in 0..<nbNeurons
            {
                let offset = depth + nbNeurons * elem
                
                deltaWeights.append(T(
                    deltaWeightsPtr[offset]
                ))
            }
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
        for depth in 0..<nbNeurons {
        for depthPrev in 0..<weightWidth
        {
            deltaWeights.append(T(_wArrays.g(depth, depthPrev)))
        }}
        if _updateBiases
        {
            for depth in 0..<nbNeurons
            {
                deltaWeights.append(T(_bArrays.g[depth]))
            }
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
        MetalKernel.get.download([_wBuffers.g_p!])
        var deltaWeightsPtr = _wBuffers.g_p!.shared.buffer
        
        for i in 0..<_wBuffers.nbElems
        {
            deltaWeights.append(T(deltaWeightsPtr[i]))
        }
        if _updateBiases
        {
            MetalKernel.get.download([_bBuffers.g_p!])
            deltaWeightsPtr = _bBuffers.g_p!.shared.buffer
            
            for i in 0..<_bBuffers.nbElems
            {
                deltaWeights.append(T(deltaWeightsPtr[i]))
            }
        }
        return deltaWeights
    }
}
