//
// Convolution2D.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

import Foundation
import MetalKit

///
/// Layer with a 2D shape neural structure, weights and biases,  an activation function and
/// batch normalization units.
///
/// This is the fundamental learning layer of a 2D model.
///
public class Convolution2D: BN2D
{
    /// Downscale factor of the resolution (height and width).
    let _stride: Int
    
    ///
    /// List of grids of weights.
    /// Shape ~ (nbWeights, kernel height, kernel width).
    ///
    var _wArrays: [WeightGrids] = []
    ///
    /// Array of biases.
    /// Shape ~ (nbChannels,).
    ///
    var _bArrays: WeightArrays! = nil
    
    ///
    /// Buffer of weights.
    /// Shape ~ (nbWeights, kernel height, kernel width).
    ///
    var _wBuffers: IWeightBuffers! = nil
    ///
    /// Buffer of biases.
    /// Shape ~ (nbChannels,).
    ///
    var _bBuffers: IWeightBuffers! = nil
    
    ///
    /// Buffer of gradients per sample for weights.
    /// Shape ~ (batch, nbWeights, kernel height, kernel width).
    ///
    var _wDeltaWeights: MetalPrivateBuffer<Float>! = nil
    ///
    /// Buffer of gradients per sample for biases.
    /// Shape ~ (batch, nbChannels).
    ///
    var _bDeltaWeights: MetalPrivateBuffer<Float>! = nil
    
    /// Number of weight kernels.
    public let nbWeights: Int
    /// Height of the weight's grid shape.
    public let weightHeight: Int
    /// Width of the weight's grid shape.
    public let weightWidth: Int
    
    /// Get forward pass GPU kernel.
    var forwardKernel: String
    {
        get {
            return "convForward"
        }
    }
    
    /// Get backward pass GPU kernel.
    var backwardKernel: String
    {
        get {
            return "convBackward"
        }
    }
    
    /// Get GPU kernel used to compute gradients of weights.
    var batchDerWeightsKernel: String
    {
        get {
            return "convBatchDerWeights"
        }
    }
    /// Get GPU kernel used to compute gradients of biases.
    var batchDerBiasesKernel: String
    {
        get {
            return "convBatchDerBiases"
        }
    }
    
    /// Get GPU kernel used to compute gradients of weights per sample.
    var derWeightsKernel: String
    {
        get {
            return "convDerWeights"
        }
    }
    /// Get GPU kernel used to compute gradients of biases per sample.
    var derBiasesKernel: String
    {
        get {
            return "convDerBiases"
        }
    }
    
    /// Get GPU kernel used to reduce gradients of weights per sample.
    var reduceWeightsKernel: String
    {
        get {
            return "convReduceWeights"
        }
    }
    /// Get GPU kernel used to reduce gradients of biases per sample.
    var reduceBiasesKernel: String
    {
        get {
            return "reduceBiases"
        }
    }
    
    /// Get the number of neurons of the previous layer.
    var nbChannelsPrev: Int
    {
        get {
            let layerPrev = self.layerPrev as! Layer2D
            return layerPrev.nbChannels
        }
    }
    
    /// Whether to update biases or not.
    var _updateBiases: Bool = true
    
    /// Cache for weights before calling `initKernel` API.
    var _weightsList = [Float]()
    
    /// Weights (without weights of batch normalization) in the CPU execution context.
    var weightsListCPU: [Float]
    {
        get {
            if _wArrays.count == 0
            {
                return _weightsList
            }
            
            var weightsTmp = [Float]()
            for elem in 0..<nbWeights
            {
                for i in 0..<weightHeight {
                for j in 0..<weightWidth
                {
                    weightsTmp.append(Float(_wArrays[elem].w(i, j)))
                }}
            }
            
            if _updateBiases {
            for depth in 0..<nbChannels
            {
                weightsTmp.append(Float(_bArrays.w[depth]))
            }}
            return weightsTmp
        }
        set {
            _weightsList = newValue
        }
    }
    
    /// Weights (without weights of batch normalization) in the GPU execution context.
    var weightsListGPU: [Float]
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
            weightsListCPU = newValue
        }
    }
    
    /// Weights in the CPU execution context.
    public override var weightsCPU: [Float]
    {
        get {
            var weightsTmp = weightsListCPU
            weightsTmp += super.weightsCPU
            return weightsTmp
        }
        set {
            if newValue.count == 0
            {
                weightsListCPU = []
                super.weightsCPU = []
                return
            }
            
            let nbConvWeights: Int
            if _updateBiases
            {
                nbConvWeights = nbWeights * weightHeight * weightWidth +
                                nbChannels
            }
            else
            {
                nbConvWeights = nbWeights * weightHeight * weightWidth
            }
            weightsListCPU = [Float](newValue[0..<nbConvWeights])
            
            let bnWeights = newValue[nbConvWeights..<newValue.count]
            super.weightsCPU = [Float](bnWeights)
        }
    }
    
    /// Weights in the GPU execution context.
    public override var weightsGPU: [Float]
    {
        get {
            var weightsTmp = weightsListGPU
            weightsTmp += super.weightsGPU
            return weightsTmp
        }
        set {
            if newValue.count == 0
            {
                weightsListGPU = []
                super.weightsGPU = []
                return
            }
            
            let nbConvWeights: Int
            if _updateBiases
            {
                nbConvWeights = nbWeights * weightHeight * weightWidth +
                                nbChannels
            }
            else
            {
                nbConvWeights = nbWeights * weightHeight * weightWidth
            }
            weightsListGPU = [Float](newValue[0..<nbConvWeights])
            
            let bnWeights = newValue[nbConvWeights..<newValue.count]
            super.weightsGPU = [Float](bnWeights)
        }
    }
    
    /// Get the coefficient to apply during the weights initialization.
    var coeffInitWeights: Double
    {
        get {
            if let activation = _activation
            {
                return activation.coeffInitWeights(
                    nPrev: nbChannelsPrev * weightHeight * weightWidth,
                    nCur: nbChannels)
            }
            return sqrt(2.0 /
                        Double(nbChannelsPrev * weightHeight * weightWidth))
        }
    }
    
    /// Number of new weights due to this layer, estimated during the Gradient Checking.
    override var nbLearnedGC: Int
    {
        get {
            var nbGC = 0
            nbGC += nbWeights * weightHeight * weightWidth
            if _updateBiases
            {
                nbGC += nbChannels
            }
            if _bn != nil || _bnGPU != nil
            {
                nbGC += 2 * nbChannels
            }
            return nbGC
        }
    }
    
    /// Downscale factor of the resolution (height and width).
    public override var strideFactor: Double
    {
        get {
            if let value = strideFactorCache
            {
                return value
            }
            else
            {
                let value = super.strideFactor * Double(_stride)
                strideFactorCache = value
                return value
            }
        }
    }
    
    /// The size of the input image this layer is looking at.
    public override var receptiveField: Int
    {
        get {
            if let value = receptiveFieldCache
            {
                return value
            }
            else
            {
                let value = super.receptiveField +
                    (weightWidth - 1) * Int(super.strideFactor)
                receptiveFieldCache = value
                return value
            }
        }
    }
    
    var _kernelIndices: (Int, Int, Int, Int)
    {
        get {
            let weightHeightHalf = weightHeight / 2
            let weightWidthHalf = weightWidth / 2
            let startI = weightWidth % 2 == 1 ? -weightHeightHalf :
                                                -weightHeightHalf+1
            let endI = weightHeightHalf
            let startJ = weightHeight % 2 == 1 ? -weightWidthHalf :
                                                 -weightWidthHalf+1
            let endJ = weightWidthHalf
            
            return (startI, endI, startJ, endJ)
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case stride
        case nbWeights
        case weightWidth
        case weightHeight
        case weights
        case updateBiases
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - size: Size (height, weight) of the weights kernels.
    ///     - nbChannels: Number of channels.
    ///     - stride: Downscale factor of the resolution (height and width).
    ///     - activation: The activation function.
    ///     - biases: Whether to update biases or not.
    ///     - bn: Whether to use batch normalization or not.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D, size: Int, nbChannels: Int, stride: Int,
                activation: String?, biases: Bool, bn: Bool,
                params: MAKit.Model.Params)
    {
        _stride = stride
        
        let width = layerPrev.width
        let height = layerPrev.height
        let widthRes = width % _stride
        let heightRes = height % _stride
        let widthNew = widthRes == 0 ? width / _stride : width / _stride + 1
        let heightNew = heightRes == 0 ? height / _stride : height / _stride + 1
        
        nbWeights = nbChannels * layerPrev.nbChannels
        weightWidth = size
        weightHeight = size
        _updateBiases = biases
        
        super.init(layerPrev: layerPrev,
                   nbChannels: nbChannels,
                   height: heightNew,
                   width: widthNew,
                   activation: activation,
                   bn: bn,
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
        _stride = try values.decode(Int.self, forKey: .stride)
        _updateBiases = try values.decode(Bool.self, forKey: .updateBiases)
        nbWeights = try values.decode(Int.self, forKey: .nbWeights)
        weightWidth = try values.decode(Int.self, forKey: .weightWidth)
        weightHeight = try values.decode(Int.self, forKey: .weightHeight)
        
        try super.init(from: decoder)
        
        let weightsList = try values.decode([Float].self, forKey: .weights)
        self.weightsListCPU = weightsList
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
        
        try container.encode(_stride, forKey: .stride)
        try container.encode(_updateBiases, forKey: .updateBiases)
        try container.encode(nbWeights, forKey: .nbWeights)
        try container.encode(weightWidth, forKey: .weightWidth)
        try container.encode(weightHeight, forKey: .weightHeight)
        
        var weightsList = [Float]()
        if MAKit.Opti.GPU
        {
            weightsList = self.weightsListGPU
        }
        else
        {
            weightsList = self.weightsListCPU
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

        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
        
        let layer = Convolution2D(
            layerPrev: layerPrev,
            size: weightWidth,
            nbChannels: nbChannels,
            stride: _stride,
            activation: _activation?.name,
            biases: _updateBiases,
            bn: _bn != nil || _bnGPU != nil,
            params: params
        )
        if inPlace
        {
            layer._wArrays = _wArrays
            layer._bArrays = _bArrays
            layer._wBuffers = _wBuffers
            layer._bBuffers = _bBuffers
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
            
            if MAKit.Opti.GPU
            {
                layer.weightsListGPU = weightsListGPU
            }
            else
            {
                layer.weightsListCPU = weightsListCPU
            }
        }
        return layer
    }
    
    ///
    /// Extract main operation of this layer.
    ///
    /// - Parameter inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new instance of `Layer`. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    public override func extract(inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = self.layerPrev as! Layer2D
        
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
        
        let layer = Convolution2D(
            layerPrev: layerPrev,
            size: weightHeight,
            nbChannels: nbChannels,
            stride: _stride,
            activation: nil,
            biases: _updateBiases,
            bn: false,
            params: params
        )
        if inPlace
        {
            layer._wArrays = _wArrays
            layer._bArrays = _bArrays
            layer._wBuffers = _wBuffers
            layer._bBuffers = _bBuffers
            layer._bn = nil
            layer._bnGPU = nil
        }
        else
        {
            if MAKit.Opti.GPU
            {
                layer.weightsListGPU = weightsListGPU
            }
            else
            {
                layer.weightsListCPU = weightsListCPU
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
        
        for grid in _wArrays
        {
            grid.reset()
        }
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
    public override func initWeightsCPU()
    {
        super.initWeightsCPU()
        
        _wArrays = [WeightGrids]()
        for _ in 0..<nbWeights
        {
            _wArrays.append(WeightGrids(width: weightWidth,
                                        height: weightHeight))
        }
        _bArrays = WeightArrays(nbChannels)
        
        if _weightsList.count == 0
        {
            let coeff = coeffInitWeights
            for elem in 0..<nbWeights
            {
                for i in 0..<weightHeight {
                for j in 0..<weightWidth
                {
                    _wArrays[elem].w(i, j, coeff * Double.random(in: -1..<1))
                }}
            }
            
            for depth in 0..<nbChannels
            {
                _bArrays.w[depth] = 0.0
            }
        }
        else
        {
            for elem in 0..<nbWeights
            {
                let offsetStart = elem * weightHeight
                
                for i in 0..<weightHeight {
                for j in 0..<weightWidth
                {
                    let offset = j + (offsetStart + i) * weightWidth
                    _wArrays[elem].w(i, j, Double(_weightsList[offset]))
                }}
            }
            
            if _updateBiases
            {
                let offset = nbWeights * weightHeight * weightWidth
                for depth in 0..<nbChannels
                {
                    _bArrays.w[depth] =
                        Double(_weightsList[offset + depth])
                }
            }
            else
            {
                for depth in 0..<nbChannels
                {
                    _bArrays.w[depth] = 0.0
                }
            }
            
            _weightsList = []
        }
    }
    
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// Their momentum and delta state are also reset.
    ///
    public override func initWeightsGPU()
    {
        super.initWeightsGPU()
        
        _wBuffers = WeightBuffers(
            nbElems: nbWeights * weightHeight * weightWidth,
            deviceID: deviceID
        )
        _bBuffers = WeightBuffers(
            nbElems: nbChannels,
            deviceID: deviceID
        )
        
        let weightsPtr = _wBuffers.w_p!.shared.buffer
        let biasesPtr = _bBuffers.w_p!.shared.buffer
        
        if _weightsList.count == 0
        {
            let coeff = Float(coeffInitWeights)
            for elem in 0..<nbWeights * weightHeight * weightWidth
            {
                weightsPtr[elem] = coeff * Float.random(in: -1..<1)
            }
            
            for depth in 0..<nbChannels
            {
                biasesPtr[depth] = 0.0
            }
        }
        else
        {
            for elem in 0..<nbWeights * weightHeight * weightWidth
            {
                weightsPtr[elem] = _weightsList[elem]
            }
            
            if _updateBiases
            {
                let offset = nbWeights * weightHeight * weightWidth
                for depth in 0..<nbChannels
                {
                    biasesPtr[depth] = _weightsList[offset + depth]
                }
            }
            else
            {
                for depth in 0..<nbChannels
                {
                    biasesPtr[depth] = 0.0
                }
            }
            
            _weightsList = []
        }
        
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
           MAKit.Gradient.sample && _wDeltaWeights == nil
        {
            _wDeltaWeights = MetalPrivateBuffer<Float>(
                batchSize * nbWeights * weightWidth * weightHeight,
                deviceID: deviceID
            )
            
            if _updateBiases
            {
                _bDeltaWeights = MetalPrivateBuffer<Float>(
                    batchSize * nbChannels, deviceID: deviceID
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
        bn?.forwardGC(self)
        _activation?.forwardGC(self)
    }
    
    func _forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.initGC(batchSize: batchSize,
                                                      nbGC: newGC)
                }}
            }
            
            let neuronsPrev = layerPrev.neurons
            let (startI, endI, startJ, endJ) = _kernelIndices
            
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var tmp: Double = _bArrays.w[depth]
                    for depthPrev in 0..<nbChannelsPrev
                    {
                        let weights =
                            _wArrays[nbChannelsPrev * depth + depthPrev]
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            if let outPrev = neuronsPrev[depthPrev].get(
                                _stride*i+k, _stride*j+l)?.gc[batch][elem].out
                            {
                                let w = weights.w(k-startI, l-startJ)
                                tmp += outPrev * w
                            }
                        }}
                    }
                    neurons[depth].get(i, j)!.gc[batch][elem].out = tmp
                }}
            }}}
            
            for batch in 0..<batchSize {
            for DEPTH in 0..<nbChannels {
            for DEPTHPREV in 0..<nbChannelsPrev {
            for I in 0..<weightHeight {
            for J in 0..<weightWidth {
            for elem in 0...1 {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var tmp: Double = _bArrays.w[depth]
                    for depthPrev in 0..<nbChannelsPrev
                    {
                        let weights =
                            _wArrays[nbChannelsPrev * depth + depthPrev]
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            if let outPrev = neuronsPrev[depthPrev].get(
                                _stride*i+k, _stride*j+l)?.v[batch].out
                            {
                                var w = weights.w(k-startI, l-startJ)
                                
                                if depth == DEPTH && depthPrev == DEPTHPREV &&
                                   k-startI == I && l-startJ == J
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
                                tmp += outPrev * w
                            }
                        }}
                    }
                    
                    let offset = nbGC +
                        elem + 2 * J + 2 * weightWidth * I +
                        2 * weightWidth * weightHeight * DEPTHPREV +
                        2 * weightWidth * weightHeight * nbChannelsPrev * DEPTH
                    neurons[depth].get(i, j)!.gc[batch][offset].out = tmp
                }}
            }}}}}}}
            
            if _updateBiases {
            for batch in 0..<batchSize {
            for DEPTH in 0..<nbChannels {
            for elem in 0...1 {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
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
                    
                    for depthPrev in 0..<nbChannelsPrev
                    {
                        let weights =
                            _wArrays[nbChannelsPrev * depth + depthPrev]
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            if let outPrev = neuronsPrev[depthPrev].get(
                                _stride*i+k, _stride*j+l)?.v[batch].out
                            {
                                let w = weights.w(k-startI, l-startJ)
                                tmp += outPrev * w
                            }
                        }}
                    }
                    
                    let offset = nbGC +
                        2 * nbWeights * weightHeight * weightWidth + elem +
                        2 * DEPTH
                    neurons[depth].get(i, j)!.gc[batch][offset].out = tmp
                }}
            }}}}}
            
            // Prepare GC for BN weights: Ɣ and β.
            if _bn != nil {
            for batch in 0..<batchSize {
            for elem in newGC-4*nbChannels..<newGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var tmp: Double = _bArrays.w[depth]
                    for depthPrev in 0..<nbChannelsPrev
                    {
                        let weights =
                            _wArrays[nbChannelsPrev * depth + depthPrev]
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            if let outPrev = neuronsPrev[depthPrev].get(
                                _stride*i+k, _stride*j+l)?.v[batch].out
                            {
                                let w = weights.w(k-startI, l-startJ)
                                tmp += outPrev * w
                            }
                        }}
                    }
                    neurons[depth].get(i, j)!.gc[batch][elem].out = tmp
                }}
            }}}}
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
        bn?.forwardFlowGC(self)
        _activation?.forwardGC(self)
    }
    
    func _forwardGCGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.initGC(batchSize: batchSize,
                                                      nbGC: newGC)
                }}
            }
            
            MetalKernel.get.download([_wBuffers.w_p!, _bBuffers.w_p!])
            MetalKernel.get.download([layerPrev.outs])
            
            let weightsPtr = _wBuffers.w_p!.shared.buffer
            let biasesPtr = _bBuffers.w_p!.shared.buffer
            
            let neuronsPrev = layerPrev.neurons
            let widthPrev = layerPrev.width
            let heightPrev = layerPrev.height
            
            let (startI, endI, startJ, endJ) = _kernelIndices
            
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var tmp: Double = Double(biasesPtr[depth])
                    for depthPrev in 0..<nbChannelsPrev
                    {
                        let offsetStartWeights =
                            (depthPrev + nbChannelsPrev * depth) * weightHeight
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            let offsetWeights = l-startJ +
                                (offsetStartWeights + k-startI) * weightWidth
                            
                            if let outPrev = neuronsPrev[depthPrev].get(
                                _stride*i+k, _stride*j+l)?.gc[batch][elem].out
                            {
                                let w = Double(weightsPtr[offsetWeights])
                                tmp += outPrev * w
                            }
                        }}
                    }
                    neurons[depth].get(i, j)!.gc[batch][elem].out = tmp
                }}
            }}}
            
            let outsPrevPtr = layerPrev.outs.shared.buffer
            
            for batch in 0..<batchSize {
            for DEPTH in 0..<nbChannels {
            for DEPTHPREV in 0..<nbChannelsPrev {
            for I in 0..<weightHeight {
            for J in 0..<weightWidth {
            for elem in 0...1 {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var tmp: Double = Double(biasesPtr[depth])
                    for depthPrev in 0..<nbChannelsPrev
                    {
                        let offsetStartPrev =
                            (depthPrev + nbChannelsPrev * batch) * heightPrev
                        let offsetStartWeights =
                            (depthPrev + nbChannelsPrev * depth) * weightHeight
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            let offsetWeights = l-startJ +
                                (offsetStartWeights + k-startI) * weightWidth
                            
                            let I1 = _stride * i + k
                            let J1 = _stride * j + l
                            if I1 >= 0, I1 < heightPrev, J1 >= 0, J1 < widthPrev
                            {
                                var w = Double(weightsPtr[offsetWeights])
                                
                                if depth == DEPTH && depthPrev == DEPTHPREV &&
                                   k-startI == I && l-startJ == J
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
                                
                                let offsetPrev = J1 +
                                    (offsetStartPrev + I1) * widthPrev
                                let outPrev = Double(outsPrevPtr[offsetPrev])
                                
                                tmp += outPrev * w
                            }
                        }}
                    }
                    
                    let offset = nbGC +
                        elem + 2 * J + 2 * weightWidth * I +
                        2 * weightWidth * weightHeight * DEPTHPREV +
                        2 * weightWidth * weightHeight * nbChannelsPrev * DEPTH
                    neurons[depth].get(i, j)!.gc[batch][offset].out = tmp
                }}
            }}}}}}}
            
            if _updateBiases {
            for batch in 0..<batchSize {
            for DEPTH in 0..<nbChannels {
            for elem in 0...1 {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
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
                    
                    for depthPrev in 0..<nbChannelsPrev
                    {
                        let offsetStartPrev =
                            (depthPrev + nbChannelsPrev * batch) * heightPrev
                        let offsetStartWeights =
                            (depthPrev + nbChannelsPrev * depth) * weightHeight
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            let offsetWeights = l-startJ +
                                (offsetStartWeights + k-startI) * weightWidth
                            
                            let I1 = _stride * i + k
                            let J1 = _stride * j + l
                            if I1 >= 0, I1 < heightPrev, J1 >= 0, J1 < widthPrev
                            {
                                let w = Double(weightsPtr[offsetWeights])
                                
                                let offsetPrev = J1 +
                                    (offsetStartPrev + I1) * widthPrev
                                let outPrev = Double(outsPrevPtr[offsetPrev])
                                
                                tmp += outPrev * w
                            }
                        }}
                    }
                    
                    let offset = nbGC +
                        2 * nbWeights * weightHeight * weightWidth + elem +
                        2 * DEPTH
                    neurons[depth].get(i, j)!.gc[batch][offset].out = tmp
                }}
            }}}}}
            
            // Prepare GC for BN weights: Ɣ and β.
            if _bnGPU != nil {
            for batch in 0..<batchSize {
            for elem in newGC-4*nbChannels..<newGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var tmp: Double = _bArrays.w[depth]
                    for depthPrev in 0..<nbChannelsPrev
                    {
                        let offsetStartPrev =
                            (depthPrev + nbChannelsPrev * batch) * heightPrev
                        let offsetStartWeights =
                            (depthPrev + nbChannelsPrev * depth) * weightHeight
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            let offsetWeights = l-startJ +
                                (offsetStartWeights + k-startI) * weightWidth
                            
                            let I1 = _stride * i + k
                            let J1 = _stride * j + l
                            if I1 >= 0, I1 < heightPrev, J1 >= 0, J1 < widthPrev
                            {
                                let w = Double(weightsPtr[offsetWeights])
                                
                                let offsetPrev = J1 +
                                    (offsetStartPrev + I1) * widthPrev
                                let outPrev = Double(outsPrevPtr[offsetPrev])
                                
                                tmp += outPrev * w
                            }
                        }}
                    }
                    neurons[depth].get(i, j)!.gc[batch][elem].out = tmp
                }}
            }}}}
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
        bn?.forward(self)
        _activation?.forwardCPU(self)
    }
    
    func _forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons
            let (startI, endI, startJ, endJ) = _kernelIndices
            
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var tmp: Double = _bArrays.w[depth]
                    for depthPrev in 0..<nbChannelsPrev
                    {
                        let weights =
                            _wArrays[nbChannelsPrev * depth + depthPrev]
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            if let outPrev = neuronsPrev[depthPrev].get(
                                _stride*i+k, _stride*j+l)?.v[elem].out
                            {
                                let w = weights.w(k-startI, l-startJ)
                                tmp += outPrev * w
                            }
                        }}
                    }
                    neurons[depth].get(i, j)!.v[elem].out = tmp
                }}
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
        _bnGPU?.forward(self)
        _activation?.forwardGPU(self)
    }
    
    private func _forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let (startI, endI, startJ, endJ) = _kernelIndices
            
            let pStart: [Int32] = [Int32(startI), Int32(endI),
                                   Int32(startJ), Int32(endJ)]
            let pStride: [UInt32] = [UInt32(_stride)]
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbChannelsPrev: [UInt32] = [UInt32(nbChannelsPrev)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [UInt32(layerPrev.width),
                                             UInt32(layerPrev.height)]
            let pDimWeights: [UInt32] = [UInt32(weightWidth),
                                         UInt32(weightHeight)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            
            if outs == nil
            {
                outs = MetalPrivateBuffer<Float>(
                    batchSize * nbChannels * width * height, deviceID: deviceID)
            }
            
            let command = MetalKernel.get.createCommand(
                forwardKernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(_wBuffers.w.metal, atIndex: 1)
            command.setBuffer(_bBuffers.w.metal, atIndex: 2)
            command.setBytes(pStart, atIndex: 3)
            command.setBytes(pStride, atIndex: 4)
            command.setBytes(pNbChannels, atIndex: 5)
            command.setBytes(pNbChannelsPrev, atIndex: 6)
            command.setBytes(pDimensions, atIndex: 7)
            command.setBytes(pDimensionsPrev, atIndex: 8)
            command.setBytes(pDimWeights, atIndex: 9)
            command.setBytes(pNbBatch, atIndex: 10)
            command.setBuffer(outs.metal, atIndex: 11)
            
            command.dispatchThreads(
                width: nbChannels * width,
                height: batchSize * height
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        _activation?.backwardCPU(self)
        bn?.backward(self)
        
        _backwardCPU()
        _backwardWeightsCPU()
    }
    
    func _backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            let (startI, endI, startJ, endJ) = _kernelIndices
            
            for elem in 0..<batchSize {
            for depthPrev in 0..<nbChannelsPrev
            {
                for i in 0..<layerPrev.height {
                for j in 0..<layerPrev.width
                {
                    var tmp: Double = 0.0
                    for depth in 0..<nbChannels
                    {
                        let weights =
                            _wArrays[nbChannelsPrev * depth + depthPrev]
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            if (i-k) % _stride == 0 && (j-l) % _stride == 0
                            {
                                if let deltaCur = neurons[depth]
                                    .get((i-k) / _stride, (j-l) / _stride)?
                                    .v[elem].delta
                                {
                                    let w = weights.w(k-startI, l-startJ)
                                    tmp += deltaCur * w
                                }
                            }
                        }}
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev[depthPrev].get(i, j)!.v[elem].delta =
                            tmp
                    }
                    else
                    {
                        neuronsPrev[depthPrev].get(i, j)!.v[elem].delta +=
                            tmp
                    }
                }}
            }}
            propagateDirty()
        }
    }
    
    func _backwardWeightsCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeDeltaWeights
        {
            // -----------------------------------------------------------------
            // Compute Gradients per batch
            // -----------------------------------------------------------------
            let neuronsPrev = layerPrev.neurons
            let (startI, endI, startJ, endJ) = _kernelIndices
            
            for depth in 0..<nbChannels
            {
                for depthPrev in 0..<nbChannelsPrev
                {
                    let weights = _wArrays[nbChannelsPrev * depth + depthPrev]
                    
                    for i in startI...endI {
                    for j in startJ...endJ
                    {
                        var tmp: Double = 0.0
                        for elem in 0..<batchSize {
                        for k in 0..<height {
                        for l in 0..<width
                        {
                            if let outPrev = neuronsPrev[depthPrev]
                                .get(_stride*k+i, _stride*l+j)?.v[elem].out
                            {
                                let deltaCur =
                                    neurons[depth].get(k, l)!.v[elem].delta
                                tmp += deltaCur * outPrev
                            }
                        }}}
                        
                        if accumulateDeltaWeights
                        {
                            tmp += weights.g(i-startI, j-startJ)
                        }
                        weights.g(i-startI, j-startJ, tmp)
                    }}
                }
                if _updateBiases
                {
                    var tmp: Double = 0.0
                    for elem in 0..<batchSize {
                    for i in 0..<height {
                    for j in 0..<width
                    {
                        let deltaCur =
                            neurons[depth].get(i, j)!.v[elem].delta
                        tmp += deltaCur
                    }}}
                    
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
        _bnGPU?.backward(self)
        
        try _backwardGPU()
        _backwardWeightsGPU()
    }
    
    private func _backwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let (startI, endI, startJ, endJ) = _kernelIndices
            
            let pStart: [Int32] = [Int32(startI), Int32(endI),
                                   Int32(startJ), Int32(endJ)]
            let pStride: [UInt32] = [UInt32(_stride)]
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbChannelsPrev: [UInt32] = [UInt32(nbChannelsPrev)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [UInt32(layerPrev.width),
                                             UInt32(layerPrev.height)]
            let pDimWeights: [UInt32] = [UInt32(weightWidth),
                                         UInt32(weightHeight)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                backwardKernel, deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(_wBuffers.w.metal, atIndex: 1)
            command.setBytes(pStart, atIndex: 2)
            command.setBytes(pStride, atIndex: 3)
            command.setBytes(pNbChannels, atIndex: 4)
            command.setBytes(pNbChannelsPrev, atIndex: 5)
            command.setBytes(pDimensions, atIndex: 6)
            command.setBytes(pDimensionsPrev, atIndex: 7)
            command.setBytes(pDimWeights, atIndex: 8)
            command.setBytes(pNbBatch, atIndex: 9)
            command.setBytes(pDirty, atIndex: 10)
            command.setBuffer(layerPrev.delta.metal, atIndex: 11)
            
            command.dispatchThreads(
                width: nbChannelsPrev * layerPrev.width,
                height: batchSize * layerPrev.height
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
    
    private func _backwardWeightsGPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeDeltaWeights
        {
            // -----------------------------------------------------------------
            // Compute Gradients per batch
            // -----------------------------------------------------------------
            let (startI, endI, startJ, endJ) = _kernelIndices
            
            let pStart: [Int32] = [Int32(startI), Int32(endI),
                                   Int32(startJ), Int32(endJ)]
            let pStride: [UInt32] = [UInt32(_stride)]
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbChannelsPrev: [UInt32] = [UInt32(nbChannelsPrev)]
            let pDimensions: [UInt32] = [UInt32(width),
                                         UInt32(height)]
            let pDimensionsPrev: [UInt32] = [UInt32(layerPrev.width),
                                             UInt32(layerPrev.height)]
            let pDimWeights: [UInt32] = [UInt32(weightWidth),
                                         UInt32(weightHeight)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pAccumulate: [UInt32] = accumulateDeltaWeights ? [1] : [0]
            
            var command: MetalCommand
            if MAKit.Gradient.batch
            {
                command = MetalKernel.get.createCommand(
                    batchDerWeightsKernel, deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(delta.metal, atIndex: 1)
                command.setBytes(pStart, atIndex: 2)
                command.setBytes(pStride, atIndex: 3)
                command.setBytes(pNbChannels, atIndex: 4)
                command.setBytes(pNbChannelsPrev, atIndex: 5)
                command.setBytes(pDimensions, atIndex: 6)
                command.setBytes(pDimensionsPrev, atIndex: 7)
                command.setBytes(pDimWeights, atIndex: 8)
                command.setBytes(pNbBatch, atIndex: 9)
                command.setBytes(pAccumulate, atIndex: 10)
                command.setBuffer(_wBuffers.g.metal, atIndex: 11)
                
                command.dispatchThreads(
                    width: nbChannels * weightWidth,
                    height: nbChannelsPrev * weightHeight
                )
                command.enqueue()
                
                if _updateBiases
                {
                    command = MetalKernel.get.createCommand(
                        batchDerBiasesKernel, deviceID: deviceID
                    )
                    command.setBuffer(delta.metal, atIndex: 0)
                    command.setBytes(pNbChannels, atIndex: 1)
                    command.setBytes(pDimensions, atIndex: 2)
                    command.setBytes(pNbBatch, atIndex: 3)
                    command.setBytes(pAccumulate, atIndex: 4)
                    command.setBuffer(_bBuffers.g.metal, atIndex: 5)
                    
                    command.dispatchThreads(nbChannels)
                    command.enqueue()
                }
            }
            else
            {
                // -------------------------------------------------------------
                // Compute Gradients per sample
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    derWeightsKernel, deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(delta.metal, atIndex: 1)
                command.setBytes(pStart, atIndex: 2)
                command.setBytes(pStride, atIndex: 3)
                command.setBytes(pNbChannels, atIndex: 4)
                command.setBytes(pNbChannelsPrev, atIndex: 5)
                command.setBytes(pDimensions, atIndex: 6)
                command.setBytes(pDimensionsPrev, atIndex: 7)
                command.setBytes(pDimWeights, atIndex: 8)
                command.setBytes(pNbBatch, atIndex: 9)
                command.setBuffer(_wDeltaWeights.metal, atIndex: 10)
                
                command.dispatchThreads(
                    width: batchSize * nbChannels * weightWidth,
                    height: nbChannelsPrev * weightHeight
                )
                command.enqueue()
            
                if _updateBiases
                {
                    command = MetalKernel.get.createCommand(
                        derBiasesKernel, deviceID: deviceID
                    )
                    command.setBuffer(delta.metal, atIndex: 0)
                    command.setBytes(pNbChannels, atIndex: 1)
                    command.setBytes(pDimensions, atIndex: 2)
                    command.setBytes(pNbBatch, atIndex: 3)
                    command.setBuffer(_bDeltaWeights.metal, atIndex: 4)
                    
                    command.dispatchThreads(
                        width: nbChannels,
                        height: batchSize
                    )
                    command.enqueue()
                }
                
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    reduceWeightsKernel, deviceID: deviceID
                )
                command.setBuffer(_wDeltaWeights.metal, atIndex: 0)
                command.setBytes(pNbChannels, atIndex: 1)
                command.setBytes(pNbChannelsPrev, atIndex: 2)
                command.setBytes(pDimWeights, atIndex: 3)
                command.setBytes(pNbBatch, atIndex: 4)
                command.setBytes(pAccumulate, atIndex: 5)
                command.setBuffer(_wBuffers.g.metal, atIndex: 6)
                
                command.dispatchThreads(
                    width: nbChannels * weightWidth,
                    height: nbChannelsPrev * weightHeight
                )
                command.enqueue()
            
                if _updateBiases
                {
                    command = MetalKernel.get.createCommand(
                        reduceBiasesKernel, deviceID: deviceID
                    )
                    command.setBuffer(_bDeltaWeights.metal, atIndex: 0)
                    command.setBytes(pNbChannels, atIndex: 1)
                    command.setBytes(pNbBatch, atIndex: 2)
                    command.setBytes(pAccumulate, atIndex: 3)
                    command.setBuffer(_bBuffers.g.metal, atIndex: 4)
                    
                    command.dispatchThreads(nbChannels)
                    command.enqueue()
                }
            }
        }
    }
    
    /// Get the weights in the CPU execution context.
    public override func collectWeightsCPU() -> [IWeightArrays]
    {
        var weights = [IWeightArrays]()
        weights += _wArrays
        if _updateBiases
        {
            weights.append(_bArrays)
        }
        if let bn = self.bn
        {
            weights += bn.collectWeights()
        }
        return weights
    }
    
    /// Get the weights in the GPU execution context.
    public override func collectWeightsGPU() -> [IWeightBuffers]
    {
        var weights = [IWeightBuffers]()
        weights.append(_wBuffers)
        if _updateBiases
        {
            weights.append(_bBuffers)
        }
        if let bnFlow = _bnGPU
        {
            weights += bnFlow.collectWeights()
        }
        return weights
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
        
        let nbChannelsPrev = (self.layerPrev as! Layer2D).nbChannels
        var deltaWeights = [T]()
        for depth in 0..<nbChannels {
        for depthPrev in 0..<nbChannelsPrev
        {
            for i in 0..<weightHeight {
            for j in 0..<weightWidth
            {
                deltaWeights.append(T(
                    _wArrays[nbChannelsPrev * depth + depthPrev].g(i, j)
                ))
            }}
        }}
        for depth in 0..<nbChannels
        {
            deltaWeights.append(T(_bArrays.g[depth]))
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
        MetalKernel.get.download([_wBuffers.g_p!, _bBuffers.g_p!])
        
        var deltaWeightsPtr = _wBuffers.g_p!.shared.buffer
        for i in 0..<_wBuffers.nbElems
        {
            deltaWeights.append(T(deltaWeightsPtr[i]))
        }
        deltaWeightsPtr = _bBuffers.g_p!.shared.buffer
        for i in 0..<_bBuffers.nbElems
        {
            deltaWeights.append(T(deltaWeightsPtr[i]))
        }
        return deltaWeights
    }
}
