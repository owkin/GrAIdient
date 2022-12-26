//
// Deconvolution2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 26/12/2022.
//

public class Deconvolution2D: Convolution2D
{
    /// Get forward pass GPU kernel.
    override var forwardKernel: String
    {
        get {
            return "deconvForward"
        }
    }
    
    /// Get backward pass GPU kernel.
    override var backwardKernel: String
    {
        get {
            return "deconvBackward"
        }
    }
    
    /// Get GPU kernel used to compute gradients of weights.
    override var batchDerWeightsKernel: String
    {
        get {
            return "deconvBatchDerWeights"
        }
    }
    
    /// Get GPU kernel used to compute gradients of weights per sample.
    override var derWeightsKernel: String
    {
        get {
            return "deconvDerWeights"
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
                let value = super.strideFactor / Double(_stride)
                strideFactorCache = value
                return value
            }
        }
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
    public override init(layerPrev: Layer2D,
                         size: Int, nbChannels: Int, stride: Int,
                         activation: String?, biases: Bool, bn: Bool,
                         params: GrAI.Model.Params)
    {
        let width = layerPrev.width
        let height = layerPrev.height
        let widthNew = width * stride
        let heightNew = height * stride
        
        super.init(layerPrev: layerPrev,
                   size: size,
                   nbChannels: nbChannels,
                   stride: stride,
                   height: heightNew,
                   width: widthNew,
                   activation: activation,
                   biases: biases,
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
        try super.init(from: decoder)
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
        
        let layer = Deconvolution2D(
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
            
            if GrAI.Opti.GPU
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
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = Deconvolution2D(
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
            if GrAI.Opti.GPU
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
    
    override func _forwardGCCPU() throws
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
                    neurons[depth].get(i, j)!.initGC(
                        batchSize: batchSize,
                        nbGC: newGC
                    )
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
                            if (i+k) % _stride == 0 && (j+l) % _stride == 0
                            {
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i+k) / _stride, (j+l) / _stride)?
                                    .gc[batch][elem].out
                                {
                                    let w = weights.w(k-startI, l-startJ)
                                    tmp += outPrev * w
                                }
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
                            if (i+k) % _stride == 0 && (j+l) % _stride == 0
                            {
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i+k) / _stride, (j+l) / _stride)?
                                    .v[batch].out
                                {
                                    var w = weights.w(k-startI, l-startJ)
                                    
                                    if depth == DEPTH &&
                                       depthPrev == DEPTHPREV &&
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
                            if (i+k) % _stride == 0 && (j+l) % _stride == 0
                            {
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i+k) / _stride, (j+l) / _stride)?
                                    .v[batch].out
                                {
                                    let w = weights.w(k-startI, l-startJ)
                                    tmp += outPrev * w
                                }
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
                            if (i+k) % _stride == 0 && (j+l) % _stride == 0
                            {
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i+k) / _stride, (j+l) / _stride)?
                                    .v[batch].out
                                {
                                    let w = weights.w(k-startI, l-startJ)
                                    tmp += outPrev * w
                                }
                            }
                        }}
                    }
                    neurons[depth].get(i, j)!.gc[batch][elem].out = tmp
                }}
            }}}}
        }
    }
    
    override func _forwardGCGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.initGC(
                        batchSize: batchSize,
                        nbGC: newGC
                    )
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
                            if (i+k) % _stride == 0 && (j+l) % _stride == 0
                            {
                                let offsetWeights = l-startJ +
                                    (offsetStartWeights + k-startI)*weightWidth
                                
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i+k) / _stride, (j+l) / _stride)?
                                    .gc[batch][elem].out
                                {
                                    let w = Double(weightsPtr[offsetWeights])
                                    tmp += outPrev * w
                                }
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
                            if (i+k) % _stride == 0 && (j+l) % _stride == 0
                            {
                                let I1 = (i+k) / _stride
                                let J1 = (j+l) / _stride
                                if I1 >= 0, I1 < heightPrev,
                                   J1 >= 0, J1 < widthPrev
                                {
                                    let offsetWeights = l-startJ +
                                    (offsetStartWeights + k-startI)*weightWidth
                                    
                                    var w = Double(weightsPtr[offsetWeights])
                                    
                                    if depth == DEPTH &&
                                       depthPrev == DEPTHPREV &&
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
                                    let outPrev = Double(
                                        outsPrevPtr[offsetPrev]
                                    )
                                    tmp += outPrev * w
                                }
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
                            if (i+k) % _stride == 0 && (j+l) % _stride == 0
                            {
                                let I1 = (i+k) / _stride
                                let J1 = (j+l) / _stride
                                if I1 >= 0, I1 < heightPrev,
                                   J1 >= 0, J1 < widthPrev
                                {
                                    let offsetWeights = l-startJ +
                                    (offsetStartWeights + k-startI)*weightWidth
                                    
                                    let w = Double(weightsPtr[offsetWeights])
                                    
                                    let offsetPrev = J1 +
                                        (offsetStartPrev + I1) * widthPrev
                                    let outPrev = Double(
                                        outsPrevPtr[offsetPrev]
                                    )
                                    tmp += outPrev * w
                                }
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
                        let offsetStartPrev =
                            (depthPrev + nbChannelsPrev * batch) * heightPrev
                        let offsetStartWeights =
                            (depthPrev + nbChannelsPrev * depth) * weightHeight
                        
                        for k in startI...endI {
                        for l in startJ...endJ
                        {
                            if (i+k) % _stride == 0 && (j+l) % _stride == 0
                            {
                                let I1 = (i+k) / _stride
                                let J1 = (j+l) / _stride
                                if I1 >= 0, I1 < heightPrev,
                                   J1 >= 0, J1 < widthPrev
                                {
                                    let offsetWeights = l-startJ +
                                    (offsetStartWeights + k-startI)*weightWidth
                                    
                                    let w = Double(weightsPtr[offsetWeights])
                                    
                                    let offsetPrev = J1 +
                                        (offsetStartPrev + I1) * widthPrev
                                    let outPrev = Double(
                                        outsPrevPtr[offsetPrev]
                                    )
                                    tmp += outPrev * w
                                }
                            }
                        }}
                    }
                    neurons[depth].get(i, j)!.gc[batch][elem].out = tmp
                }}
            }}}}
        }
    }
    
    override func _forward(_ elem: Int)
    {
        if let previousLayer = _previousLayer as? Layer2D
        {
            let neuronesPrev = previousLayer._neurones
            
            let semiTailleX = weightWidth/2
            let semiTailleY = weightHeight/2
            let debutX = weightWidth % 2 == 1 ? -semiTailleX :
                                                -semiTailleX+1
            let finX = semiTailleX
            let debutY = weightHeight % 2 == 1 ? -semiTailleY :
                                                 -semiTailleY+1
            let finY = semiTailleY
            
            for depth in 0..<nbFilters
            {
                let neurones = _neurones[depth]
                
                for x in 0..<height {
                for y in 0..<width
                {
                    var agregation: Double = _biases.get(depth)!._w
                    for curPoids in 0..<nbFiltersPrev
                    {
                        let poids = _weights[nbFiltersPrev * depth + curPoids]
                        var somme: Double = 0.0
                        
                        for i in debutX...finX {
                        for j in debutY...finY
                        {
                            let neuroneTmp: Neurone?
                            if (x+i) % _stride != 0
                            {
                                neuroneTmp = nil
                            }
                            else if (y+j) % _stride != 0
                            {
                                neuroneTmp = nil
                            }
                            else
                            {
                                let X = (x+i) / _stride
                                let Y = (y+j) / _stride
                                neuroneTmp = neuronesPrev[curPoids].get(X, Y)
                            }
                            if let sortiePrev = neuroneTmp?._var[elem].sortie,
                               let wIJ = poids.get(i-debutX, j-debutY)?._w
                            {
                                somme += sortiePrev * wIJ
                            }
                        }}
                        
                        agregation += somme
                    }
                    neurones.get(x, y)!._var[elem].sortie = agregation
                }}
            }
        }
    }
    
    override func _backward(_ elem: Int)
    {
        if let previousLayer = _previousLayer as? Layer2D, mustComputeBackward
        {
            let neuronesPrev = previousLayer._neurones
            
            let semiTailleX = weightWidth/2
            let semiTailleY = weightHeight/2
            let debutX = weightWidth % 2 == 1 ? -semiTailleX :
                                                -semiTailleX+1
            let finX = semiTailleX
            let debutY = weightHeight % 2 == 1 ? -semiTailleY :
                                                 -semiTailleY+1
            let finY = semiTailleY
            
            for prevDepth in 0..<nbFiltersPrev
            {
                for x in 0..<previousLayer.height {
                for y in 0..<previousLayer.width
                {
                    var miniDelta: Double = 0.0
                    for depth in 0..<nbFilters
                    {
                        let neurones = _neurones[depth]
                        let poids = _weights[nbFiltersPrev*depth+prevDepth]
                        var somme: Double = 0.0
                        
                        for i in debutX...finX {
                        for j in debutY...finY
                        {
                            if let neurone = neurones.get(
                                _stride*x-i, _stride*y-j),
                               let wIJ = poids.get(i-debutX, j-debutY)?._w
                            {
                                somme += neurone._var[elem].miniDelta * wIJ
                            }
                        }}
                        
                        miniDelta += somme
                    }
                    
                    if previousLayer.isDirty
                    {
                    neuronesPrev[prevDepth].get(x, y)!._var[elem].miniDelta =
                        miniDelta
                    }
                    else
                    {
                    neuronesPrev[prevDepth].get(x, y)!._var[elem].miniDelta +=
                        miniDelta
                    }
                }}
            }
        }
    }
    
    override func _deriveePartielle(_ elem: Int)
    {
        if let previousLayer = _previousLayer as? Layer2D, mustComputeUpdate
        {
            let neuronesPrev = previousLayer._neurones
            
            let semiTailleX = weightWidth/2
            let semiTailleY = weightHeight/2
            let debutX = weightWidth % 2 == 1 ? -semiTailleX :
                                                -semiTailleX+1
            let finX = semiTailleX
            let debutY = weightHeight % 2 == 1 ? -semiTailleY :
                                                 -semiTailleY+1
            let finY = semiTailleY
            
            for depth in 0..<nbFilters
            {
                let neurones = _neurones[depth]
                
                for curPoids in 0..<nbFiltersPrev
                {
                    let poids = _weights[nbFiltersPrev * depth + curPoids]
                    
                    for i in debutX...finX {
                    for j in debutY...finY
                    {
                        var deriveePartielle: Double = 0.0
                        
                        for x in 0..<height {
                        for y in 0..<width
                        {
                            let neuroneTmp: Neurone?
                            if (x+i) % _stride != 0
                            {
                                neuroneTmp = nil
                            }
                            else if (y+j) % _stride != 0
                            {
                                neuroneTmp = nil
                            }
                            else
                            {
                                let X = (x+i) / _stride
                                let Y = (y+j) / _stride
                                neuroneTmp = neuronesPrev[curPoids].get(X,Y)
                            }
                            if let sortiePrev = neuroneTmp?._var[elem].sortie,
                               let miniDelta = neurones.get(x,
                                                    y)?._var[elem].miniDelta
                            {
                                deriveePartielle += miniDelta * sortiePrev
                            }
                        }}
                        
                        poids.get(i-debutX,
                                  j-debutY)?._var[elem].deriveePartielle =
                            deriveePartielle
                    }}
                }
                
                if updateBiases
                {
                    var deriveeBiais: Double = 0.0
                    
                    for x in 0..<height {
                    for y in 0..<width
                    {
                        let neuroneXY = neurones.get(x, y)
                        if let miniDeltaIJ = neuroneXY?._var[elem].miniDelta
                        {
                            deriveeBiais += miniDeltaIJ
                        }
                    }}
                    
                    _biases.get(depth)?._var[elem].deriveePartielle =
                        deriveeBiais
                }
            }
            
            for depth in 0..<nbFilters
            {
                for curPoids in 0..<nbFiltersPrev
                {
                    let poids = _weights[nbFiltersPrev * depth + curPoids]
                    
                    for i in 0..<height {
                    for j in 0..<width
                    {
                        if let connexionIJ = poids.get(i, j)
                        {
                            var partialGrad: Double = 0.0
                            for elem in 0..<batchSize
                            {
                                let deriveePartielle =
                                    connexionIJ._var[elem].deriveePartielle
                                partialGrad += deriveePartielle
                            }
                            
                            if !accumulateWeights
                            {
                                connexionIJ._derivative = partialGrad
                            }
                            else
                            {
                                connexionIJ._derivative += partialGrad
                            }
                        }
                    }}
                }
                if updateBiases, let biais = _biases.get(depth)
                {
                    var partialGrad: Double = 0.0
                    for elem in 0..<batchSize
                    {
                        let deriveePartielle =
                            biais._var[elem].deriveePartielle
                        partialGrad += deriveePartielle
                    }
                    
                    if !accumulateWeights
                    {
                        biais._derivative = partialGrad
                    }
                    else
                    {
                        biais._derivative += partialGrad
                    }
                }
            }
        }
    }
}
