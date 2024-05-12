//
// Deconvolution2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 26/12/2022.
//

///
/// Layer with a 2D shape neural structure, weights and biases,  an activation function and
/// batch normalization units.
///
/// This allows to up sample size resolution.
///
/// The implementation here corresponds to the no padding version of the link below:
/// https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
/// In the PyTorch documentation, we have padding = 0 and dilation = 1:
/// https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
///
/// The most standard way is to use an even kernel size with a stride greater than 1.
/// This will multiply the previous layer's size by stride.
///
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
        let widthNew = (width - 1) * stride + size
        let heightNew = (height - 1) * stride + size
        
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
            bn: _norm != nil || _normGPU != nil,
            params: params
        )
        if inPlace
        {
            layer._wArrays = _wArrays
            layer._bArrays = _bArrays
            layer._wBuffers = _wBuffers
            layer._bBuffers = _bBuffers
            layer._norm = _norm
            layer._normGPU = _normGPU
        }
        else
        {
            // only one of them should be cloned
            if let bn = _normGPU
            {
                layer._norm = bn.clone()
            }
            else if let bn = _norm
            {
                layer._norm = bn.clone()
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
    /// Extract main operation of this layer without the activation part.
    ///
    /// This API will create a new layer in the same context as this.
    /// 
    /// - Parameter inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new instance of `Layer`. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    public override func removeActivation(inPlace: Bool) -> Layer
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
            layer._norm = nil
            layer._normGPU = nil
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
    
    ///
    /// Extract main operation of this layer without the activation part.
    ///
    /// - Parameter params: Contextual parameters linking to the model.
    ///
    /// - Returns: A new layer.
    ///
    public override func removeActivation(params: GrAI.Model.Params) -> Layer
    {
        let layerPrev = self.layerPrev as! Layer2D
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
        if GrAI.Opti.GPU
        {
            layer.weightsListGPU = weightsListGPU
        }
        else
        {
            layer.weightsListCPU = weightsListCPU
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
            let (startI, endI, startJ, endJ, _, _) = kernelIndices
            
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
                            if (i-k+startI) % _stride == 0 &&
                               (j-l+startJ) % _stride == 0
                            {
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i-k+startI) / _stride,
                                         (j-l+startJ) / _stride)?
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
                            if (i-k+startI) % _stride == 0 &&
                               (j-l+startJ) % _stride == 0
                            {
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i-k+startI) / _stride,
                                         (j-l+startJ) / _stride)?
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
                            if (i-k+startI) % _stride == 0 &&
                               (j-l+startJ) % _stride == 0
                            {
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i-k+startI) / _stride,
                                         (j-l+startJ) / _stride)?
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
            if _norm != nil {
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
                            if (i-k+startI) % _stride == 0 &&
                               (j-l+startJ) % _stride == 0
                            {
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i-k+startI) / _stride,
                                         (j-l+startJ) / _stride)?
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
            
            let weightsPtr = _wBuffers.w.download()
            let biasesPtr = _bBuffers.w.download()
            
            let neuronsPrev = layerPrev.neurons
            let widthPrev = layerPrev.width
            let heightPrev = layerPrev.height
            
            let (startI, endI, startJ, endJ, _, _) = kernelIndices
            
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
                            if (i-k+startI) % _stride == 0 &&
                               (j-l+startJ) % _stride == 0
                            {
                                let offsetWeights = l-startJ +
                                    (offsetStartWeights + k-startI)*weightWidth
                                
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i-k+startI) / _stride,
                                         (j-l+startJ) / _stride)?
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
            
            let outsPrevPtr = layerPrev.outs.download()
            
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
                            if (i-k+startI) % _stride == 0 &&
                               (j-l+startJ) % _stride == 0
                            {
                                let I1 = (i-k+startI) / _stride
                                let J1 = (j-l+startJ) / _stride
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
                            if (i-k+startI) % _stride == 0 &&
                               (j-l+startJ) % _stride == 0
                            {
                                let I1 = (i-k+startI) / _stride
                                let J1 = (j-l+startJ) / _stride
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
            if _norm != nil {
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
                            if (i-k+startI) % _stride == 0 &&
                               (j-l+startJ) % _stride == 0
                            {
                                let I1 = (i-k+startI) / _stride
                                let J1 = (j-l+startJ) / _stride
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
    
    override func _forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons
            let (startI, endI, startJ, endJ, _, _) = kernelIndices
            
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
                            if (i-k+startI) % _stride == 0 &&
                               (j-l+startJ) % _stride == 0
                            {
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((i-k+startI) / _stride,
                                         (j-l+startJ) / _stride)?
                                    .v[elem].out
                                {
                                    let w = weights.w(k-startI, l-startJ)
                                    tmp += outPrev * w
                                }
                            }
                        }}
                    }
                    neurons[depth].get(i, j)!.v[elem].out = tmp
                }}
            }}
        }
    }
    
    override func _backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            let (startI, endI, startJ, endJ, _, _) = kernelIndices
            
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
                            if let deltaCur = neurons[depth].get(
                                _stride*i+k-startI,
                                _stride*j+l-startJ)?.v[elem].delta
                            {
                                let w = weights.w(k-startI, l-startJ)
                                tmp += deltaCur * w
                            }
                        }}
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev[depthPrev].get(i, j)!.v[elem].delta = tmp
                    }
                    else
                    {
                        neuronsPrev[depthPrev].get(i, j)!.v[elem].delta += tmp
                    }
                }}
            }}
            propagateDirty()
        }
    }
    
    override func _backwardWeightsCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeDeltaWeights
        {
            // -----------------------------------------------------------------
            // Compute Gradients per batch
            // -----------------------------------------------------------------
            let neuronsPrev = layerPrev.neurons
            let (startI, endI, startJ, endJ, _, _) = kernelIndices
            
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
                            if (k-i+startI) % _stride == 0 &&
                               (l-j+startJ) % _stride == 0
                            {
                                if let outPrev = neuronsPrev[depthPrev]
                                    .get((k-i+startI) / _stride,
                                         (l-j+startJ) / _stride)?.v[elem].out
                                {
                                    let deltaCur =
                                        neurons[depth].get(k, l)!.v[elem].delta
                                    tmp += deltaCur * outPrev
                                }
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
}
