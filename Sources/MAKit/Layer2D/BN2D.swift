//
// BN2D.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

import MetalKit

public class BN2D: Activation2D, LayerUpdate, LayerExtract
{
    var _bn: BatchNormalizationBase? = nil
    var _bnGPU: BatchNormalizationGPU? = nil
    
    /// Whether to compute weights' gradients or not.
    public var computeDeltaWeights: Bool = true
    
    /// Whether gradients of weights must be accumulated or not.
    public var accumulateDeltaWeights: Bool = false
    
    /// Weights in the CPU execution context.
    public var weightsCPU: [Float]
    {
        get {
            var weightsTmp = [Float]()
            if let bn = _bn
            {
                weightsTmp += bn.weights
            }
            return weightsTmp
        }
        set {
            if let bn = _bn
            {
                bn.weights = newValue
            }
        }
    }
    
    /// Weights in the GPU execution context.
    public var weightsGPU: [Float]
    {
        get {
            var weightsTmp = [Float]()
            if let bn = _bnGPU
            {
                weightsTmp += bn.weights
            }
            else if let bn = _bn
            {
                weightsTmp += bn.weights
            }
            return weightsTmp
        }
        set {
            if let bn = _bnGPU
            {
                bn.weights = newValue
            }
            else if let bn = _bn
            {
                bn.weights = newValue
            }
        }
    }
    
    public var statsCPU: [Float]
    {
        get {
            var statsTmp = [Float]()
            if let bn = _bn
            {
                statsTmp += bn.stats
            }
            return statsTmp
        }
        set {
            if let bn = _bn
            {
                bn.stats = newValue
            }
        }
    }
    
    public var statsGPU: [Float]
    {
        get {
            var statsTmp = [Float]()
            if let bn = _bnGPU
            {
                statsTmp += bn.stats
            }
            else if let bn = _bn
            {
                statsTmp += bn.stats
            }
            return statsTmp
        }
        set {
            if let bn = _bnGPU
            {
                bn.stats = newValue
            }
            else if let bn = _bn
            {
                bn.stats = newValue
            }
        }
    }
    
    var bn: BatchNormalization?
    {
        get {
            return _bn as? BatchNormalization
        }
    }
    
    /// Number of new weights due to this layer, estimated during the Gradient Checking.
    var nbLearnedGC: Int
    {
        get {
            return 2 * nbFilters
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case BN = "BatchNormalization"
    }
    
    public override init(layerPrev: Layer2D, activation: String?,
                         params: MAKit.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbFilters: layerPrev.nbFilters,
                   height: layerPrev.height,
                   width: layerPrev.width,
                   activation: activation,
                   params: params)
        
        _bn = BatchNormalizationBase(self)
    }
    
    public init(layerPrev: Layer2D,
                nbFilters: Int, height: Int, width: Int,
                activation: String?, bn: Bool,
                params: MAKit.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbFilters: nbFilters,
                   height: height,
                   width: width,
                   activation: activation,
                   params: params)
        if bn
        {
            _bn = BatchNormalizationBase(self)
        }
    }
    
    ///
    /// Create an instance of Layer by decoding from the given decoder.
    ///
    /// This initializer throws an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    public required init(from decoder: Decoder) throws
    {
        let values = try decoder.container(keyedBy: Keys.self)
        _bn = try values.decodeIfPresent(BatchNormalizationBase.self,
                                         forKey: .BN)
        try super.init(from: decoder)
    }
    
    ///
    /// Encode this value into the given encoder.
    ///
    /// If the value fails to encode anything, `encoder` will encode an empty
    /// keyed container in its place.
    ///
    /// This function throws an error if any values are invalid for the given
    /// encoder's format.
    ///
    /// - Parameter encoder: The encoder to write data to.
    ///
    public override func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        if let bn = _bnGPU
        {
            try container.encode(bn, forKey: Keys.BN)
        }
        else if let bn = _bn
        {
            try container.encode(bn, forKey: Keys.BN)
        }
        try super.encode(to: encoder)
    }
    
    ///
    /// Create a new instance of `Layer` with same values as this.
    ///
    /// - Parameters:
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///     - inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new instance of `Layer`. When `inPlace` is false, `initKernel` is
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
            
        let layer = BN2D(
            layerPrev: layerPrev,
            activation: _activation?.name,
            params: params
        )
        if inPlace
        {
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
    public func extract(inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = self.layerPrev as! Layer2D
        
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
        
        let layer = BN2D(
            layerPrev: layerPrev,
            activation: nil,
            params: params
        )
        if inPlace
        {
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
        }
        
        return layer
    }
    
    ///
    /// Clean state resources in the CPU execution context.
    ///
    /// We reset batch normalization.
    ///
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        bn?.resetKernel()
    }
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We reset batch normalization.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _bnGPU?.resetKernel()
    }
    
    ///
    /// Initialize hard resources in the CPU execution context.
    ///
    /// We initialize batch normalization.
    ///
    public override func initKernelCPU()
    {
        super.initKernelCPU()
        
        if let bn = _bnGPU
        {
            _bn = BatchNormalization(bn: bn)
        }
        else if let bn = _bn
        {
            _bn = BatchNormalization(bn: bn)
        }
        bn?.initKernel()
        
        if !MAKit.Loop.gradientChecking
        {
            _bnGPU = nil
        }
    }
    
    ///
    /// Initialize hard resources in the GPU execution context.
    ///
    /// We initialize batch normalization.
    ///
    public override func initKernelGPU()
    {
        super.initKernelGPU()
        
        if let bn = _bnGPU
        {
            _bnGPU = BatchNormalizationGPU(bn: bn)
        }
        else if let bn = _bn
        {
            _bnGPU = BatchNormalizationGPU(bn: bn)
        }
        _bnGPU?.initKernel(deviceID: deviceID)
        
        if !MAKit.Loop.gradientChecking
        {
            _bn = nil
        }
    }
    
    ///
    /// Initialize weights in the CPU execution context.
    ///
    /// We initialize batch normalization's weights.
    ///
    public func initWeightsCPU()
    {
        bn?.initWeights()
    }
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// We initialize batch normalization's weights.
    ///
    public func initWeightsGPU()
    {
        _bnGPU?.initWeights()
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        try _forwardGCCPU()
        bn!.forwardGC(self)
        _activation?.forwardGC(self)
    }
    
    private func _forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            for j in 0..<nbFilters
            {
                neurones[j].initGC(batchSize: batchSize, nbGC: newGC)
            }
            
            let neuronesPrev = layerPrev.neurones
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbFilters {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurones[depth].get(i, j)!.gc[batch][elem].out =
                        neuronesPrev[depth].get(i, j)!.gc[batch][elem].out
                }}}
            }}
            
            // Prepare GC for BN weights: Ɣ and β.
            for batch in 0..<batchSize {
            for elem in newGC-4*nbFilters..<newGC
            {
                for depth in 0..<nbFilters {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurones[depth].get(i, j)!.gc[batch][elem].out =
                        neuronesPrev[depth].get(i, j)!.v[batch].out
                }}}
            }}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try _forwardGCGPU()
        bn!.forwardFlowGC(self)
        _activation?.forwardGC(self)
    }
    
    private func _forwardGCGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            for j in 0..<nbFilters
            {
                neurones[j].initGC(batchSize: batchSize, nbGC: newGC)
            }
            
            let neuronesPrev = layerPrev.neurones
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbFilters {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurones[depth].get(i, j)!.gc[batch][elem].out =
                        neuronesPrev[depth].get(i, j)!.gc[batch][elem].out
                }}}
            }}
            
            MetalKernel.get.download([layerPrev.outs])
            let outsPrevPtr = layerPrev.outs.shared.buffer
            
            // Prepare GC for BN weights: Ɣ and β.
            for batch in 0..<batchSize {
            for elem in newGC-4*nbFilters..<newGC
            {
                for depth in 0..<nbFilters
                {
                    let offsetStart =
                        (depth + nbFilters * batch) * height
                    
                    for i in 0..<height {
                    for j in 0..<width
                    {
                        let offset = j + (offsetStart + i) * width
                        
                        neurones[depth].get(i, j)!.gc[batch][elem].out =
                            Double(outsPrevPtr[offset])
                    }}
                }
            }}
        }
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize
            {
                for depth in 0..<nbFilters {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurones[depth].get(i, j)!.v[elem].out =
                        neuronesPrev[depth].get(i, j)!.v[elem].out
                }}}
            }
            
            bn!.forward(self)
            _activation?.forwardCPU(self)
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let nbElems = outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command = MetalKernel.get.createCommand(
                "sum1", deviceID: deviceID)
            
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            let threads = command.threadExecutionWidth
            let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
            let threadsPerGrid = MTLSize(width: nbElems,
                                         height: 1,
                                         depth: 1)
            command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
            command.enqueue()
            
            _bnGPU!.forward(self)
            _activation?.forwardGPU(self)
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        _activation?.backwardCPU(self)
        bn!.backward(self)
        
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize
            {
                for depth in 0..<nbFilters {
                for i in 0..<height {
                for j in 0..<width
                {
                    if layerPrev.dirty
                    {
                        neuronesPrev[depth].get(i, j)!.v[elem].delta =
                            neurones[depth].get(i, j)!.v[elem].delta
                    }
                    else
                    {
                        neuronesPrev[depth].get(i, j)!.v[elem].delta +=
                            neurones[depth].get(i, j)!.v[elem].delta
                    }
                }}}
            }
            
            propagateDirty()
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        _activation?.backwardGPU(self)
        _bnGPU!.backward(self)
        
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command: MetalCommand
            if layerPrev.dirty
            {
                command = MetalKernel.get.createCommand(
                    "sum1", deviceID: deviceID)
            }
            else
            {
                command = MetalKernel.get.createCommand(
                    "sum2", deviceID: deviceID)
            }
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(layerPrev.delta.metal, atIndex: 2)
            
            let threads = command.threadExecutionWidth
            let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
            let threadsPerGrid = MTLSize(width: nbElems,
                                         height: 1,
                                         depth: 1)
            command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
            command.enqueue()
            
            propagateDirty()
        }
    }
    
    /// Get the weights in the CPU execution context.
    public func collectWeightsCPU() -> [IWeightArrays]
    {
        var weights = [IWeightArrays]()
        if let bn = self.bn
        {
            weights += bn.collectWeights()
        }
        return weights
    }
    
    /// Get the weights in the GPU execution context.
    public func collectWeightsGPU() -> [IWeightBuffers]
    {
        return _bnGPU!.collectWeights()
    }
    
    func getOutsGC(depth: Int, elem: Int) -> [Double]
    {
        var sorties = [Double](repeating: 0.0,
                               count: batchSize * height * width)
        
        for batch in 0..<batchSize
        {
            let offsetStart = batch * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                sorties[offset] =
                    neurones[depth].get(i, j)!.gc[batch][elem].out
            }}
        }
        
        return sorties
    }
    
    func setOutsGC(depth: Int, elem: Int, outs: [Double])
    {
        for batch in 0..<batchSize
        {
            let offsetStart = batch * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                neurones[depth].get(i, j)!.gc[batch][elem].out = outs[offset]
            }}
        }
    }
    
    func getOuts(_ depth: Int) -> [Double]
    {
        var outs = [Double](repeating: 0.0,
                            count: batchSize * height * width)
        for elem in 0..<batchSize
        {
            let offsetStart = elem * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                outs[offset] = neurones[depth].get(i, j)!.v[elem].out
            }}
        }
        return outs
    }
    
    func setOuts(depth: Int, outs: [Double])
    {
        for elem in 0..<batchSize
        {
            let offsetStart = elem * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                neurones[depth].get(i, j)!.v[elem].out = outs[offset]
            }}
        }
    }
    
    func getDelta(_ depth: Int) -> [Double]
    {
        var delta = [Double](repeating: 0.0,
                             count: batchSize * height * width)
        for elem in 0..<batchSize
        {
            let offsetStart = elem * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                delta[offset] = neurones[depth].get(i, j)!.v[elem].delta
            }}
        }
        return delta
    }
    
    func setDelta(depth: Int, delta: [Double])
    {
        for elem in 0..<batchSize
        {
            let offsetStart = elem * height
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offset = j + (offsetStart + i) * width
                neurones[depth].get(i, j)!.v[elem].delta = delta[offset]
            }}
        }
    }
}
