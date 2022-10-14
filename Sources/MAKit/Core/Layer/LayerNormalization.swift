//
// LayerNormalization.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

import MetalKit

public protocol Cloneable
{
    func clone() -> Self
}

public class BatchNormalizationBase: Codable, Cloneable
{
    let _nbNeurones: Int
    var _nbElems = 0
    
    var _weightsList = [Float]()
    var _statsList = [Float]()
    
    var weights: [Float]
    {
        get {
            return _weightsList
        }
        set {
            _weightsList = newValue
        }
    }
    var stats: [Float]
    {
        get {
            return _statsList
        }
        set {
            _statsList = newValue
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case nbNeurones
        case nbElems
        case weights
        case stats
    }
    
    init(nbNeurones: Int)
    {
        _nbNeurones = nbNeurones
    }
    
    convenience init(_ layer: BN2D)
    {
        self.init(nbNeurones: layer.nbFilters)
    }
    
    public required init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        
        _nbNeurones = try container.decode(Int.self, forKey: .nbNeurones)
        _nbElems = try container.decode(Int.self, forKey: .nbElems)
        
        _weightsList = try container.decode([Float].self, forKey: .weights)
        _statsList = try container.decode([Float].self, forKey: .stats)
    }
    
    init(bn: BatchNormalizationBase)
    {
        _nbNeurones = bn._nbNeurones
        _nbElems = bn._nbElems
        
        let weights = bn.weights
        self.weights = weights
        
        let stats = bn.stats
        self.stats = stats
    }
    
    init(bn: BatchNormalizationBase, depth: Int)
    {
        _nbNeurones = 1
        _nbElems = bn._nbElems
        
        let weights = bn.weights
        self.weights = [weights[depth], weights[bn._nbNeurones + depth]]
        
        let stats = bn.stats
        self.stats = [stats[depth], stats[bn._nbNeurones + depth]]
    }
    
    public func clone() -> Self
    {
        return BatchNormalizationBase(bn: self) as! Self
    }
    
    public func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        
        try container.encode(_nbNeurones, forKey: .nbNeurones)
        try container.encode(_nbElems, forKey: .nbElems)
        
        let weights = self.weights
        let stats = self.stats
        try container.encode(weights, forKey: .weights)
        try container.encode(stats, forKey: .stats)
    }
}

public class BatchNormalization: BatchNormalizationBase
{
    let _Ɛ: Double = 1e-5
    
    var _Ɣ: WeightArrays! = nil
    var _β: WeightArrays! = nil
    
    var _σ2 = [Double]()
    
    var _Eμ = [Double]()
    var _Eσ2 = [Double]()
    
    var _xHat = [[Double]]()
    
    /// Weights in the CPU execution context.
    override var weights: [Float]
    {
        get {
            if _Ɣ == nil
            {
                return super.weights
            }
            
            var weightsTmp = [Float]()
            for Ɣ in _Ɣ.w
            {
                weightsTmp.append(Float(Ɣ))
            }
            for β in _β.w
            {
                weightsTmp.append(Float(β))
            }
            return weightsTmp
        }
        set {
            if newValue.count > 0 && newValue.count != 2 * _nbNeurones
            {
                fatalError(
                    "Weights do not have the expected number of elements."
                )
            }
            super.weights = newValue
        }
    }
    
    override var stats: [Float]
    {
        get {
            if _Eμ.count == 0
            {
                return super.stats
            }
            
            var statsTmp = [Float]()
            for depth in 0..<_nbNeurones
            {
                statsTmp.append(Float(_Eμ[depth]))
            }
            for depth in 0..<_nbNeurones
            {
                statsTmp.append(Float(_Eσ2[depth]))
            }
            return statsTmp
        }
        set {
            if newValue.count > 0 &&
               newValue.count != 2 * _nbNeurones
            {
                fatalError(
                    "Stats do not have the expected number of elements."
                )
            }
            super.stats = newValue
        }
    }
    
    public override func clone() -> Self
    {
        return BatchNormalization(bn: self) as! Self
    }
    
    ///
    /// Clean state resources in the CPU execution context.
    ///
    /// We do not clean Ɣ and β but must reset their momentum state.
    /// Note that we do not have to reset their delta because here they are independent on
    /// batch size.
    ///
    func resetKernel()
    {
        _Ɣ?.reset()
        _β?.reset()
    }
    
    ///
    /// Initialize hard resources in the CPU execution context.
    ///
    /// We initialize the stats.
    ///
    func initKernel()
    {
        initStats()
        
        _σ2 = [Double](repeating: 0.0, count: _nbNeurones)
        _xHat = [[Double]](repeating: [], count: _nbNeurones)
    }
    
    ///
    /// Initialize weights in the CPU execution context.
    ///
    /// Their momentum state is also reset.
    /// Note that we also initialize the delta which are independent on the batch size.
    ///
    func initWeights()
    {
        _β = WeightArrays(_nbNeurones)
        _Ɣ = WeightArrays(_nbNeurones)
        if _weightsList.count == 0
        {
            for depth in 0..<_nbNeurones
            {
                _Ɣ.w[depth] = 1.0
                _β.w[depth] = 0.0
            }
        }
        else
        {
            for depth in 0..<_nbNeurones
            {
                _Ɣ.w[depth] = Double(_weightsList[depth])
                _β.w[depth] = Double(_weightsList[_nbNeurones + depth])
            }
            _weightsList = []
        }
    }
    
    func initStats()
    {
        _Eμ = [Double](repeating: 0.0, count: _nbNeurones)
        _Eσ2 = [Double](repeating: 0.0, count: _nbNeurones)
        
        if _statsList.count > 0
        {
            for depth in 0..<_nbNeurones
            {
                _Eμ[depth] = Double(_statsList[depth])
                _Eσ2[depth] = Double(_statsList[_nbNeurones + depth])
            }
            _statsList = []
        }
    }
    
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    func forwardGC(_ layer: BN2D)
    {
        let nbGC = layer.nbGC
        let nbFilters = layer.nbFilters
        let Ɛ = layer.Ɛ
        
        Concurrency.slice(nbFilters)
        {
            (depth: Int) in
            
            for elem in 0..<nbGC
            {
                let outs: [Double]
                if elem >= nbGC-4*nbFilters && elem < nbGC-2*nbFilters &&
                   depth == (elem-nbGC+4*nbFilters)/2
                {
                    if elem % 2 == 0
                    {
                        outs = Normalization.forwardGC(
                            outs: layer.getOutsGC(depth: depth, elem: elem),
                            β: _β.w[depth],
                            Ɣ: _Ɣ.w[depth]+Ɛ
                        )
                    }
                    else
                    {
                        outs = Normalization.forwardGC(
                            outs: layer.getOutsGC(depth: depth, elem: elem),
                            β: _β.w[depth],
                            Ɣ: _Ɣ.w[depth]-Ɛ
                        )
                    }
                }
                else if elem >= nbGC-2*nbFilters &&
                        depth == (elem-nbGC+2*nbFilters)/2
                {
                    if elem % 2 == 0
                    {
                        outs = Normalization.forwardGC(
                            outs: layer.getOutsGC(depth: depth, elem: elem),
                            β: _β.w[depth]+Ɛ,
                            Ɣ: _Ɣ.w[depth]
                        )
                    }
                    else
                    {
                        outs = Normalization.forwardGC(
                            outs: layer.getOutsGC(depth: depth, elem: elem),
                            β: _β.w[depth]-Ɛ,
                            Ɣ: _Ɣ.w[depth]
                        )
                    }
                }
                else
                {
                    outs = Normalization.forwardGC(
                        outs: layer.getOutsGC(depth: depth, elem: elem),
                        β: _β.w[depth],
                        Ɣ: _Ɣ.w[depth]
                    )
                }
                layer.setOutsGC(depth: depth, elem: elem, outs: outs)
            }
        }
    }
    
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    func forwardFlowGC(_ layer: BN2D)
    {
        layer._bnGPU?.applyWeights(bn: self)
        forwardGC(layer)
    }
    
    /// Apply the forward pass in the CPU execution context.
    func forward(_ layer: BN2D)
    {
        let nbFilters = layer.nbFilters
        
        if layer.phase != nil && layer.phase! == .Training
        {
            _Eμ.withUnsafeMutableBufferPointer { EμPointer in
            _Eσ2.withUnsafeMutableBufferPointer { Eσ2Pointer in
            _xHat.withUnsafeMutableBufferPointer { xhatPointer in
            _σ2.withUnsafeMutableBufferPointer { σ2Pointer in
            Concurrency.slice(nbFilters)
            {
                (depth: Int) in
                
                let (outs, xHat, μ, σ2) = Normalization.forward(
                    outs: layer.getOuts(depth),
                    β: _β.w[depth],
                    Ɣ: _Ɣ.w[depth]
                )
                layer.setOuts(depth: depth, outs: outs)
                
                xhatPointer[depth] = xHat
                σ2Pointer[depth] = σ2
                
                if _nbElems == 0
                {
                    EμPointer[depth] = μ
                    Eσ2Pointer[depth] = σ2
                }
                else
                {
                    EμPointer[depth] = 0.9 * EμPointer[depth] + 0.1 * μ
                    Eσ2Pointer[depth] = 0.9 * Eσ2Pointer[depth] + 0.1 * σ2
                }
            }}}}}
            
            if _nbElems == 0
            {
                _nbElems = layer.batchSize * layer.height * layer.width
            }
        }
        else
        {
            Concurrency.slice(nbFilters)
            {
                (depth: Int) in
                
                let outs = Normalization.forward(
                    outs: layer.getOuts(depth),
                    μ: _Eμ[depth],
                    σ2: Double(_nbElems) / Double(_nbElems - 1) *
                        _Eσ2[depth],
                    β: _β.w[depth],
                    Ɣ: _Ɣ.w[depth]
                )
                layer.setOuts(depth: depth, outs: outs)
            }
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    func backward(_ layer: BN2D)
    {
        let nbFilters = layer.nbFilters
        
        if layer.phase != nil && layer.phase! == .Training
        {
            var deltaβ = [Double](repeating: 0.0, count: nbFilters)
            var deltaƔ = [Double](repeating: 0.0, count: nbFilters)
            
            for depth in 0..<nbFilters
            {
                let (delta, dβ, dƔ) = Normalization.backward(
                    delta: layer.getDelta(depth),
                    xHat: _xHat[depth],
                    σ2: _σ2[depth],
                    Ɣ: _Ɣ.w[depth]
                )
                layer.setDelta(depth: depth, delta: delta)
                
                deltaβ[depth] += dβ
                deltaƔ[depth] += dƔ
            }
            
            for depth in 0..<nbFilters
            {
                if !layer.accumulateDeltaWeights
                {
                    _Ɣ.g[depth] = deltaƔ[depth]
                    _β.g[depth] = deltaβ[depth]
                }
                else
                {
                    _Ɣ.g[depth] += deltaƔ[depth]
                    _β.g[depth] += deltaβ[depth]
                }
            }
        }
        else
        {
            for depth in 0..<nbFilters
            {
                let delta = Normalization.backward(
                    delta: layer.getDelta(depth),
                    σ2: Double(_nbElems) / Double(_nbElems - 1) *
                        _Eσ2[depth],
                    Ɣ: _Ɣ.w[depth]
                )
                layer.setDelta(depth: depth, delta: delta)
            }
        }
    }
    
    /// Get the weights in the CPU execution context.
    func collectWeights() -> [IWeightArrays]
    {
        return [_Ɣ, _β]
    }
}

class BatchNormalizationGPU: BatchNormalizationBase
{
    var _ƔBuffers: IWeightBuffers! = nil
    var _βBuffers: IWeightBuffers! = nil
    
    var _μ: MetalBuffer<Float>! = nil
    var _Eμ: MetalPrivateBuffer<Float>! = nil
    var _σ2: MetalBuffer<Float>! = nil
    var _Eσ2: MetalPrivateBuffer<Float>! = nil
    
    var _xHat: MetalBuffer<Float>! = nil
    
    var _sum1: MetalBuffer<Float>! = nil
    var _sum2: MetalBuffer<Float>! = nil
   
    /// GPU device on which model is executed.
    var _deviceID = 0
    
    /// Weights in the GPU execution context.
    override var weights: [Float]
    {
        get {
            if _ƔBuffers == nil
            {
                return super.weights
            }
            
            MetalKernel.get.download([_βBuffers.w_p!, _ƔBuffers.w_p!])
            
            var weightsTmp = [Float]()
            weightsTmp += _ƔBuffers.w_p!.shared.array
            weightsTmp += _βBuffers.w_p!.shared.array
            return weightsTmp
        }
        set {
            if newValue.count > 0 && newValue.count != 2 * _nbNeurones
            {
                fatalError(
                    "Weights do not have the expected number of elements."
                )
            }
            super.weights = newValue
        }
    }
    
    override var stats: [Float]
    {
        get {
            if _Eμ == nil
            {
                return super.stats
            }
            
            MetalKernel.get.download([_Eμ, _Eσ2])
            
            var statsTmp = [Float]()
            statsTmp += _Eμ.shared.array
            statsTmp += _Eσ2.shared.array
            return statsTmp
        }
        set {
            if newValue.count > 0 && newValue.count != 2 * _nbNeurones
            {
                fatalError(
                    "Stats do not have the expected number of elements."
                )
            }
            super.stats = newValue
        }
    }
    
    public override func clone() -> Self
    {
        return BatchNormalizationGPU(bn: self) as! Self
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We do not clean Ɣ and β but must reset their momentum state.
    ///
    func resetKernel()
    {
        _μ = nil
        _σ2 = nil
        _xHat = nil
        _sum1 = nil
        _sum2 = nil
        
        _ƔBuffers?.reset()
        _βBuffers?.reset()
    }
    
    ///
    /// Initialize hard resources in the GPU execution context.
    ///
    /// We initialize the stats.
    ///
    /// - Parameter deviceID: The id of GPU where to run the model.
    ///
    func initKernel(deviceID: Int)
    {
        _deviceID = deviceID
        initStats()
    }
    
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// Their momentum and delta state are also reset.
    ///
    func initWeights()
    {
        _βBuffers = WeightBuffers(nbElems: _nbNeurones, deviceID: _deviceID)
        _ƔBuffers = WeightBuffers(nbElems: _nbNeurones, deviceID: _deviceID)
        
        let βPtr = _βBuffers.w_p!.shared.buffer
        let ƔPtr = _ƔBuffers.w_p!.shared.buffer
        
        if _weightsList.count == 0
        {
            for depth in 0..<_nbNeurones
            {
                ƔPtr[depth] = 1.0
                βPtr[depth] = 0.0
            }
        }
        else
        {
            for depth in 0..<_nbNeurones
            {
                ƔPtr[depth] = _weightsList[depth]
                βPtr[depth] = _weightsList[_nbNeurones + depth]
            }
            _weightsList = []
        }
        
        MetalKernel.get.upload([_βBuffers.w_p!, _ƔBuffers.w_p!])
    }
    
    func initStats()
    {
        _Eμ = MetalPrivateBuffer<Float>(_nbNeurones,
                                        deviceID: _deviceID)
        _Eσ2 = MetalPrivateBuffer<Float>(_nbNeurones,
                                         deviceID: _deviceID)
        
        let EμPtr = _Eμ.shared.buffer
        let Eσ2Ptr = _Eσ2.shared.buffer
        
        if _statsList.count == 0
        {
            for depth in 0..<_nbNeurones
            {
                EμPtr[depth] = 0.0
                Eσ2Ptr[depth] = 0.0
            }
        }
        else
        {
            for depth in 0..<_nbNeurones
            {
                EμPtr[depth] = _statsList[depth]
                Eσ2Ptr[depth] = _statsList[_nbNeurones + depth]
            }
            _statsList = []
        }
        
        MetalKernel.get.upload([_Eμ, _Eσ2])
    }
    
    func applyWeights(bn: BatchNormalization)
    {
        let weights = self.weights
        for depth in 0..<_nbNeurones
        {
            bn._Ɣ.w[depth] = Double(weights[depth])
            bn._β.w[depth] = Double(weights[_nbNeurones + depth])
        }
    }
    
    /// Apply the forward pass in the GPU execution context.
    func forward(_ layer: BN2D)
    {
        if layer.phase != nil && layer.phase! == .Training
        {
            forwardTraining(layer)
        }
        else
        {
            fowardInference(layer)
        }
    }
    
    private func computeμ(_ layer: BN2D)
    {
        let nbNeurones = layer.nbFilters
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbNeurones: [UInt32] = [UInt32(nbNeurones)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pFirstCall: [UInt32] = _nbElems == 0 ? [1] : [0]
        
        if _μ == nil
        {
            _μ = MetalPrivateBuffer<Float>(_nbNeurones,
                                           deviceID: _deviceID)
        }
        
        let command = MetalKernel.get.createCommand("computeConvμ",
                                                    deviceID: _deviceID)
        
        command.setBuffer(layer.outs.metal, atIndex: 0)
        command.setBytes(pNbNeurones, atIndex: 1)
        command.setBytes(pNbBatch, atIndex: 2)
        command.setBytes(pDimensions, atIndex: 3)
        command.setBytes(pFirstCall, atIndex: 4)
        command.setBuffer(_μ.metal, atIndex: 5)
        command.setBuffer(_Eμ.metal, atIndex: 6)
        
        let threads = command.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: _nbNeurones, height: 1, depth: 1)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
    
    private func computeσ2(_ layer: BN2D)
    {
        let nbNeurones = layer.nbFilters
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbNeurones: [UInt32] = [UInt32(nbNeurones)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pFirstCall: [UInt32] = _nbElems == 0 ? [1] : [0]
        
        if _σ2 == nil
        {
            _σ2 = MetalPrivateBuffer<Float>(_nbNeurones,
                                            deviceID: _deviceID)
        }
        
        let command = MetalKernel.get.createCommand("computeConvσ2",
                                                    deviceID: _deviceID)
        
        command.setBuffer(layer.outs.metal, atIndex: 0)
        command.setBuffer(_μ.metal, atIndex: 1)
        command.setBytes(pNbNeurones, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pDimensions, atIndex: 4)
        command.setBytes(pFirstCall, atIndex: 5)
        command.setBuffer(_σ2.metal, atIndex: 6)
        command.setBuffer(_Eσ2.metal, atIndex: 7)
        
        let threads = command.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: _nbNeurones, height: 1, depth: 1)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
    
    private func forwardTraining(_ layer: BN2D)
    {
        computeμ(layer)
        computeσ2(layer)
        
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbNeurones: [UInt32] = [UInt32(_nbNeurones)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        if _xHat == nil
        {
            _xHat = MetalPrivateBuffer<Float>(
                batchSize * _nbNeurones * width * height, deviceID: _deviceID)
        }
        
        let command = MetalKernel.get.createCommand("forwardBNConvTraining",
                                                    deviceID: _deviceID)
        
        command.setBuffer(_βBuffers.w.metal, atIndex: 0)
        command.setBuffer(_ƔBuffers.w.metal, atIndex: 1)
        command.setBuffer(_μ.metal, atIndex: 2)
        command.setBuffer(_σ2.metal, atIndex: 3)
        command.setBytes(pNbNeurones, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
        command.setBytes(pDimensions, atIndex: 6)
        command.setBuffer(layer.outs.metal, atIndex: 7)
        command.setBuffer(_xHat.metal, atIndex: 8)
        
        let threadsPerThreadgroup = MTLSizeMake(8, 8, 8)
        let threadsPerGrid = MTLSize(width: width,
                                     height: height,
                                     depth: _nbNeurones * batchSize)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
        
        if _nbElems == 0
        {
            _nbElems = batchSize * width * height
        }
    }
    
    private func fowardInference(_ layer: BN2D)
    {
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbNeurones: [UInt32] = [UInt32(_nbNeurones)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pM: [UInt32] = [UInt32(_nbElems)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        let command = MetalKernel.get.createCommand("forwardBNConvInference",
                                                    deviceID: _deviceID)
        
        command.setBuffer(_βBuffers.w.metal, atIndex: 0)
        command.setBuffer(_ƔBuffers.w.metal, atIndex: 1)
        command.setBuffer(_Eμ.metal, atIndex: 2)
        command.setBuffer(_Eσ2.metal, atIndex: 3)
        command.setBytes(pNbNeurones, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
        command.setBytes(pM, atIndex: 6)
        command.setBytes(pDimensions, atIndex: 7)
        command.setBuffer(layer.outs.metal, atIndex: 8)
        
        let threadsPerThreadgroup = MTLSizeMake(8, 8, 8)
        let threadsPerGrid = MTLSize(width: width,
                                     height: height,
                                     depth: _nbNeurones * batchSize)
        command.dispatchThreads(threadsPerGrid:threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
    
    /// Apply the backward pass in the GPU execution context.
    func backward(_ layer: BN2D)
    {
        if layer.phase != nil && layer.phase! == .Training
        {
            backwardTraining(layer)
        }
        else
        {
            backwardInference(layer)
        }
    }
    
    private func computeTmp(_ layer: BN2D)
    {
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbNeurones: [UInt32] = [UInt32(_nbNeurones)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pAccumulate: [UInt32] = layer.accumulateDeltaWeights ? [1] : [0]
        
        if _sum1 == nil
        {
            _sum1 = MetalPrivateBuffer<Float>(_nbNeurones,
                                              deviceID: _deviceID)
            _sum2 = MetalPrivateBuffer<Float>(_nbNeurones,
                                              deviceID: _deviceID)
        }
        
        let command = MetalKernel.get.createCommand("computeConvTmp",
                                                    deviceID: _deviceID)
        
        command.setBuffer(layer.delta.metal, atIndex: 0)
        command.setBuffer(_xHat.metal, atIndex: 1)
        command.setBuffer(_ƔBuffers.w.metal, atIndex: 2)
        command.setBytes(pNbNeurones, atIndex: 3)
        command.setBytes(pNbBatch, atIndex: 4)
        command.setBytes(pDimensions, atIndex: 5)
        command.setBytes(pAccumulate, atIndex: 6)
        command.setBuffer(_sum1.metal, atIndex: 7)
        command.setBuffer(_sum2.metal, atIndex: 8)
        command.setBuffer(_ƔBuffers.g.metal, atIndex: 9)
        command.setBuffer(_βBuffers.g.metal, atIndex: 10)
        
        let threads = command.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: _nbNeurones, height: 1, depth: 1)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
    
    private func backwardTraining(_ layer: BN2D)
    {
        computeTmp(layer)
        
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbNeurones: [UInt32] = [UInt32(_nbNeurones)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        let command = MetalKernel.get.createCommand("backwardBNConvTraining",
                                                    deviceID: _deviceID)
        
        command.setBuffer(_σ2.metal, atIndex: 0)
        command.setBuffer(_xHat.metal, atIndex: 1)
        command.setBuffer(_ƔBuffers.w.metal, atIndex: 2)
        command.setBuffer(_sum1.metal, atIndex: 3)
        command.setBuffer(_sum2.metal, atIndex: 4)
        command.setBytes(pNbNeurones, atIndex: 5)
        command.setBytes(pNbBatch, atIndex: 6)
        command.setBytes(pDimensions, atIndex: 7)
        command.setBuffer(layer.delta.metal, atIndex: 8)
        
        let threadsPerThreadgroup = MTLSizeMake(8, 8, 8)
        let threadsPerGrid = MTLSize(width: width,
                                     height: height,
                                     depth: _nbNeurones * batchSize)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
    
    private func backwardInference(_ layer: BN2D)
    {
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbNeurones: [UInt32] = [UInt32(_nbNeurones)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pM: [UInt32] = [UInt32(_nbElems)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        let command = MetalKernel.get.createCommand("backwardBNConvInference",
                                                    deviceID: _deviceID)
        
        command.setBuffer(_ƔBuffers.w.metal, atIndex: 0)
        command.setBuffer(_Eσ2.metal, atIndex: 1)
        command.setBytes(pNbNeurones, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pM, atIndex: 4)
        command.setBytes(pDimensions, atIndex: 5)
        command.setBuffer(layer.delta.metal, atIndex: 6)
        
        let threadsPerThreadgroup = MTLSizeMake(8, 8, 8)
        let threadsPerGrid = MTLSize(width: width,
                                     height: height,
                                     depth: _nbNeurones * batchSize)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
    
    /// Get the weights in the GPU execution context.
    func collectWeights() -> [IWeightBuffers]
    {
        return [_ƔBuffers, _βBuffers]
    }
}
