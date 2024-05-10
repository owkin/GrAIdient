//
// LayerNormalization.swift
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

/// API for something that can be copied as is.
public protocol Cloneable
{
    /// Copy this.
    func clone() -> Self
}

/// A layer that applies normalization, containing weights.
public class LayerWeightsNormalization: Codable, Cloneable
{
    /// Number of independent units of normalization.
    let _nbNeurons: Int
    /// Number of elements in one normalization unit.
    var _nbElems = 0
    
    /// Cache for weights before calling `initKernel` API.
    var _weightsList = [Float]()
    
    /// Weights in the CPU execution context.
    var weights: [Float]
    {
        get {
            return _weightsList
        }
        set {
            _weightsList = newValue
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case nbNeurons
        case nbElems
        case weights
    }
    
    ///
    /// Create a layer with independent units of normalization.
    ///
    /// - Parameter nbNeurons: Number of independent units.
    ///
    init(nbNeurons: Int)
    {
        _nbNeurons = nbNeurons
    }
    
    ///
    /// Create a layer with independent units of normalization.
    ///
    /// - Parameter layer: The layer with the structure we want to apply the normalization to .
    ///
    convenience init(_ layer: BN2D)
    {
        self.init(nbNeurons: layer.nbChannels)
    }
    
    ///
    /// Create a layer with independent units of normalization.
    ///
    /// - Parameter layer: The layer with the structure we want to apply the normalization to .
    ///
    convenience init(_ layer: InstanceNorm2D)
    {
        self.init(nbNeurons: layer.nbChannels)
    }
    
    ///
    /// Create a layer with independent units of normalization.
    ///
    /// - Parameter layer: The layer with the structure we want to apply the normalization to .
    ///
    convenience init(_ layer: AdaIN)
    {
        self.init(nbNeurons: layer.nbChannels)
    }
    
    ///
    /// Create a layer with independent units of normalization.
    ///
    /// - Parameter layer: The layer with the structure we want to apply the normalization to .
    ///
    convenience init(_ layer: LayerNormSeq)
    {
        self.init(nbNeurons: layer.nbNeurons)
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
        
        _nbNeurons = try container.decode(Int.self, forKey: .nbNeurons)
        _nbElems = try container.decode(Int.self, forKey: .nbElems)
        
        _weightsList = try container.decode([Float].self, forKey: .weights)
    }
    
    ///
    /// Copy a normalization layer.
    ///
    /// - Parameter norm: The layer to copy.
    ///
    init(norm: LayerWeightsNormalization)
    {
        _nbNeurons = norm._nbNeurons
        _nbElems = norm._nbElems
        
        let weights = norm.weights
        self.weights = weights
    }
    
    /// Copy this.
    public func clone() -> Self
    {
        return LayerWeightsNormalization(norm: self) as! Self
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
    public func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        
        try container.encode(_nbNeurons, forKey: .nbNeurons)
        try container.encode(_nbElems, forKey: .nbElems)
        
        let weights = self.weights
        try container.encode(weights, forKey: .weights)
    }
}

/// A layer that applies batch normalization, containing weights and stats.
public class LayerWeightsStatsNormalization: LayerWeightsNormalization
{
    /// Cache for stats before calling `initKernel` API.
    var _statsList = [Float]()
    
    /// Stats in the CPU execution context.
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
        case stats
    }
    
    ///
    /// Create a layer with independent units of normalization.
    ///
    /// - Parameter nbNeurons: Number of independent units.
    ///
    override init(nbNeurons: Int)
    {
        super.init(nbNeurons: nbNeurons)
    }
    
    ///
    /// Create a layer with independent units of normalization.
    ///
    /// - Parameter layer: The layer with the structure we want to apply the normalization to .
    ///
    convenience init(_ layer: BN2D)
    {
        self.init(nbNeurons: layer.nbChannels)
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
        _statsList = try container.decode([Float].self, forKey: .stats)
        try super.init(from: decoder)
    }
    
    ///
    /// Copy a normalization layer.
    ///
    /// - Parameter norm: The layer to copy.
    ///
    init(norm: LayerWeightsStatsNormalization)
    {
        super.init(norm: norm)
        
        let stats = norm.stats
        self.stats = stats
    }
    
    /// Copy this.
    public override func clone() -> Self
    {
        return LayerWeightsStatsNormalization(norm: self) as! Self
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
        
        let stats = self.stats
        try container.encode(stats, forKey: .stats)
        
        try super.encode(to: encoder)
    }
}

/// A layer that applies batch normalization in the CPU execution context.
public class BatchNormalization: LayerWeightsStatsNormalization
{
    /// Slight modification to avoid "divide by 0" errors.
    let _Ɛ: Double = 1e-5
    
    ///
    /// Array of weights to scale the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _Ɣ: WeightArrays! = nil
    ///
    /// Array of biases to add to the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _β: WeightArrays! = nil
    
    ///
    /// List of deviations of data for the different independent batch normalization units.
    /// Shape ~ (nbNeurons,).
    ///
    var _σ2 = [Double]()
    
    ///
    /// List of global averages of data for the different independent batch normalization units.
    /// Shape ~ (nbNeurons,).
    ///
    var _Eμ = [Double]()
    ///
    /// List of global deviations of data for the different independent batch normalization units.
    /// Shape ~ (nbNeurons,).
    ///
    var _Eσ2 = [Double]()
    
    ///
    /// The list of data normalized without taking into account the biases and the weights.
    /// Shape ~ (nbNeurons, (batch x height x width)).
    ///
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
            if newValue.count > 0 && newValue.count != 2 * _nbNeurons
            {
                fatalError(
                    "Weights do not have the expected number of elements."
                )
            }
            super.weights = newValue
        }
    }
    
    /// Stats in the CPU execution context.
    override var stats: [Float]
    {
        get {
            if _Eμ.count == 0
            {
                return super.stats
            }
            
            var statsTmp = [Float]()
            for depth in 0..<_nbNeurons
            {
                statsTmp.append(Float(_Eμ[depth]))
            }
            for depth in 0..<_nbNeurons
            {
                statsTmp.append(Float(_Eσ2[depth]))
            }
            return statsTmp
        }
        set {
            if newValue.count > 0 &&
               newValue.count != 2 * _nbNeurons
            {
                fatalError(
                    "Stats do not have the expected number of elements."
                )
            }
            super.stats = newValue
        }
    }
    
    /// Copy this.
    public override func clone() -> Self
    {
        return BatchNormalization(norm: self) as! Self
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
        
        _σ2 = [Double](repeating: 0.0, count: _nbNeurons)
        _xHat = [[Double]](repeating: [], count: _nbNeurons)
    }
    
    ///
    /// Initialize weights in the CPU execution context.
    ///
    /// Their momentum state is also reset.
    /// Note that we also initialize the delta which are independent on the batch size.
    ///
    func initWeights()
    {
        _β = WeightArrays(_nbNeurons)
        _Ɣ = WeightArrays(_nbNeurons)
        if _weightsList.count == 0
        {
            for depth in 0..<_nbNeurons
            {
                _Ɣ.w[depth] = 1.0
                _β.w[depth] = 0.0
            }
        }
        else
        {
            for depth in 0..<_nbNeurons
            {
                _Ɣ.w[depth] = Double(_weightsList[depth])
                _β.w[depth] = Double(_weightsList[_nbNeurons + depth])
            }
            _weightsList = []
        }
    }
    
    /// Initialize stats in the CPU execution context.
    func initStats()
    {
        _Eμ = [Double](repeating: 0.0, count: _nbNeurons)
        _Eσ2 = [Double](repeating: 0.0, count: _nbNeurons)
        
        if _statsList.count > 0
        {
            for depth in 0..<_nbNeurons
            {
                _Eμ[depth] = Double(_statsList[depth])
                _Eσ2[depth] = Double(_statsList[_nbNeurons + depth])
            }
            _statsList = []
        }
    }
    
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    func forwardGC(_ layer: BN2D)
    {
        let nbGC = layer.nbGC
        let nbChannels = layer.nbChannels
        let Ɛ = layer.Ɛ
        
        Concurrency.slice(nbChannels)
        {
            (depth: Int) in
            
            for elem in 0..<nbGC
            {
                let outs: [Double]
                if elem >= nbGC-4*nbChannels && elem < nbGC-2*nbChannels &&
                   depth == (elem-nbGC+4*nbChannels)/2
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
                else if elem >= nbGC-2*nbChannels &&
                        depth == (elem-nbGC+2*nbChannels)/2
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
        layer._normGPU?.applyWeights(norm: self)
        forwardGC(layer)
    }
    
    /// Apply the forward pass in the CPU execution context.
    func forward(_ layer: BN2D)
    {
        let nbChannels = layer.nbChannels
        
        if layer.phase != nil && layer.phase! == .Training
        {
            _Eμ.withUnsafeMutableBufferPointer { EμPointer in
            _Eσ2.withUnsafeMutableBufferPointer { Eσ2Pointer in
            _xHat.withUnsafeMutableBufferPointer { xHatPointer in
            _σ2.withUnsafeMutableBufferPointer { σ2Pointer in
            Concurrency.slice(nbChannels)
            {
                (depth: Int) in
                
                let (outs, xHat, μ, σ2) = Normalization.forward(
                    outs: layer.getOuts(depth),
                    β: _β.w[depth],
                    Ɣ: _Ɣ.w[depth]
                )
                layer.setOuts(depth: depth, outs: outs)
                
                xHatPointer[depth] = xHat
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
            Concurrency.slice(nbChannels)
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
        let nbChannels = layer.nbChannels
        
        if layer.phase != nil && layer.phase! == .Training
        {
            var deltaβ = [Double](repeating: 0.0, count: nbChannels)
            var deltaƔ = [Double](repeating: 0.0, count: nbChannels)
            
            for depth in 0..<nbChannels
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
            
            for depth in 0..<nbChannels
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
            for depth in 0..<nbChannels
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
    func collectWeights() -> [WeightArrays]
    {
        return [_Ɣ, _β]
    }
}

/// A layer that applies batch normalization in the GPU execution context.
class BatchNormalizationGPU: LayerWeightsStatsNormalization
{
    ///
    /// Buffer of weights to scale the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _Ɣ: WeightBuffers! = nil
    ///
    /// Buffer of biases to add to the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _β: WeightBuffers! = nil
    
    ///
    /// Buffer of averages of data for the different independent batch normalization units.
    /// Shape ~ (nbNeurons,).
    ///
    var _μ: FloatBuffer! = nil
    ///
    /// Buffer of global averages of data for the different independent batch normalization units.
    /// Shape ~ (nbNeurons,).
    ///
    var _Eμ: FloatBuffer! = nil
    ///
    /// Buffer of deviations of data for the different independent batch normalization units.
    /// Shape ~ (nbNeurons,).
    ///
    var _σ2: FloatBuffer! = nil
    ///
    /// Buffer of global deviations of data for the different independent batch normalization units.
    /// Shape ~ (nbNeurons,).
    ///
    var _Eσ2: FloatBuffer! = nil
    
    ///
    /// Buffer of data normalized without taking into account the biases and the weights.
    /// Shape ~ (batch, nbNeurons, height, width).
    ///
    var _xHat: FloatBuffer! = nil
    
    ///
    /// Buffer used to compute backward pass.
    /// Shape ~ (nbNeurons,).
    ///
    var _sum1: FloatBuffer! = nil
    ///
    /// Buffer used to compute backward pass.
    /// Shape ~ (nbNeurons,).
    ///
    var _sum2: FloatBuffer! = nil
   
    /// GPU device on which model is executed.
    var _deviceID = 0
    
    /// Weights in the GPU execution context.
    override var weights: [Float]
    {
        get {
            if _Ɣ == nil
            {
                return super.weights
            }
            
            var weightsTmp = _Ɣ!.w.download()
            weightsTmp += _β!.w.download()
            return weightsTmp
        }
        set {
            if newValue.count > 0 && newValue.count != 2 * _nbNeurons
            {
                fatalError(
                    "Weights do not have the expected number of elements."
                )
            }
            super.weights = newValue
        }
    }
    
    /// Stats in the GPU execution context.
    override var stats: [Float]
    {
        get {
            if _Eμ == nil
            {
                return super.stats
            }
            
            var statsTmp = _Eμ.download()
            statsTmp += _Eσ2.download()
            return statsTmp
        }
        set {
            if newValue.count > 0 && newValue.count != 2 * _nbNeurons
            {
                fatalError(
                    "Stats do not have the expected number of elements."
                )
            }
            super.stats = newValue
        }
    }
    
    /// Copy this.
    public override func clone() -> Self
    {
        return BatchNormalizationGPU(norm: self) as! Self
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
        
        _Ɣ?.reset()
        _β?.reset()
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
        _β = WeightBuffers(nbElems: _nbNeurons, deviceID: _deviceID)
        _Ɣ = WeightBuffers(nbElems: _nbNeurons, deviceID: _deviceID)
        
        if _weightsList.count == 0
        {
            _weightsList = [Float](repeating: 0.0, count: 2 * _nbNeurons)
            for depth in 0..<_nbNeurons
            {
                _weightsList[depth] = 1.0
            }
        }
        
        _Ɣ.w.initialize(array: &_weightsList)
        _β.w.initialize(array: &_weightsList, start: _nbNeurons)
        
        _weightsList = []
    }
    
    /// Initialize stats in the GPU execution context.
    func initStats()
    {
        _Eμ = FloatBuffer(nbElems: _nbNeurons, deviceID: _deviceID)
        _Eσ2 = FloatBuffer(nbElems: _nbNeurons, deviceID: _deviceID)
        
        if _statsList.count != 0
        {
            _Eμ.initialize(array: &_statsList)
            _Eσ2.initialize(array: &_statsList, start: _nbNeurons)
        }
        else
        {
            _Eμ.initialize()
            _Eσ2.initialize()
        }
        _statsList = []
    }
    
    ///
    /// Get the weights and biases back to the CPU execution context.
    ///
    /// This function is necessary for the Gradient Checking in the GPU execution context.
    ///
    /// - Parameter norm: The layer in the CPU execution context.
    ///
    func applyWeights(norm: BatchNormalization)
    {
        let weights = self.weights
        for depth in 0..<_nbNeurons
        {
            norm._Ɣ.w[depth] = Double(weights[depth])
            norm._β.w[depth] = Double(weights[_nbNeurons + depth])
        }
    }
    
    /// Apply the forward pass in the GPU execution context.
    func forward(_ layer: BN2D)
    {
        if layer.phase != nil && layer.phase! == .Training
        {
            _forwardTraining(layer)
        }
        else
        {
            _fowardInference(layer)
        }
    }
    
    /// Compute the averages of the different independent batch normalization units.
    private func _computeμ(_ layer: BN2D)
    {
        let nbChannels = layer.nbChannels
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pFirstCall: [UInt32] = _nbElems == 0 ? [1] : [0]
        
        if _μ == nil
        {
            _μ = FloatBuffer(nbElems: _nbNeurons, deviceID: _deviceID)
        }
        
        let command = MetalKernel.get.createCommand(
            "computeBNConvμ", deviceID: _deviceID
        )
        command.setBuffer(layer.outs.metal(), atIndex: 0)
        command.setBytes(pNbChannels, atIndex: 1)
        command.setBytes(pNbBatch, atIndex: 2)
        command.setBytes(pDimensions, atIndex: 3)
        command.setBytes(pFirstCall, atIndex: 4)
        command.setBuffer(_μ.metal(), atIndex: 5)
        command.setBuffer(_Eμ.metal(), atIndex: 6)
        
        command.dispatchThreads(_nbNeurons)
        command.enqueue()
    }
    
    /// Compute the deviations of the different independent batch normalization units.
    private func _computeσ2(_ layer: BN2D)
    {
        let nbChannels = layer.nbChannels
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pFirstCall: [UInt32] = _nbElems == 0 ? [1] : [0]
        
        if _σ2 == nil
        {
            _σ2 = FloatBuffer(nbElems: _nbNeurons, deviceID: _deviceID)
        }
        
        let command = MetalKernel.get.createCommand(
            "computeBNConvσ2", deviceID: _deviceID
        )
        command.setBuffer(layer.outs.metal(), atIndex: 0)
        command.setBuffer(_μ.metal(), atIndex: 1)
        command.setBytes(pNbChannels, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pDimensions, atIndex: 4)
        command.setBytes(pFirstCall, atIndex: 5)
        command.setBuffer(_σ2.metal(), atIndex: 6)
        command.setBuffer(_Eσ2.metal(), atIndex: 7)
        
        command.dispatchThreads(_nbNeurons)
        command.enqueue()
    }
    
    /// Apply the forward training pass in the GPU execution context.
    private func _forwardTraining(_ layer: BN2D)
    {
        _computeμ(layer)
        _computeσ2(layer)
        
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        if _xHat == nil
        {
            _xHat = FloatBuffer(nbElems: 
                batchSize * _nbNeurons * width * height,
                deviceID: _deviceID
            )
        }
        
        let command = MetalKernel.get.createCommand(
            "forwardBNConvTraining", deviceID: _deviceID
        )
        command.setBuffer(_β.w.metal(), atIndex: 0)
        command.setBuffer(_Ɣ.w.metal(), atIndex: 1)
        command.setBuffer(_μ.metal(), atIndex: 2)
        command.setBuffer(_σ2.metal(), atIndex: 3)
        command.setBytes(pNbChannels, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
        command.setBytes(pDimensions, atIndex: 6)
        command.setBuffer(layer.outs.metal(), atIndex: 7)
        command.setBuffer(_xHat.metal(), atIndex: 8)
        
        command.dispatchThreads(
            width: _nbNeurons * width,
            height: batchSize * height
        )
        command.enqueue()
        
        if _nbElems == 0
        {
            _nbElems = batchSize * width * height
        }
    }
    
    /// Apply the forward inference pass in the GPU execution context.
    private func _fowardInference(_ layer: BN2D)
    {
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pM: [UInt32] = [UInt32(_nbElems)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        let command = MetalKernel.get.createCommand(
            "forwardBNConvInference",
            deviceID: _deviceID
        )
        command.setBuffer(_β.w.metal(), atIndex: 0)
        command.setBuffer(_Ɣ.w.metal(), atIndex: 1)
        command.setBuffer(_Eμ.metal(), atIndex: 2)
        command.setBuffer(_Eσ2.metal(), atIndex: 3)
        command.setBytes(pNbChannels, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
        command.setBytes(pM, atIndex: 6)
        command.setBytes(pDimensions, atIndex: 7)
        command.setBuffer(layer.outs.metal(), atIndex: 8)
    
        command.dispatchThreads(
            width: _nbNeurons * width,
            height: batchSize * height
        )
        command.enqueue()
    }
    
    /// Apply the backward pass in the GPU execution context.
    func backward(_ layer: BN2D)
    {
        if layer.phase != nil && layer.phase! == .Training
        {
            _backwardTraining(layer)
        }
        else
        {
            _backwardInference(layer)
        }
    }
    
    /// Compute the gradients of weights  in the GPU execution context.
    private func _backwardWeights(_ layer: BN2D)
    {
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pAccumulate: [UInt32] = layer.accumulateDeltaWeights ? [1] : [0]
        
        if _sum1 == nil
        {
            _sum1 = FloatBuffer(nbElems: _nbNeurons, deviceID: _deviceID)
            _sum2 = FloatBuffer(nbElems: _nbNeurons, deviceID: _deviceID)
        }
        
        let command = MetalKernel.get.createCommand(
            "backwardWeightsBNConv", deviceID: _deviceID
        )
        command.setBuffer(layer.delta.metal(), atIndex: 0)
        command.setBuffer(_xHat.metal(), atIndex: 1)
        command.setBuffer(_Ɣ.w.metal(), atIndex: 2)
        command.setBytes(pNbChannels, atIndex: 3)
        command.setBytes(pNbBatch, atIndex: 4)
        command.setBytes(pDimensions, atIndex: 5)
        command.setBytes(pAccumulate, atIndex: 6)
        command.setBuffer(_sum1.metal(), atIndex: 7)
        command.setBuffer(_sum2.metal(), atIndex: 8)
        command.setBuffer(_Ɣ.g.metal(), atIndex: 9)
        command.setBuffer(_β.g.metal(), atIndex: 10)
        
        command.dispatchThreads(_nbNeurons)
        command.enqueue()
    }
    
    /// Apply the backward training pass in the GPU execution context.
    private func _backwardTraining(_ layer: BN2D)
    {
        _backwardWeights(layer)
        
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        let command = MetalKernel.get.createCommand(
            "backwardBNConvTraining", deviceID: _deviceID
        )
        command.setBuffer(_σ2.metal(), atIndex: 0)
        command.setBuffer(_xHat.metal(), atIndex: 1)
        command.setBuffer(_Ɣ.w.metal(), atIndex: 2)
        command.setBuffer(_sum1.metal(), atIndex: 3)
        command.setBuffer(_sum2.metal(), atIndex: 4)
        command.setBytes(pNbChannels, atIndex: 5)
        command.setBytes(pNbBatch, atIndex: 6)
        command.setBytes(pDimensions, atIndex: 7)
        command.setBuffer(layer.delta.metal(), atIndex: 8)
        
        command.dispatchThreads(
            width: _nbNeurons * width,
            height: batchSize * height
        )
        command.enqueue()
    }
    
    /// Apply the backward inference pass in the GPU execution context.
    private func _backwardInference(_ layer: BN2D)
    {
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pM: [UInt32] = [UInt32(_nbElems)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        let command = MetalKernel.get.createCommand(
            "backwardBNConvInference", deviceID: _deviceID
        )
        command.setBuffer(_Ɣ.w.metal(), atIndex: 0)
        command.setBuffer(_Eσ2.metal(), atIndex: 1)
        command.setBytes(pNbChannels, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pM, atIndex: 4)
        command.setBytes(pDimensions, atIndex: 5)
        command.setBuffer(layer.delta.metal(), atIndex: 6)
        
        command.dispatchThreads(
            width: _nbNeurons * width,
            height: batchSize * height
        )
        command.enqueue()
    }
    
    /// Get the weights in the GPU execution context.
    func collectWeights() -> [WeightBuffers]
    {
        return [_Ɣ, _β]
    }
}

/// A layer that applies instance normalization in the CPU execution context.
public class InstanceNormalization: LayerWeightsNormalization
{
    /// Slight modification to avoid "divide by 0" errors.
    let _Ɛ: Double = 1e-5
    
    ///
    /// Array of weights to scale the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _Ɣ: WeightArrays! = nil
    ///
    /// Array of biases to add to the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _β: WeightArrays! = nil
    
    ///
    /// List of deviations of data for the different independent batch normalization units.
    /// Shape ~ ((batch x nbNeurons),).
    ///
    var _σ2 = [Double]()
    
    ///
    /// The list of data normalized without taking into account the biases and the weights.
    /// Shape ~ ((batch x nbNeurons), (height x width)).
    ///
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
            if newValue.count > 0 && newValue.count != 2 * _nbNeurons
            {
                fatalError(
                    "Weights do not have the expected number of elements."
                )
            }
            super.weights = newValue
        }
    }
    
    /// Copy this.
    public override func clone() -> Self
    {
        return InstanceNormalization(norm: self) as! Self
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
        _σ2 = []
        _xHat = []
        
        _Ɣ?.reset()
        _β?.reset()
    }
    
    ///
    /// Initialize weights in the CPU execution context.
    ///
    /// Their momentum state is also reset.
    /// Note that we also initialize the delta which are independent on the batch size.
    ///
    func initWeights()
    {
        _β = WeightArrays(_nbNeurons)
        _Ɣ = WeightArrays(_nbNeurons)
        if _weightsList.count == 0
        {
            for depth in 0..<_nbNeurons
            {
                _Ɣ.w[depth] = 1.0
                _β.w[depth] = 0.0
            }
        }
        else
        {
            for depth in 0..<_nbNeurons
            {
                _Ɣ.w[depth] = Double(_weightsList[depth])
                _β.w[depth] = Double(_weightsList[_nbNeurons + depth])
            }
            _weightsList = []
        }
    }
    
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    func forwardGC(_ layer: InstanceNorm2D)
    {
        let nbGC = layer.nbGC
        let nbChannels = layer.nbChannels
        let Ɛ = layer.Ɛ
        
        Concurrency.slice(nbChannels)
        {
            (depth: Int) in
            
            for batch in 0..<layer.batchSize {
            for elem in 0..<nbGC
            {
                let outs: [Double]
                if elem >= nbGC-4*nbChannels && elem < nbGC-2*nbChannels &&
                   depth == (elem-nbGC+4*nbChannels)/2
                {
                    if elem % 2 == 0
                    {
                        outs = Normalization.forwardGC(
                            outs: layer.getOutsGC(
                                depth: depth, batch: batch, elem: elem
                            ),
                            β: _β.w[depth],
                            Ɣ: _Ɣ.w[depth]+Ɛ
                        )
                    }
                    else
                    {
                        outs = Normalization.forwardGC(
                            outs: layer.getOutsGC(
                                depth: depth, batch: batch, elem: elem
                            ),
                            β: _β.w[depth],
                            Ɣ: _Ɣ.w[depth]-Ɛ
                        )
                    }
                }
                else if elem >= nbGC-2*nbChannels &&
                        depth == (elem-nbGC+2*nbChannels)/2
                {
                    if elem % 2 == 0
                    {
                        outs = Normalization.forwardGC(
                            outs: layer.getOutsGC(
                                depth: depth, batch: batch, elem: elem
                            ),
                            β: _β.w[depth]+Ɛ,
                            Ɣ: _Ɣ.w[depth]
                        )
                    }
                    else
                    {
                        outs = Normalization.forwardGC(
                            outs: layer.getOutsGC(
                                depth: depth, batch: batch, elem: elem
                            ),
                            β: _β.w[depth]-Ɛ,
                            Ɣ: _Ɣ.w[depth]
                        )
                    }
                }
                else
                {
                    outs = Normalization.forwardGC(
                        outs: layer.getOutsGC(
                            depth: depth, batch: batch, elem: elem
                        ),
                        β: _β.w[depth],
                        Ɣ: _Ɣ.w[depth]
                    )
                }
                layer.setOutsGC(
                    depth: depth, batch: batch, elem: elem, outs: outs
                )
            }}
        }
    }
    
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    func forwardFlowGC(_ layer: InstanceNorm2D)
    {
        layer._normGPU?.applyWeights(norm: self)
        forwardGC(layer)
    }
    
    /// Apply the forward pass in the CPU execution context.
    func forward(_ layer: InstanceNorm2D)
    {
        if _σ2.count == 0
        {
            _σ2 = [Double](
                repeating: 0.0, count: _nbNeurons * layer.batchSize
            )
            _xHat = [[Double]](
                repeating: [], count: _nbNeurons * layer.batchSize
            )
        }
        
        let nbChannels = layer.nbChannels
        
        _xHat.withUnsafeMutableBufferPointer { xHatPointer in
        _σ2.withUnsafeMutableBufferPointer { σ2Pointer in
        Concurrency.slice(nbChannels)
        {
            (depth: Int) in
            
            for batch in 0..<layer.batchSize
            {
                let (outs, xHat, _, σ2) = Normalization.forward(
                    outs: layer.getOuts(depth: depth, batch: batch),
                    β: _β.w[depth],
                    Ɣ: _Ɣ.w[depth]
                )
                layer.setOuts(depth: depth, batch: batch, outs: outs)
                
                xHatPointer[depth + nbChannels * batch] = xHat
                σ2Pointer[depth + nbChannels * batch] = σ2
            }
        }}}
    }
    
    /// Apply the forward pass in the CPU execution context.
    func forward(_ layer: AdaIN)
    {
        if _σ2.count == 0
        {
            _σ2 = [Double](
                repeating: 0.0, count: _nbNeurons * layer.batchSize
            )
            _xHat = [[Double]](
                repeating: [], count: _nbNeurons * layer.batchSize
            )
        }
        
        let nbChannels = layer.nbChannels
        
        _xHat.withUnsafeMutableBufferPointer { xHatPointer in
        _σ2.withUnsafeMutableBufferPointer { σ2Pointer in
        Concurrency.slice(nbChannels)
        {
            (depth: Int) in
            
            for batch in 0..<layer.batchSize
            {
                let β = layer.getOutStyle(
                    depth: depth + nbChannels, batch: batch
                )
                let Ɣ = layer.getOutStyle(
                    depth: depth, batch: batch
                )
                let (outs, xHat, _, σ2) = Normalization.forward(
                    outs: layer.getOutsPrev(depth: depth, batch: batch),
                    β: β,
                    Ɣ: Ɣ
                )
                layer.setOuts(depth: depth, batch: batch, outs: outs)
                
                xHatPointer[depth + nbChannels * batch] = xHat
                σ2Pointer[depth + nbChannels * batch] = σ2
            }
        }}}
    }
    
    /// Apply the backward pass in the CPU execution context.
    func backward(_ layer: InstanceNorm2D)
    {
        let nbChannels = layer.nbChannels
        
        var deltaβ = [Double](repeating: 0.0, count: nbChannels)
        var deltaƔ = [Double](repeating: 0.0, count: nbChannels)
        
        for batch in 0..<layer.batchSize {
        for depth in 0..<nbChannels
        {
            let (delta, dβ, dƔ) = Normalization.backward(
                delta: layer.getDelta(depth: depth, batch: batch),
                xHat: _xHat[depth + nbChannels * batch],
                σ2: _σ2[depth + nbChannels * batch],
                Ɣ: _Ɣ.w[depth]
            )
            layer.setDelta(depth: depth, batch: batch, delta: delta)
            
            deltaβ[depth] += dβ
            deltaƔ[depth] += dƔ
        }}
        
        for depth in 0..<nbChannels
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
    
    /// Apply the backward pass in the CPU execution context.
    func backward(_ layer: AdaIN)
    {
        let nbChannels = layer.nbChannels
        
        for batch in 0..<layer.batchSize {
        for depth in 0..<nbChannels
        {
            let Ɣ = layer.getOutStyle(
                depth: depth, batch: batch
            )
            let (delta, dβ, dƔ) = Normalization.backward(
                delta: layer.getDelta(depth: depth, batch: batch),
                xHat: _xHat[depth + nbChannels * batch],
                σ2: _σ2[depth + nbChannels * batch],
                Ɣ: Ɣ
            )
            layer.setDeltaPrev(depth: depth, batch: batch, delta: delta)
            
            layer.setDeltaStyle(
                depth: depth + nbChannels, batch: batch, delta: dβ
            )
            layer.setDeltaStyle(
                depth: depth, batch: batch, delta: dƔ
            )
        }}
    }
    
    /// Get the weights in the CPU execution context.
    func collectWeights() -> [WeightArrays]
    {
        return [_Ɣ, _β]
    }
}

/// A layer that applies instance normalization in the GPU execution context.
class InstanceNormalizationGPU: LayerWeightsNormalization
{
    ///
    /// Buffer of weights to scale the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _Ɣ: WeightBuffers! = nil
    ///
    /// Buffer of biases to add to the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _β: WeightBuffers! = nil
    
    ///
    /// Buffer of averages of data for the different independent batch normalization units.
    /// Shape ~ (batch, nbNeurons).
    ///
    var _μ: FloatBuffer! = nil
    ///
    /// Buffer of deviations of data for the different independent batch normalization units.
    /// Shape ~ (batch, nbNeurons).
    ///
    var _σ2: FloatBuffer! = nil
    
    ///
    /// Buffer of data normalized without taking into account the biases and the weights.
    /// Shape ~ (batch, nbNeurons, height, width).
    ///
    var _xHat: FloatBuffer! = nil
    
    ///
    /// Buffer used to compute backward pass.
    /// Shape ~ (nbNeurons,).
    ///
    var _sum1: FloatBuffer! = nil
    ///
    /// Buffer used to compute backward pass.
    /// Shape ~ (nbNeurons,).
    ///
    var _sum2: FloatBuffer! = nil
   
    /// GPU device on which model is executed.
    var _deviceID = 0
    
    /// Weights in the GPU execution context.
    override var weights: [Float]
    {
        get {
            if _Ɣ == nil
            {
                return super.weights
            }
            
            var weightsTmp = _Ɣ!.w.download()
            weightsTmp += _β!.w.download()
            return weightsTmp
        }
        set {
            if newValue.count > 0 && newValue.count != 2 * _nbNeurons
            {
                fatalError(
                    "Weights do not have the expected number of elements."
                )
            }
            super.weights = newValue
        }
    }
    
    /// Copy this.
    public override func clone() -> Self
    {
        return InstanceNormalizationGPU(norm: self) as! Self
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We do not clean Ɣ and β but must reset their momentum state.
    ///
    func resetKernel()
    {
        _σ2 = nil
        _xHat = nil
        _sum1 = nil
        _sum2 = nil
        
        _Ɣ?.reset()
        _β?.reset()
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
    }
    
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// Their momentum and delta state are also reset.
    ///
    func initWeights()
    {
        _β = WeightBuffers(nbElems: _nbNeurons, deviceID: _deviceID)
        _Ɣ = WeightBuffers(nbElems: _nbNeurons, deviceID: _deviceID)
        
        if _weightsList.count == 0
        {
            _weightsList = [Float](repeating: 0.0, count: 2 * _nbNeurons)
            for depth in 0..<_nbNeurons
            {
                _weightsList[depth] = 1.0
            }
        }
        
        _Ɣ.w.initialize(array: &_weightsList)
        _β.w.initialize(array: &_weightsList, start: _nbNeurons)
        
        _weightsList = []
    }
    
    ///
    /// Get the weights and biases back to the CPU execution context.
    ///
    /// This function is necessary for the Gradient Checking in the GPU execution context.
    ///
    /// - Parameter norm: The layer in the CPU execution context.
    ///
    func applyWeights(norm: InstanceNormalization)
    {
        let weights = self.weights
        for depth in 0..<_nbNeurons
        {
            norm._Ɣ.w[depth] = Double(weights[depth])
            norm._β.w[depth] = Double(weights[_nbNeurons + depth])
        }
    }
    
    /// Apply the forward pass in the GPU execution context.
    func forward(_ layer: InstanceNorm2D)
    {
        _computeμ(layer)
        _computeσ2(layer)
        
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        if _xHat == nil
        {
            _xHat = FloatBuffer(nbElems: 
                batchSize * _nbNeurons * width * height,
                deviceID: _deviceID
            )
        }
        
        let command = MetalKernel.get.createCommand(
            "forwardInstanceNormConv", deviceID: _deviceID
        )
        command.setBuffer(_β.w.metal(), atIndex: 0)
        command.setBuffer(_Ɣ.w.metal(), atIndex: 1)
        command.setBuffer(_μ.metal(), atIndex: 2)
        command.setBuffer(_σ2.metal(), atIndex: 3)
        command.setBytes(pNbChannels, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
        command.setBytes(pDimensions, atIndex: 6)
        command.setBuffer(layer.outs.metal(), atIndex: 7)
        command.setBuffer(_xHat.metal(), atIndex: 8)
        
        command.dispatchThreads(
            width: _nbNeurons * width,
            height: batchSize * height
        )
        command.enqueue()
    }
    
    /// Apply the forward pass in the GPU execution context.
    func forward(_ layer: AdaIN)
    {
        _computeμ(layer)
        _computeσ2(layer)
        
        let layerFirst = layer._layersPrev.first as! Layer2D
        let layerLast = layer._layersPrev.last as! Layer1D
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        if _xHat == nil
        {
            _xHat = FloatBuffer(nbElems: 
                batchSize * _nbNeurons * width * height,
                deviceID: _deviceID
            )
        }
        
        let command = MetalKernel.get.createCommand(
            "forwardAdaIN", deviceID: _deviceID
        )
        command.setBuffer(layerFirst.outs.metal(), atIndex: 0)
        command.setBuffer(layerLast.outs.metal(), atIndex: 1)
        command.setBuffer(_μ.metal(), atIndex: 2)
        command.setBuffer(_σ2.metal(), atIndex: 3)
        command.setBytes(pNbChannels, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
        command.setBytes(pDimensions, atIndex: 6)
        command.setBuffer(layer.outs.metal(), atIndex: 7)
        command.setBuffer(_xHat.metal(), atIndex: 8)
        
        command.dispatchThreads(
            width: _nbNeurons * width,
            height: batchSize * height
        )
        command.enqueue()
    }
    
    /// Compute the averages of the different independent normalization units.
    private func _computeμ(_ layer: InstanceNorm2D)
    {
        let nbChannels = layer.nbChannels
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        if _μ == nil
        {
            _μ = FloatBuffer(nbElems: 
                batchSize * _nbNeurons, deviceID: _deviceID
            )
        }
        
        let command = MetalKernel.get.createCommand(
            "computeInstanceNormConvμ", deviceID: _deviceID
        )
        command.setBuffer(layer.outs.metal(), atIndex: 0)
        command.setBytes(pNbChannels, atIndex: 1)
        command.setBytes(pNbBatch, atIndex: 2)
        command.setBytes(pDimensions, atIndex: 3)
        command.setBuffer(_μ.metal(), atIndex: 4)
        
        command.dispatchThreads(width: _nbNeurons, height: batchSize)
        command.enqueue()
    }
    
    /// Compute the averages of the different independent normalization units.
    private func _computeμ(_ layer: AdaIN)
    {
        let layerFirst = layer._layersPrev.first as! Layer2D
        let nbChannels = layer.nbChannels
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        if _μ == nil
        {
            _μ = FloatBuffer(nbElems: 
                batchSize * _nbNeurons, deviceID: _deviceID
            )
        }
        
        let command = MetalKernel.get.createCommand(
            "computeInstanceNormConvμ", deviceID: _deviceID
        )
        command.setBuffer(layerFirst.outs.metal(), atIndex: 0)
        command.setBytes(pNbChannels, atIndex: 1)
        command.setBytes(pNbBatch, atIndex: 2)
        command.setBytes(pDimensions, atIndex: 3)
        command.setBuffer(_μ.metal(), atIndex: 4)
        
        command.dispatchThreads(width: _nbNeurons, height: batchSize)
        command.enqueue()
    }
    
    /// Compute the deviations of the different independent normalization units.
    private func _computeσ2(_ layer: InstanceNorm2D)
    {
        let nbChannels = layer.nbChannels
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        if _σ2 == nil
        {
            _σ2 = FloatBuffer(nbElems: 
                batchSize * _nbNeurons, deviceID: _deviceID
            )
        }
        
        let command = MetalKernel.get.createCommand(
            "computeInstanceNormConvσ2", deviceID: _deviceID
        )
        command.setBuffer(layer.outs.metal(), atIndex: 0)
        command.setBuffer(_μ.metal(), atIndex: 1)
        command.setBytes(pNbChannels, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pDimensions, atIndex: 4)
        command.setBuffer(_σ2.metal(), atIndex: 5)
        
        command.dispatchThreads(width: _nbNeurons, height: batchSize)
        command.enqueue()
    }
    
    /// Compute the deviations of the different independent normalization units.
    private func _computeσ2(_ layer: AdaIN)
    {
        let layerFirst = layer._layersPrev.first as! Layer2D
        let nbChannels = layer.nbChannels
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        if _σ2 == nil
        {
            _σ2 = FloatBuffer(nbElems: 
                batchSize * _nbNeurons, deviceID: _deviceID
            )
        }
        
        let command = MetalKernel.get.createCommand(
            "computeInstanceNormConvσ2", deviceID: _deviceID
        )
        command.setBuffer(layerFirst.outs.metal(), atIndex: 0)
        command.setBuffer(_μ.metal(), atIndex: 1)
        command.setBytes(pNbChannels, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pDimensions, atIndex: 4)
        command.setBuffer(_σ2.metal(), atIndex: 5)
        
        command.dispatchThreads(width: _nbNeurons, height: batchSize)
        command.enqueue()
    }
    
    /// Apply the backward pass in the GPU execution context.
    func backward(_ layer: InstanceNorm2D)
    {
        _backwardWeights(layer)
        
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        
        let command = MetalKernel.get.createCommand(
            "backwardInstanceNormConv", deviceID: _deviceID
        )
        command.setBuffer(_σ2.metal(), atIndex: 0)
        command.setBuffer(_xHat.metal(), atIndex: 1)
        command.setBuffer(_Ɣ.w.metal(), atIndex: 2)
        command.setBuffer(_sum1.metal(), atIndex: 3)
        command.setBuffer(_sum2.metal(), atIndex: 4)
        command.setBytes(pNbChannels, atIndex: 5)
        command.setBytes(pNbBatch, atIndex: 6)
        command.setBytes(pDimensions, atIndex: 7)
        command.setBuffer(layer.delta.metal(), atIndex: 8)
        
        command.dispatchThreads(
            width: _nbNeurons * width,
            height: batchSize * height
        )
        command.enqueue()
    }
    
    /// Apply the backward pass in the GPU execution context.
    func backward(_ layer: AdaIN)
    {
        _backward(layer)
        
        let layerFirst = layer._layersPrev.first as! Layer2D
        let layerLast = layer._layersPrev.last as! Layer1D
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pDirty: [UInt32] = layerFirst.dirty ? [1] : [0]
        
        let command = MetalKernel.get.createCommand(
            "backward1AdaIN", deviceID: _deviceID
        )
        command.setBuffer(layer.delta.metal(), atIndex: 0)
        command.setBuffer(_σ2.metal(), atIndex: 1)
        command.setBuffer(_xHat.metal(), atIndex: 2)
        command.setBuffer(layerLast.outs.metal(), atIndex: 3)
        command.setBuffer(_sum1.metal(), atIndex: 4)
        command.setBuffer(_sum2.metal(), atIndex: 5)
        command.setBytes(pNbChannels, atIndex: 6)
        command.setBytes(pNbBatch, atIndex: 7)
        command.setBytes(pDimensions, atIndex: 8)
        command.setBytes(pDirty, atIndex: 9)
        command.setBuffer(layerFirst.delta.metal(), atIndex: 10)
        
        command.dispatchThreads(
            width: _nbNeurons * width,
            height: batchSize * height
        )
        command.enqueue()
    }
    
    /// Compute the gradients of weights  in the GPU execution context.
    private func _backwardWeights(_ layer: InstanceNorm2D)
    {
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pAccumulate: [UInt32] = layer.accumulateDeltaWeights ? [1] : [0]
        
        if _sum1 == nil
        {
            _sum1 = FloatBuffer(nbElems: 
                batchSize * _nbNeurons, deviceID: _deviceID
            )
            _sum2 = FloatBuffer(nbElems: 
                batchSize * _nbNeurons, deviceID: _deviceID
            )
        }
        
        let command = MetalKernel.get.createCommand(
            "backwardWeightsInstanceNormConv", deviceID: _deviceID
        )
        command.setBuffer(layer.delta.metal(), atIndex: 0)
        command.setBuffer(_xHat.metal(), atIndex: 1)
        command.setBuffer(_Ɣ.w.metal(), atIndex: 2)
        command.setBytes(pNbChannels, atIndex: 3)
        command.setBytes(pNbBatch, atIndex: 4)
        command.setBytes(pDimensions, atIndex: 5)
        command.setBytes(pAccumulate, atIndex: 6)
        command.setBuffer(_sum1.metal(), atIndex: 7)
        command.setBuffer(_sum2.metal(), atIndex: 8)
        command.setBuffer(_Ɣ.g.metal(), atIndex: 9)
        command.setBuffer(_β.g.metal(), atIndex: 10)
        
        command.dispatchThreads(_nbNeurons)
        command.enqueue()
    }
    
    /// Compute the gradients of weights  in the GPU execution context.
    private func _backward(_ layer: AdaIN)
    {
        let layerLast = layer._layersPrev.last as! Layer1D
        let batchSize = layer.batchSize
        let width = layer.width
        let height = layer.height
        
        let pNbChannels: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pDirty: [UInt32] = layerLast.dirty ? [1] : [0]
        
        if _sum1 == nil
        {
            _sum1 = FloatBuffer(nbElems: 
                batchSize * _nbNeurons, deviceID: _deviceID
            )
            _sum2 = FloatBuffer(nbElems: 
                batchSize * _nbNeurons, deviceID: _deviceID
            )
        }
        
        let command = MetalKernel.get.createCommand(
            "backward2AdaIN", deviceID: _deviceID
        )
        command.setBuffer(layer.delta.metal(), atIndex: 0)
        command.setBuffer(_xHat.metal(), atIndex: 1)
        command.setBuffer(layerLast.outs.metal(), atIndex: 2)
        command.setBytes(pNbChannels, atIndex: 3)
        command.setBytes(pNbBatch, atIndex: 4)
        command.setBytes(pDimensions, atIndex: 5)
        command.setBytes(pDirty, atIndex: 6)
        command.setBuffer(_sum1.metal(), atIndex: 7)
        command.setBuffer(_sum2.metal(), atIndex: 8)
        command.setBuffer(layerLast.delta.metal(), atIndex: 9)
        
        command.dispatchThreads(width: _nbNeurons, height: batchSize)
        command.enqueue()
    }
    
    /// Get the weights in the GPU execution context.
    func collectWeights() -> [IWeightBuffers]
    {
        return [_Ɣ, _β]
    }
}

/// A layer that applies layer normalization in the CPU execution context.
public class LayerNormalization: LayerWeightsNormalization
{
    /// Slight modification to avoid "divide by 0" errors.
    let _Ɛ: Double = 1e-5
    
    ///
    /// Array of weights to scale the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _Ɣ: WeightArrays! = nil
    ///
    /// Array of biases to add to the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _β: WeightArrays! = nil
    
    ///
    /// List of deviations of data for the different independent batch normalization units.
    /// Shape ~ ((batch x sequence),).
    ///
    var _σ2 = [Double]()
    
    ///
    /// The list of data normalized without taking into account the biases and the weights.
    /// Shape ~ ((batch x sequence), (nbNeurons)).
    ///
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
            if newValue.count > 0 && newValue.count != 2 * _nbNeurons
            {
                fatalError(
                    "Weights do not have the expected number of elements."
                )
            }
            super.weights = newValue
        }
    }
    
    /// Copy this.
    public override func clone() -> Self
    {
        return LayerNormalization(norm: self) as! Self
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
        _σ2 = []
        _xHat = []
        
        _Ɣ?.reset()
        _β?.reset()
    }
    
    ///
    /// Initialize weights in the CPU execution context.
    ///
    /// Their momentum state is also reset.
    /// Note that we also initialize the delta which are independent on the batch size.
    ///
    func initWeights()
    {
        _β = WeightArrays(_nbNeurons)
        _Ɣ = WeightArrays(_nbNeurons)
        if _weightsList.count == 0
        {
            for depth in 0..<_nbNeurons
            {
                _Ɣ.w[depth] = 1.0
                _β.w[depth] = 0.0
            }
        }
        else
        {
            for depth in 0..<_nbNeurons
            {
                _Ɣ.w[depth] = Double(_weightsList[depth])
                _β.w[depth] = Double(_weightsList[_nbNeurons + depth])
            }
            _weightsList = []
        }
    }
    
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    func forwardGC(_ layer: LayerNormSeq)
    {
        let nbGC = layer.nbGC
        let nbNeurons = layer.nbNeurons
        let Ɛ = layer.Ɛ
        
        Concurrency.slice(layer.sequence)
        {
            (seq: Int) in
            
            for batch in 0..<layer.batchSize {
            for elem in 0..<nbGC
            {
                var β = [Double]()
                var Ɣ = [Double]()
                
                if elem >= nbGC-4*nbNeurons && elem < nbGC-2*nbNeurons
                {
                    let DEPTH = (elem - nbGC + 4 * nbNeurons) / 2
                    
                    for depth in 0..<nbNeurons
                    {
                        β.append(_β.w[depth])
                    }
                    
                    if elem % 2 == 0
                    {
                        for depth in 0..<nbNeurons
                        {
                            if depth == DEPTH
                            {
                                Ɣ.append(_Ɣ.w[depth]+Ɛ)
                            }
                            else
                            {
                                Ɣ.append(_Ɣ.w[depth])
                            }
                        }
                    }
                    else
                    {
                        for depth in 0..<nbNeurons
                        {
                            if depth == DEPTH
                            {
                                Ɣ.append(_Ɣ.w[depth]-Ɛ)
                            }
                            else
                            {
                                Ɣ.append(_Ɣ.w[depth])
                            }
                        }
                    }
                }
                else if elem >= nbGC-2*nbNeurons
                {
                    let DEPTH = (elem - nbGC + 2 * nbNeurons) / 2
                    
                    for depth in 0..<nbNeurons
                    {
                        Ɣ.append(_Ɣ.w[depth])
                    }
                    
                    if elem % 2 == 0
                    {
                        for depth in 0..<nbNeurons
                        {
                            if depth == DEPTH
                            {
                                β.append(_β.w[depth]+Ɛ)
                            }
                            else
                            {
                                β.append(_β.w[depth])
                            }
                        }
                    }
                    else
                    {
                        for depth in 0..<nbNeurons
                        {
                            if depth == DEPTH
                            {
                                β.append(_β.w[depth]-Ɛ)
                            }
                            else
                            {
                                β.append(_β.w[depth])
                            }
                        }
                    }
                }
                else
                {
                    for depth in 0..<nbNeurons
                    {
                        Ɣ.append(_Ɣ.w[depth])
                        β.append(_β.w[depth])
                    }
                }
                
                let outs = Normalization.forwardGC(
                    outs: layer.getOutsGC(
                        batch: batch, seq: seq, elem: elem
                    ),
                    β: β,
                    Ɣ: Ɣ
                )
                layer.setOutsGC(
                    batch: batch, seq: seq, elem: elem, outs: outs
                )
            }}
        }
    }
    
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    func forwardFlowGC(_ layer: LayerNormSeq)
    {
        layer._normGPU?.applyWeights(norm: self)
        forwardGC(layer)
    }
    
    /// Apply the forward pass in the CPU execution context.
    func forward(_ layer: LayerNormSeq)
    {
        if _σ2.count == 0
        {
            _σ2 = [Double](
                repeating: 0.0, count: layer.batchSize * layer.sequence
            )
            _xHat = [[Double]](
                repeating: [],
                count: layer.batchSize * layer.sequence * _nbNeurons
            )
        }
        
        let sequence = layer.sequence
        var β = [Double]()
        var Ɣ = [Double]()
        for depth in 0..<_nbNeurons
        {
            β.append(_β.w[depth])
            Ɣ.append(_Ɣ.w[depth])
        }
                
        _xHat.withUnsafeMutableBufferPointer { xHatPointer in
        _σ2.withUnsafeMutableBufferPointer { σ2Pointer in
        Concurrency.slice(sequence)
        {
            (seq: Int) in
            
            for batch in 0..<layer.batchSize
            {
                let (outs, xHat, _, σ2) = Normalization.forward(
                    outs: layer.getOuts(batch: batch, seq: seq),
                    β: β,
                    Ɣ: Ɣ
                )
                layer.setOuts(batch: batch, seq: seq, outs: outs)
                
                xHatPointer[seq + sequence * batch] = xHat
                σ2Pointer[seq + sequence * batch] = σ2
            }
        }}}
    }
    
    /// Apply the backward pass in the CPU execution context.
    func backward(_ layer: LayerNormSeq)
    {
        let sequence = layer.sequence
        let nbNeurons = layer.nbNeurons
        
        var deltaβ = [Double](repeating: 0, count: nbNeurons)
        var deltaƔ = [Double](repeating: 0, count: nbNeurons)
        
        var Ɣ = [Double]()
        for depth in 0..<nbNeurons
        {
            Ɣ.append(_Ɣ.w[depth])
        }
        
        for batch in 0..<layer.batchSize {
        for seq in 0..<sequence
        {
            let delta1 = layer.getDelta(batch: batch, seq: seq)
            
            let delta2 = Normalization.backward(
                delta: delta1,
                xHat: _xHat[seq + sequence * batch],
                σ2: _σ2[seq + sequence * batch],
                Ɣ: Ɣ
            )
            layer.setDelta(batch: batch, seq: seq, delta: delta2)
            
            for depth in 0..<_nbNeurons
            {
                deltaβ[depth] += delta1[depth]
                deltaƔ[depth] +=
                    _xHat[seq + sequence * batch][depth] * delta1[depth]
            }
        }}
        
        for depth in 0..<nbNeurons
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
    
    /// Get the weights in the CPU execution context.
    func collectWeights() -> [IWeightArrays]
    {
        return [_Ɣ, _β]
    }
}

/// A layer that applies layer normalization in the GPU execution context.
class LayerNormalizationGPU: LayerWeightsNormalization
{
    ///
    /// Buffer of weights to scale the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _Ɣ: WeightBuffers! = nil
    ///
    /// Buffer of biases to add to the normalization result.
    /// Shape ~ (nbNeurons,).
    ///
    var _β: WeightBuffers! = nil
    
    ///
    /// Buffer of averages of data for the different independent batch normalization units.
    /// Shape ~ (batch, sequence).
    ///
    var _μ: FloatBuffer! = nil
    ///
    /// Buffer of deviations of data for the different independent batch normalization units.
    /// Shape ~ (batch, sequence).
    ///
    var _σ2: FloatBuffer! = nil
    
    ///
    /// Buffer of data normalized without taking into account the biases and the weights.
    /// Shape ~ (batch, sequence, nbNeurons).
    ///
    var _xHat: FloatBuffer! = nil
    
    ///
    /// Buffer used to compute backward pass.
    /// Shape ~ (batch, sequence).
    ///
    var _sum1: FloatBuffer! = nil
    ///
    /// Buffer used to compute backward pass.
    /// Shape ~ (batch, sequence).
    ///
    var _sum2: FloatBuffer! = nil
   
    /// GPU device on which model is executed.
    var _deviceID = 0
    
    /// Weights in the GPU execution context.
    override var weights: [Float]
    {
        get {
            if _Ɣ == nil
            {
                return super.weights
            }
            
            var weightsTmp = _Ɣ!.w.download()
            weightsTmp += _β!.w.download()
            return weightsTmp
        }
        set {
            if newValue.count > 0 && newValue.count != 2 * _nbNeurons
            {
                fatalError(
                    "Weights do not have the expected number of elements."
                )
            }
            super.weights = newValue
        }
    }
    
    /// Copy this.
    public override func clone() -> Self
    {
        return LayerNormalizationGPU(norm: self) as! Self
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We do not clean Ɣ and β but must reset their momentum state.
    ///
    func resetKernel()
    {
        _σ2 = nil
        _xHat = nil
        _sum1 = nil
        _sum2 = nil
        
        _Ɣ?.reset()
        _β?.reset()
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
    }
    
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// Their momentum and delta state are also reset.
    ///
    func initWeights()
    {
        _β = WeightBuffers(nbElems: _nbNeurons, deviceID: _deviceID)
        _Ɣ = WeightBuffers(nbElems: _nbNeurons, deviceID: _deviceID)
        
        if _weightsList.count == 0
        {
            _weightsList = [Float](repeating: 0.0, count: 2 * _nbNeurons)
            for depth in 0..<_nbNeurons
            {
                _weightsList[depth] = 1.0
            }
        }
        
        _Ɣ.w.initialize(array: &_weightsList)
        _β.w.initialize(array: &_weightsList, start: _nbNeurons)
        
        _weightsList = []
    }
    
    ///
    /// Get the weights and biases back to the CPU execution context.
    ///
    /// This function is necessary for the Gradient Checking in the GPU execution context.
    ///
    /// - Parameter norm: The layer in the CPU execution context.
    ///
    func applyWeights(norm: LayerNormalization)
    {
        let weights = self.weights
        for depth in 0..<_nbNeurons
        {
            norm._Ɣ.w[depth] = Double(weights[depth])
            norm._β.w[depth] = Double(weights[_nbNeurons + depth])
        }
    }
    
    /// Apply the forward pass in the GPU execution context.
    func forward(_ layer: LayerNormSeq)
    {
        _computeμ(layer)
        _computeσ2(layer)
        
        let batchSize = layer.batchSize
        let sequence = layer.sequence
        
        let pNbNeurons: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        if _xHat == nil
        {
            _xHat = FloatBuffer(nbElems: 
                batchSize * sequence * _nbNeurons,
                deviceID: _deviceID
            )
        }
        
        let kernel = _nbNeurons % 4 == 0 ?
            "forwardLayerNormSeq4" : "forwardLayerNormSeq"
        let coeff = _nbNeurons % 4 == 0 ? 4 : 1
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: _deviceID
        )
        command.setBuffer(_β.w.metal(), atIndex: 0)
        command.setBuffer(_Ɣ.w.metal(), atIndex: 1)
        command.setBuffer(_μ.metal(), atIndex: 2)
        command.setBuffer(_σ2.metal(), atIndex: 3)
        command.setBytes(pNbNeurons, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
        command.setBytes(pSequence, atIndex: 6)
        command.setBuffer(layer.outs.metal(), atIndex: 7)
        command.setBuffer(_xHat.metal(), atIndex: 8)
        
        command.dispatchThreads(
            width: _nbNeurons / coeff,
            height: batchSize * sequence
        )
        command.enqueue()
    }
    
    /// Compute the averages of the different independent normalization units.
    private func _computeμ(_ layer: LayerNormSeq)
    {
        let batchSize = layer.batchSize
        let sequence = layer.sequence
        
        let pNbNeurons: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        if _μ == nil
        {
            _μ = FloatBuffer(nbElems: 
                batchSize * sequence, deviceID: _deviceID
            )
        }
        
        let kernel = _nbNeurons % 4 == 0 ?
            "computeLayerNormSeqμ4" : "computeLayerNormSeqμ"
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: _deviceID
        )
        command.setBuffer(layer.outs.metal(), atIndex: 0)
        command.setBytes(pNbNeurons, atIndex: 1)
        command.setBytes(pNbBatch, atIndex: 2)
        command.setBytes(pSequence, atIndex: 3)
        command.setBuffer(_μ.metal(), atIndex: 4)
        
        command.dispatchThreads(width: sequence, height: batchSize)
        command.enqueue()
    }
    
    /// Compute the deviations of the different independent normalization units.
    private func _computeσ2(_ layer: LayerNormSeq)
    {
        let batchSize = layer.batchSize
        let sequence = layer.sequence
        
        let pNbNeurons: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        if _σ2 == nil
        {
            _σ2 = FloatBuffer(nbElems: 
                batchSize * sequence, deviceID: _deviceID
            )
        }
        
        let kernel = _nbNeurons % 4 == 0 ?
            "computeLayerNormSeqσ24" : "computeLayerNormSeqσ2"
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: _deviceID
        )
        command.setBuffer(layer.outs.metal(), atIndex: 0)
        command.setBuffer(_μ.metal(), atIndex: 1)
        command.setBytes(pNbNeurons, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pSequence, atIndex: 4)
        command.setBuffer(_σ2.metal(), atIndex: 5)
        
        command.dispatchThreads(width: sequence, height: batchSize)
        command.enqueue()
    }
    
    /// Apply the backward pass in the GPU execution context.
    func backward(_ layer: LayerNormSeq)
    {
        _backwardWeights1(layer)
        _backwardWeights2(layer)
        
        let batchSize = layer.batchSize
        let sequence = layer.sequence
        
        let pNbNeurons: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        let kernel = _nbNeurons % 4 == 0 ?
            "backwardLayerNormSeq4" : "backwardLayerNormSeq"
        let coeff = _nbNeurons % 4 == 0 ? 4 : 1
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: _deviceID
        )
        command.setBuffer(_σ2.metal(), atIndex: 0)
        command.setBuffer(_xHat.metal(), atIndex: 1)
        command.setBuffer(_Ɣ.w.metal(), atIndex: 2)
        command.setBuffer(_sum1.metal(), atIndex: 3)
        command.setBuffer(_sum2.metal(), atIndex: 4)
        command.setBytes(pNbNeurons, atIndex: 5)
        command.setBytes(pNbBatch, atIndex: 6)
        command.setBytes(pSequence, atIndex: 7)
        command.setBuffer(layer.delta.metal(), atIndex: 8)
        
        command.dispatchThreads(
            width: _nbNeurons / coeff,
            height: batchSize * sequence
        )
        command.enqueue()
    }
    
    /// Compute the gradients of weights  in the GPU execution context.
    private func _backwardWeights1(_ layer: LayerNormSeq)
    {
        let batchSize = layer.batchSize
        let sequence = layer.sequence
        
        let pNbNeurons: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        
        if _sum1 == nil
        {
            _sum1 = FloatBuffer(nbElems: 
                batchSize * sequence, deviceID: _deviceID
            )
            _sum2 = FloatBuffer(nbElems: 
                batchSize * sequence, deviceID: _deviceID
            )
        }
        
        let kernel = _nbNeurons % 4 == 0 ?
            "backwardWeights1LayerNormSeq4" : "backwardWeights1LayerNormSeq"
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: _deviceID
        )
        command.setBuffer(layer.delta.metal(), atIndex: 0)
        command.setBuffer(_xHat.metal(), atIndex: 1)
        command.setBuffer(_Ɣ.w.metal(), atIndex: 2)
        command.setBytes(pNbNeurons, atIndex: 3)
        command.setBytes(pNbBatch, atIndex: 4)
        command.setBytes(pSequence, atIndex: 5)
        command.setBuffer(_sum1.metal(), atIndex: 6)
        command.setBuffer(_sum2.metal(), atIndex: 7)
        
        command.dispatchThreads(width: sequence, height: batchSize)
        command.enqueue()
    }
    
    /// Compute the gradients of weights  in the GPU execution context.
    private func _backwardWeights2(_ layer: LayerNormSeq)
    {
        let batchSize = layer.batchSize
        let sequence = layer.sequence
        
        let pNbNeurons: [UInt32] = [UInt32(_nbNeurons)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pSequence: [UInt32] = [UInt32(sequence)]
        let pAccumulate: [UInt32] = layer.accumulateDeltaWeights ? [1] : [0]
        
        let kernel = _nbNeurons % 4 == 0 ?
            "backwardWeights2LayerNormSeq4" : "backwardWeights2LayerNormSeq"
        let coeff = _nbNeurons % 4 == 0 ? 4 : 1
        let command = MetalKernel.get.createCommand(
            kernel, deviceID: _deviceID
        )
        command.setBuffer(layer.delta.metal(), atIndex: 0)
        command.setBuffer(_xHat.metal(), atIndex: 1)
        command.setBytes(pNbNeurons, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBytes(pSequence, atIndex: 4)
        command.setBytes(pAccumulate, atIndex: 5)
        command.setBuffer(_Ɣ.g.metal(), atIndex: 6)
        command.setBuffer(_β.g.metal(), atIndex: 7)
        
        command.dispatchThreads(_nbNeurons / coeff)
        command.enqueue()
    }
    
    /// Get the weights in the GPU execution context.
    func collectWeights() -> [IWeightBuffers]
    {
        return [_Ɣ, _β]
    }
}
