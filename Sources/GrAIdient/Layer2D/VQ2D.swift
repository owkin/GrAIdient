//
// VQ2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 29/03/2023.
//

import Foundation
import MetalKit

/// Error occuring during the layer forward or backward propagation.
public enum VQError: Error
{
    /// Could not find a positive index value.
    case IndexValue
    /// Call to loss API is redundant.
    case RedundantLoss
}

extension VQError: CustomStringConvertible
{
    public var description: String
    {
        switch self
        {
        case .IndexValue:
            return "Could not find a positive index value."
        case .RedundantLoss:
            return "Call to loss API is redundant."
        }
    }
}

/// Layer with a 2D shape neural structure and weights.
public class VQ2D: LayerOutput2D, LayerWeightInit
{
    /// The number of vector approximations.
    public let K: Int
    
    /// Coefficient for commitment.
    public var beta: Double = 1.0
    
    ///
    /// Indices of maximal elements.
    /// Shape ~ (batch, height, width).
    ///
    public var indices: MetalBuffer<Int32>! = nil
    
    ///
    /// Grid of weights.
    /// Shape ~ (K, nbChannels).
    ///
    var _wArrays: WeightGrids! = nil
    
    ///
    /// Buffer of weights.
    /// Shape ~ (K, nbChannels).
    ///
    var _wBuffers: IWeightBuffers! = nil
    
    ///
    /// Buffer of gradients per sample for biases.
    /// Shape ~ (batch, K, nbChannels).
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
            for depth in 0..<nbChannels
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
            return (nbChannels, K)
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case K
        case beta
        case weights
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - K: The number of vector approximations.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                K: Int,
                params: GrAI.Model.Params)
    {
        self.K = K
        try! super.init(layerPrev: layerPrev, params: params)
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
        let layerPrev = mapping[idPrev] as! Layer2D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = VQ2D(
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
        
        _wArrays = WeightGrids(width: nbChannels, height: K)
        
        for k in 0..<K {
        for depth in 0..<nbChannels
        {
            let offset = depth + nbChannels * k
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
            nbElems: K * nbChannels,
            deviceID: deviceID
        )
        
        let weightsPtr = _wBuffers.w_p!.shared.buffer
        for elem in 0..<K * nbChannels
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
                batchSize * height * width,
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
                batchSize * K * nbChannels, deviceID: deviceID
            )
        }
        
        if indices == nil
        {
            indices = MetalPrivateBuffer<Int32>(
                batchSize * height * width,
                deviceID: deviceID
            )
        }
    }
    
    ///
    /// Check and setup ground truth in the CPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///
    public override func checkGroundTruthCPU<T: BinaryFloatingPoint>(
        _ groundTruth: [T],
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int) throws
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// Setup groundTruth state in the GPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - format: The data format.
    ///
    public override func checkGroundTruthGPU<T: BinaryFloatingPoint>(
        _ groundTruth: [T],
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int,
        format: ImageFormat) throws
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// Check and setup ground truth in the GPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///
    public override func checkGroundTruthGPU(
        _ groundTruth: MetalBuffer<Float>,
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int) throws
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons
            let indicesPtr = (indices as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                var minIndex = -1
                var minValue: Double? = nil
                
                for k in 0..<K
                {
                    var value: Double = 0.0
                    for depth in 0..<nbChannels
                    {
                        let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
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
                
                for depth in 0..<nbChannels
                {
                    neurons[depth].get(i, j)!.v[elem].out =
                        _wArrays.w(minIndex, depth)
                }
                indicesPtr[j + (elem * height + i) * width] = Int32(minIndex)
            }}}
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pK: [UInt32] = [UInt32(K)]
            
            let command = MetalKernel.get.createCommand(
                "vq2DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(_wBuffers.w.metal, atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pDimensions, atIndex: 3)
            command.setBytes(pK, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            command.setBuffer(indices.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: height * width,
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
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            let indicesPtr = (indices as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                let minIndex = Int(indicesPtr[j + (elem * height + i) * width])
                if minIndex >= 0 {
                for depth in 0..<nbChannels
                {
                    let vq = _wArrays.w(minIndex, depth)
                    let deltaCur = neurons[depth].get(i, j)!.v[elem].delta
                    let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta =
                            deltaCur
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            deltaCur
                    }
                    
                    // Commitment term.
                    neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                        beta / Double(batchSize * height * width) *
                        2.0 * (outPrev - vq)
                }}
                else if layerPrev.dirty {
                for depth in 0..<nbChannels
                {
                    neuronsPrev[depth].get(i, j)!.v[elem].delta = 0.0
                }}
            }}}
            propagateDirty()
        }
    }
    
    fileprivate func _backwardWeightsCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeDeltaWeights
        {
            let neuronsPrev = layerPrev.neurons
            let indicesPtr = (indices as! MetalSharedBuffer<Int32>).buffer
            
            if !accumulateDeltaWeights
            {
                for k in 0..<K {
                for depth in 0..<nbChannels
                {
                    _wArrays.g(k, depth, 0.0)
                }}
            }
            
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                let minIndex =
                    Int(indicesPtr[j + (elem * height + i) * width])
                if minIndex >= 0 {
                for depth in 0..<nbChannels
                {
                    let vq = _wArrays.w(minIndex, depth)
                    let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                    
                    let g = _wArrays.g(minIndex, depth)
                    _wArrays.g(
                        minIndex, depth,
                        g + coeff / Double(batchSize * height * width) *
                        2.0 * (vq - outPrev)
                    )
                }}
            }}}
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
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pK: [UInt32] = [UInt32(K)]
            let pBeta: [Float] = [Float(beta)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "vq2DBackward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBuffer(_wBuffers.w.metal, atIndex: 2)
            command.setBuffer(indices.metal, atIndex: 3)
            command.setBytes(pNbChannels, atIndex: 4)
            command.setBytes(pDimensions, atIndex: 5)
            command.setBytes(pK, atIndex: 6)
            command.setBytes(pBeta, atIndex: 7)
            command.setBytes(pNbBatch, atIndex: 8)
            command.setBytes(pDirty, atIndex: 9)
            command.setBuffer(layerPrev.delta.metal, atIndex: 10)
            
            command.dispatchThreads(
                width: nbChannels * width,
                height: batchSize * height
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
    
    fileprivate func _backwardWeightsGPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeDeltaWeights
        {
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
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
                    "vq2DBatchDerWeights", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(_wBuffers.w.metal, atIndex: 1)
                command.setBuffer(indices.metal, atIndex: 2)
                command.setBytes(pNbChannels, atIndex: 3)
                command.setBytes(pDimensions, atIndex: 4)
                command.setBytes(pK, atIndex: 5)
                command.setBytes(pCoeff, atIndex: 6)
                command.setBytes(pNbBatch, atIndex: 7)
                command.setBuffer(_wBuffers.g.metal, atIndex: 8)
                
                command.dispatchThreads(width: nbChannels, height: K)
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
                    "vq2DDerWeights", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(_wBuffers.w.metal, atIndex: 1)
                command.setBuffer(indices.metal, atIndex: 2)
                command.setBytes(pNbChannels, atIndex: 3)
                command.setBytes(pDimensions, atIndex: 4)
                command.setBytes(pK, atIndex: 5)
                command.setBytes(pCoeff, atIndex: 6)
                command.setBytes(pNbBatch, atIndex: 7)
                command.setBuffer(_wDeltaWeights.metal, atIndex: 8)
                
                command.dispatchThreads(
                    width: nbChannels,
                    height: batchSize * K
                )
                command.enqueue()
                
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "vq2DReduceWeights", deviceID: deviceID
                )
                command.setBuffer(_wDeltaWeights.metal, atIndex: 0)
                command.setBytes(pNbChannels, atIndex: 1)
                command.setBytes(pK, atIndex: 2)
                command.setBytes(pNbBatch, atIndex: 3)
                command.setBytes(pAccumulate, atIndex: 4)
                command.setBuffer(_wBuffers.g.metal, atIndex: 5)
                
                command.dispatchThreads(width: nbChannels, height: K)
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
        
        if let layerPrev = self.layerPrev as? Layer2D
        {
            let neuronsPrev = layerPrev.neurons
            let indicesPtr = (indices as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                let minIndex =
                    Int(indicesPtr[j + (elem * height + i) * width])
                if minIndex >= 0
                {
                    var value: Double = 0.0
                    for depth in 0..<nbChannels
                    {
                        let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                        let vq = neurons[depth].get(i, j)!.v[elem].out
                        value += pow(outPrev - vq, 2.0)
                    }
                    losses[elem] += T(value)
                }
            }}}
        }
        return T(coeff) / T(batchSize * height * width) *
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
        
        let layerPrev = self.layerPrev as! Layer2D
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let command = MetalKernel.get.createCommand(
            "vq2DLoss", deviceID: deviceID
        )
        command.setBuffer(layerPrev.outs.metal, atIndex: 0)
        command.setBuffer(outs.metal, atIndex: 1)
        command.setBuffer(indices.metal, atIndex: 2)
        command.setBytes(pNbChannels, atIndex: 3)
        command.setBytes(pDimensions, atIndex: 4)
        command.setBytes(pNbBatch, atIndex: 5)
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
        
        return T(coeff) * T(loss) / T(batchSize * height * width)
    }
    
    /// Compute the derivative of the loss in the CPU execution context.
    public func lossDerivativeCPU() throws
    {
        if dirty
        {
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels {
            for i in 0..<height {
            for j in 0..<width
            {
                neurons[depth].get(i, j)!.v[elem].delta = 0.0
            }}}}
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

/// Layer with a 2D shape neural structure and weights.
public class VQGrad2D: VQ2D
{
    /// Scale coefficient for taking into account pixels with high magnitude of gradient norm.
    public var magnitudeCoeff: Double = 2.0
    
    /// Number of threads per thread group in the GPU execution context.
    private let _threadsPerThreadgroup = 64
    
    /// Layer computing a map of maximal activations with respect to the loss.
    let _layerCAM: LayerCAM2D
    
    ///
    /// Maximal CAM elements.
    /// Shape ~ (batch, nbThreadgroups).
    ///
    private var _camMax: MetalPrivateBuffer<Float>! = nil
    
    /// Number of thread groups in the GPU execution context.
    var nbThreadgroups: Int
    {
        get {
            let value = Double(height * width) /
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
    public override init(layerPrev: Layer2D,
                         K: Int,
                         params: GrAI.Model.Params)
    {
        var paramsHidden = GrAI.Model.Params(params: params)
        paramsHidden.hidden = true
        
        _layerCAM = try! LayerCAM2D(layerPrev: layerPrev, params: paramsHidden)
        
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
        _layerCAM = try container.decode(LayerCAM2D.self, forKey: .layerCAM)
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
        let layerPrev = mapping[idPrev] as! Layer2D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = VQGrad2D(
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
        if let layerPrev = self.layerPrev as? Layer2D
        {
            if layerPrev.dirty
            {
                throw UpdateError.Dirty
            }
            
            try _layerCAM.forwardCPU()
            let neuronsCAM = _layerCAM.neurons
            
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons
            let indicesPtr = (indices as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize
            {
                var camMax: Double = 0.0
                for i in 0..<height {
                for j in 0..<width
                {
                    let cam: Double = neuronsCAM[0].get(i, j)!.v[elem].out
                    camMax = max(cam, camMax)
                }}
                
                for i in 0..<height {
                for j in 0..<width
                {
                    let cam: Double = neuronsCAM[0].get(i, j)!.v[elem].out
                    if cam / camMax >= camMax / magnitudeCoeff
                    {
                        var minIndex = -1
                        var minValue: Double? = nil
                        
                        for k in 0..<K
                        {
                            var value: Double = 0.0
                            for depth in 0..<nbChannels
                            {
                                let outPrev =
                                    neuronsPrev[depth].get(i, j)!.v[elem].out
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
                        
                        for depth in 0..<nbChannels
                        {
                            neurons[depth].get(i, j)!.v[elem].out =
                                _wArrays.w(minIndex, depth)
                        }
                        indicesPtr[j + (elem * height + i) * width] =
                            Int32(minIndex)
                    }
                    else
                    {
                        indicesPtr[j + (elem * height + i) * width] = -1
                    }
                }}
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
        if let layerPrev = self.layerPrev as? Layer2D
        {
            if layerPrev.dirty
            {
                throw UpdateError.Dirty
            }
            
            // -----------------------------------------------------------------
            // Begin the reduction that is specific to the gradient norm max.
            // -----------------------------------------------------------------
            
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pNbThreadgroups: [UInt32] = [UInt32(nbThreadgroups)]
            
            let command = MetalKernel.get.createCommand(
                "vqLayerCAMMax2D", deviceID: deviceID
            )
            command.setBuffer(_layerCAM.outs.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pNbThreadgroups, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBuffer(_camMax.metal, atIndex: 5)
            
            let threadsPerThreadgroup = MTLSizeMake(
                _threadsPerThreadgroup, 1, 1
            )
            let threadsPerGrid = MTLSize(
                width: height * width,
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
        if let layerPrev = self.layerPrev as? Layer2D
        {
            if layerPrev.dirty
            {
                throw UpdateError.Dirty
            }
            
            try _layerCAM.forwardGPU()
            try _computeLayerCAMMaxGPU()
            
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pK: [UInt32] = [UInt32(K)]
            let pMagnitudeCoeff: [Float] = [Float(magnitudeCoeff)]
            
            let command = MetalKernel.get.createCommand(
                "vqGrad2DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(_layerCAM.outs.metal, atIndex: 1)
            command.setBuffer(_camMax.metal, atIndex: 2)
            command.setBuffer(_wBuffers.w.metal, atIndex: 3)
            command.setBytes(pNbChannels, atIndex: 4)
            command.setBytes(pDimensions, atIndex: 5)
            command.setBytes(pK, atIndex: 6)
            command.setBytes(pMagnitudeCoeff, atIndex: 7)
            command.setBytes(pNbBatch, atIndex: 8)
            command.setBuffer(outs.metal, atIndex: 9)
            command.setBuffer(indices.metal, atIndex: 10)
            
            command.dispatchThreads(
                width: height * width,
                height: batchSize
            )
            command.enqueue()
        }
    }
    
    override func _backwardCPU() {}
    
    override func _backwardGPU() throws {}
}
