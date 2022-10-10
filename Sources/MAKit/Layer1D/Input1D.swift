//
// Input1D.swift
// MAKit
//
// Created by Jean-François Reboud on 10/10/2022.
//

import MetalKit

class InputArrays1D: InputArrays<Layer1D>, IWeightArrays
{
    var w: [Double]
    {
        get {
            var cur = 0
            var outs = [Double](repeating: 0.0, count: nbElems)
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbNeurones
            {
                outs[cur] = _layer.neurones.get(depth)!.v[elem].out
                cur += 1
            }}
            return outs
        }
        set {
            var cur = 0
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbNeurones
            {
                _layer.neurones.get(depth)!.v[elem].out = newValue[cur]
                cur += 1
            }}
        }
    }
    
    var g: [Double]
    {
        get {
            var cur = 0
            var delta = [Double](repeating: 0.0, count: nbElems)
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbNeurones
            {
                delta[cur] = _layer.neurones.get(depth)!.v[elem].delta
                cur += 1
            }}
            return delta
        }
        set {
            var cur = 0
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbNeurones
            {
                _layer.neurones.get(depth)!.v[elem].delta = newValue[cur]
                cur += 1
            }}
        }
    }
}

class InputBuffers1D: InputBuffers<Layer1D>, IWeightBuffers
{
    var w: MetalBuffer<Float>
    {
        get {
            return _layer.outs
        }
    }
    
    var g: MetalBuffer<Float>
    {
        get {
            return _layer.delta
        }
    }
}

public class Input1D: LayerInput1D, LayerUpdate
{
    var _wArrays: InputArrays1D! = nil
    var _wBuffers: InputBuffers1D! = nil
    
    /// Whether to compute weights' gradients or not.
    public var computeDeltaWeights: Bool = false
    
    /// Whether gradients of weights must be accumulated or not.
    public var accumulateDeltaWeights: Bool = false
    
    /// Weights in the CPU execution context.
    public var weightsCPU: [Float]
    {
        get { [] }
        set {}
    }
    /// Weights in the GPU execution context.
    public var weightsGPU: [Float]
    {
        get { [] }
        set {}
    }
    
    public init(nbNeurones: Int, params: MAKit.Model.Params)
    {
        super.init(layerPrev: nil,
                   nbNeurones: nbNeurones,
                   params: params)
        computeDelta = false
    }
    
    public init(layerPrev: Layer1D, params: MAKit.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbNeurones: layerPrev.nbNeurones,
                   params: params)
        computeDelta = false
    }
    
    ///
    /// Create an instance of Layer by decoding from the given decoder.
    ///
    /// This initializer throws an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    required init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
        computeDelta = false
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
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
        
        let layer: Input1D
        if idPrev > -1
        {
            layer = Input1D(
                layerPrev: mapping[idPrev] as! Layer1D,
                params: params
            )
        }
        else
        {
            layer = Input1D(nbNeurones: nbNeurones, params: params)
        }
        return layer
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean weights.
    ///
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        _wArrays = nil
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean weights.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _wBuffers = nil
    }
    
    ///
    /// Initialize state resources in the CPU execution context.
    ///
    /// We initialize the neurones' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        try super.checkStateCPU(batchSize: batchSize)
        
        if _wArrays == nil && computeDeltaWeights
        {
            _wArrays = InputArrays1D(
                layer: self,
                nbElems: nbNeurones * batchSize
            )
        }
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurones' forward state.
    /// We initialize the weights.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if _wBuffers == nil && computeDeltaWeights
        {
            _wBuffers = InputBuffers1D(
                layer: self,
                nbElems: nbNeurones * batchSize,
                deviceID: deviceID
            )
        }
    }
    
    /// Initialize weights in the CPU execution context.
    public func initWeightsCPU() {}
    /// Initialize weights in the GPU execution context.
    public func initWeightsGPU() {}
    
    public func setDataCPU<T: BinaryFloatingPoint>(_ data: [[T]]) throws
    {
        let batchSize = data.count
        try checkStateCPU(batchSize: batchSize)
        
        for (elem, sample) in data.enumerated()
        {
            if sample.count != nbNeurones
            {
                throw LayerError.DataSize
            }
            
            for (i, feature) in sample.enumerated() {
            if let neurone = neurones.get(i)
            {
                neurone.v[elem].out = Double(feature)
            }}
        }
    }
    
    public func setDataGPU<T: BinaryFloatingPoint>(_ data: [[T]]) throws
    {
        let batchSize = data.count
        try checkStateForwardGPU(batchSize: batchSize)
        
        // Wait for previous loop to end to avoid race condition with
        // didModifyRange in the following example:
        // FullyConnected.backwardWeightsGPU accesses layerPrev.outs.
        MetalKernel.get.download([outs])
        
        if batchSize * nbNeurones != data.count * data.first!.count
        {
            throw LayerError.DataSize
        }
        
        let outsPtr = outs.shared.buffer
        for elem in 0..<batchSize
        {
            for depth in 0..<nbNeurones
            {
                let offset = depth + nbNeurones * elem
                outsPtr[offset] = Float(data[elem][depth])
            }
        }
        MetalKernel.get.upload([outs])
    }
    
    public func setDataGPU(
        _ data: MetalPrivateBuffer<Float>,
        batchSize: Int) throws
    {
        if batchSize * nbNeurones != data.nbElems
        {
            throw LayerError.DataSize
        }
        try checkStateForwardGPU(batchSize: batchSize)
        
        outs = data
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D, computeForward
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurones
                {
                    neurones.get(depth)?.v[elem].out =
                        neuronesPrev.get(depth)!.v[elem].out
                }
            }
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D, computeForward
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
            let threadsPerGrid = MTLSize(width: nbElems, height: 1, depth: 1)
            command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                               threadsPerThreadgroup: threadsPerThreadgroup)
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurones
                {
                    let delta = neurones.get(depth)!.v[elem].delta
                    if layerPrev.dirty
                    {
                        neuronesPrev.get(depth)!.v[elem].delta = delta
                    }
                    else
                    {
                        neuronesPrev.get(depth)!.v[elem].delta += delta
                    }
                }
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
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
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
        return [_wArrays]
    }

    /// Get the weights in the GPU execution context.
    public func collectWeightsGPU() -> [IWeightBuffers]
    {
        return [_wBuffers]
    }
}
