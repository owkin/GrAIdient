//
// Input2D.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

import MetalKit

public enum ImageFormat
{
    case RGB, Neurone
}

class InputArrays2D: InputArrays<Layer2D>, IWeightArrays
{
    var w: [Double]
    {
        get {
            var cur = 0
            var outs = [Double](repeating: 0.0, count: nbElems)
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbFilters {
            for i in 0..<_layer.height {
            for j in 0..<_layer.width
            {
                outs[cur] = _layer.neurones[depth].get(i, j)!.v[elem].out
                cur += 1
            }}}}
            return outs
        }
        set {
            var cur = 0
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbFilters {
            for i in 0..<_layer.height {
            for j in 0..<_layer.width
            {
                _layer.neurones[depth].get(i, j)!.v[elem].out = newValue[cur]
                cur += 1
            }}}}
        }
    }
    
    var g: [Double]
    {
        get {
            var cur = 0
            var delta = [Double](repeating: 0.0, count: nbElems)
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbFilters {
            for i in 0..<_layer.height {
            for j in 0..<_layer.width
            {
                delta[cur] = _layer.neurones[depth].get(i, j)!.v[elem].delta
                cur += 1
            }}}}
            return delta
        }
        set {
            var cur = 0
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbFilters {
            for i in 0..<_layer.height {
            for j in 0..<_layer.width
            {
                _layer.neurones[depth].get(i, j)!.v[elem].delta = newValue[cur]
                cur += 1
            }}}}
        }
    }
}

class InputBuffers2D: InputBuffers<Layer2D>, IWeightBuffers
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

public class Input2D: LayerInput2D, LayerResize, LayerUpdate
{
    var _wArrays: InputArrays2D! = nil
    var _wBuffers: InputBuffers2D! = nil
    
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
    
    public convenience init(width: Int, height: Int,
                            params: MAKit.Model.Params)
    {
        self.init(nbFilters: 3,
                  width: width,
                  height: height,
                  params: params)
        computeDelta = false
    }
    
    public init(nbFilters: Int, width: Int, height: Int,
                params: MAKit.Model.Params)
    {
        super.init(layerPrev: nil,
                   nbFilters: nbFilters,
                   height: height,
                   width: width,
                   params: params)
        computeDelta = false
    }
    
    public init(layerPrev: Layer2D, params: MAKit.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbFilters: layerPrev.nbFilters,
                   height: layerPrev.height,
                   width: layerPrev.width,
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
    public required init(from decoder: Decoder) throws
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
        
        let layer: Input2D
        if idPrev > -1
        {
            layer = Input2D(
                layerPrev: mapping[idPrev] as! Layer2D,
                params: params
            )
        }
        else
        {
            layer = Input2D(
                nbFilters: nbFilters, width: width, height: height,
                params: params
            )
        }
        return layer
    }
    
    ///
    /// Resize this layer.
    ///
    /// - Parameters:
    ///     - imageWidth: New size width.
    ///     - imageHeight: New size height.
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///
    /// - Returns: A new instance of `Layer`. When `inPlace` is false, `initKernel` is
    ///  necessary in order to recreate hard resources.
    ///
    public func resize(
        imageWidth: Int,
        imageHeight: Int,
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
        
        let layer: Input2D
        if idPrev > -1
        {
            layer = Input2D(
                layerPrev: mapping[idPrev] as! Layer2D, params: params)
        }
        else
        {
            layer = Input2D(
                nbFilters: nbFilters, width: imageWidth, height: imageHeight,
                params: params)
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
            _wArrays = InputArrays2D(
                layer: self,
                nbElems: nbFilters * height * width * batchSize
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
            _wBuffers = InputBuffers2D(
                layer: self,
                nbElems: nbFilters * height * width * batchSize,
                deviceID: deviceID
            )
        }
    }
    
    /// Initialize weights in the CPU execution context.
    public func initWeightsCPU() {}
    /// Initialize weights in the GPU execution context.
    public func initWeightsGPU() {}
    
    public func setDataCPU<T: BinaryFloatingPoint>(
        _ data: [T],
        batchSize: Int,
        format: ImageFormat) throws
    {
        if batchSize * nbFilters * height * width != data.count
        {
            throw LayerError.DataSize
        }
        try checkStateCPU(batchSize: batchSize)
        
        switch format
        {
        case .RGB:
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let offset = j + (elem * height + i) * width
                    for depth in 0..<nbFilters
                    {
                        neurones[depth].get(i, j)!.v[elem].out =
                            Double(data[nbFilters * offset + depth])
                    }
                }}
            }
        case .Neurone:
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    for depth in 0..<nbFilters
                    {
                        let offsetStart = (depth + nbFilters * elem) * height
                        let offset = j + (offsetStart + i) * width
                        
                        neurones[depth].get(i, j)!.v[elem].out =
                            Double(data[offset])
                    }
                }}
            }
        }
    }
    
    public func setDataGPU<T: BinaryFloatingPoint>(
        _ data: [T],
        batchSize: Int,
        format: ImageFormat) throws
    {
        if batchSize * nbFilters * height * width != data.count
        {
            throw LayerError.DataSize
        }
        try checkStateForwardGPU(batchSize: batchSize)
        
        // Wait for previous loop to end to avoid race condition with
        // didModifyRange in the following example:
        // Convolution.backwardWeightsGPU accesses layerPrev.outs.
        MetalKernel.get.download([outs])
        
        let outsPtr = outs.shared.buffer
        switch format
        {
        case .RGB:
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let offsetGet = j + (elem * height + i) * width
                    for depth in 0..<nbFilters
                    {
                        let offsetStartSet = (depth + nbFilters * elem) * height
                        let offsetSet = j + (offsetStartSet + i) * width
                        
                        outsPtr[offsetSet] =
                            Float(data[nbFilters * offsetGet + depth])
                    }
                }}
            }
        case .Neurone:
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    for depth in 0..<nbFilters
                    {
                        let offsetStart = (depth + nbFilters * elem) * height
                        let offset = j + (offsetStart + i) * width
                        
                        outsPtr[offset] = Float(data[offset])
                    }
                }}
            }
        }
        MetalKernel.get.upload([outs])
    }
    
    public func setDataGPU(
        _ data: MetalPrivateBuffer<Float>,
        batchSize: Int) throws
    {
        if batchSize * nbFilters * height * width != data.nbElems
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
        if let layerPrev = self.layerPrev as? Layer2D, computeForward
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize {
            for depth in 0..<nbFilters
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurones[depth].get(i, j)!.v[elem].out =
                        neuronesPrev[depth].get(i, j)!.v[elem].out
                }}
            }}
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeForward
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
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize {
            for depth in 0..<nbFilters
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let delta = neurones[depth].get(i, j)!.v[elem].delta
                    if layerPrev.dirty
                    {
                        neuronesPrev[depth].get(i, j)!
                            .v[elem].delta = delta
                    }
                    else
                    {
                        neuronesPrev[depth].get(i, j)!
                            .v[elem].delta += delta
                    }
                }}
            }}
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
        return [_wArrays]
    }

    /// Get the weights in the GPU execution context.
    public func collectWeightsGPU() -> [IWeightBuffers]
    {
        return [_wBuffers]
    }
}
