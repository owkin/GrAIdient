//
// Input2D.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

import MetalKit

///
/// Format of images.
///
/// RGB: Image structure is (batch, channel, height, width).
/// Neuron: Image structure is (batch, height, width, channel).
///
public enum ImageFormat
{
    case RGB, Neuron
}

/// Arrays needed to update the inputs of a layer.
class InputArrays2D: InputArrays<Layer2D>, IWeightArrays
{
    /// Inputs array: the array to update.
    var w: [Double]
    {
        get {
            var cur = 0
            var outs = [Double](repeating: 0.0, count: nbElems)
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbChannels {
            for i in 0..<_layer.height {
            for j in 0..<_layer.width
            {
                outs[cur] = _layer.neurons[depth].get(i, j)!.v[elem].out
                cur += 1
            }}}}
            return outs
        }
        set {
            var cur = 0
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbChannels {
            for i in 0..<_layer.height {
            for j in 0..<_layer.width
            {
                _layer.neurons[depth].get(i, j)!.v[elem].out = newValue[cur]
                cur += 1
            }}}}
        }
    }
    
    /// Gradients array.
    var g: [Double]
    {
        get {
            var cur = 0
            var delta = [Double](repeating: 0.0, count: nbElems)
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbChannels {
            for i in 0..<_layer.height {
            for j in 0..<_layer.width
            {
                delta[cur] = _layer.neurons[depth].get(i, j)!.v[elem].delta
                cur += 1
            }}}}
            return delta
        }
        set {
            var cur = 0
            for elem in 0..<_layer.batchSize {
            for depth in 0..<_layer.nbChannels {
            for i in 0..<_layer.height {
            for j in 0..<_layer.width
            {
                _layer.neurons[depth].get(i, j)!.v[elem].delta = newValue[cur]
                cur += 1
            }}}}
        }
    }
}

/// GPU buffers needed to update the inputs of a layer.
class InputBuffers2D: InputBuffers<Layer2D>, IWeightBuffers
{
    /// Inputs buffer: the buffer to be update.
    var w: MetalBuffer<Float>
    {
        get {
            return _layer.outs
        }
    }
    
    /// Gradients buffer.
    var g: MetalBuffer<Float>
    {
        get {
            return _layer.delta
        }
    }
}

/// First layer with a 2D shape neural structure.
public class Input2D: LayerInput2D, LayerResize, LayerUpdate
{
    /// Grid of "weights".
    var _wArrays: InputArrays2D! = nil
    /// Buffer of "weights".
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
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(nbChannels: Int, width: Int, height: Int,
                params: MAKit.Model.Params)
    {
        super.init(layerPrev: nil,
                   nbChannels: nbChannels,
                   height: height,
                   width: width,
                   params: params)
        computeDelta = false
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D, params: MAKit.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbChannels: layerPrev.nbChannels,
                   height: layerPrev.height,
                   width: layerPrev.width,
                   params: params)
        computeDelta = false
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
        computeDelta = false
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
                nbChannels: nbChannels, width: width, height: height,
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
                layerPrev: mapping[idPrev] as! Layer2D, params: params
            )
        }
        else
        {
            layer = Input2D(
                nbChannels: nbChannels, width: imageWidth, height: imageHeight,
                params: params
            )
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
    /// We initialize the neurons' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        try super.checkStateCPU(batchSize: batchSize)
        
        if _wArrays == nil && computeDeltaWeights
        {
            _wArrays = InputArrays2D(
                layer: self,
                nbElems: nbChannels * height * width * batchSize
            )
        }
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    /// We initialize the weights.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if _wBuffers == nil && computeDeltaWeights
        {
            _wBuffers = InputBuffers2D(
                layer: self,
                nbElems: nbChannels * height * width * batchSize,
                deviceID: deviceID
            )
        }
    }
    
    /// Initialize weights in the CPU execution context.
    public func initWeightsCPU() {}
    /// Initialize weights in the GPU execution context.
    public func initWeightsGPU() {}
    
    ///
    /// API to set data in the CPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The data to set.
    ///     - format: The data format.
    ///
    public func setDataCPU<T: BinaryFloatingPoint>(
        _ data: [[T]],
        format: ImageFormat) throws
    {
        let batchSize = data.count
        let dataTmp = data.reduce([], +)
        try setDataCPU(dataTmp, batchSize: batchSize, format: format)
    }
    
    ///
    /// API to set data in the CPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The data to set.
    ///     - batchSize: The batch size of data.
    ///     - format: The data format.
    ///
    public func setDataCPU<T: BinaryFloatingPoint>(
        _ data: [T],
        batchSize: Int,
        format: ImageFormat) throws
    {
        if batchSize * nbChannels * height * width != data.count
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
                    for depth in 0..<nbChannels
                    {
                        neurons[depth].get(i, j)!.v[elem].out =
                            Double(data[nbChannels * offset + depth])
                    }
                }}
            }
        case .Neuron:
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    for depth in 0..<nbChannels
                    {
                        let offsetStart = (depth + nbChannels * elem) * height
                        let offset = j + (offsetStart + i) * width
                        
                        neurons[depth].get(i, j)!.v[elem].out =
                            Double(data[offset])
                    }
                }}
            }
        }
    }
    
    ///
    /// API to set data in the GPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The data to set.
    ///     - format: The data format.
    ///
    public func setDataGPU<T: BinaryFloatingPoint>(
        _ data: [[T]],
        format: ImageFormat) throws
    {
        let batchSize = data.count
        let dataTmp = data.reduce([], +)
        try setDataGPU(dataTmp, batchSize: batchSize, format: format)
    }
    
    ///
    /// API to set data in the GPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The data to set.
    ///     - batchSize: The batch size of data.
    ///     - format: The data format.
    ///
    public func setDataGPU<T: BinaryFloatingPoint>(
        _ data: [T],
        batchSize: Int,
        format: ImageFormat) throws
    {
        if batchSize * nbChannels * height * width != data.count
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
                    for depth in 0..<nbChannels
                    {
                        let offsetStartSet =
                            (depth + nbChannels * elem) * height
                        let offsetSet = j + (offsetStartSet + i) * width
                        
                        outsPtr[offsetSet] =
                            Float(data[nbChannels * offsetGet + depth])
                    }
                }}
            }
        case .Neuron:
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    for depth in 0..<nbChannels
                    {
                        let offsetStart = (depth + nbChannels * elem) * height
                        let offset = j + (offsetStart + i) * width
                        
                        outsPtr[offset] = Float(data[offset])
                    }
                }}
            }
        }
        MetalKernel.get.upload([outs])
    }
    
    ///
    /// API to set data in the GPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The data to set.
    ///     - batchSize: The batch size of data.
    ///
    public func setDataGPU(
        _ data: MetalPrivateBuffer<Float>,
        batchSize: Int) throws
    {
        if batchSize * nbChannels * height * width != data.nbElems
        {
            throw LayerError.DataSize
        }
        try checkStateForwardGPU(batchSize: batchSize)
        
        outs = data
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D, computeForward
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.v[elem].out =
                        neuronsPrev[depth].get(i, j)!.v[elem].out
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
        if let layerPrev = self.layerPrev as? Layer2D, computeForward
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let nbElems = outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command = MetalKernel.get.createCommand(
                "sum1", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            let threads = command.threadExecutionWidth
            let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
            let threadsPerGrid = MTLSize(width: nbElems, height: 1, depth: 1)
            command.dispatchThreads(
                threadsPerGrid: threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let delta = neurons[depth].get(i, j)!.v[elem].delta
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!
                            .v[elem].delta = delta
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!
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
    /// Throw an error if batch size is greater than the first batch size.
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
                    "sum1", deviceID: deviceID
                )
            }
            else
            {
                command = MetalKernel.get.createCommand(
                    "sum2", deviceID: deviceID
                )
            }
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(layerPrev.delta.metal, atIndex: 2)
            
            let threads = command.threadExecutionWidth
            let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
            let threadsPerGrid = MTLSize(width: nbElems,
                                         height: 1,
                                         depth: 1)
            command.dispatchThreads(
                threadsPerGrid: threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
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
