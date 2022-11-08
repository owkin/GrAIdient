//
// MaxPool2D.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

///
/// Layer with a 2D shape neural structure.
///
/// This layer downscales the previous layer, outputing the maximal elements contained in a moving kernel.
///
public class MaxPool2D: Layer2D
{
    ///
    /// Indices of maximal elements.
    /// Shape ~ (batch, nbChannels, height, width).
    ///
    var _indicesMax: MetalBuffer<Int32>! = nil
    
    /// Size of the maximal kernel.
    let _size: Int
    /// Downscale factor of the resolution (height and width).
    let _stride: Int
    
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
                    (_size - 1) * Int(super.strideFactor)
                receptiveFieldCache = value
                return value
            }
        }
    }
    
    var _kernelIndices: (Int, Int)
    {
        get {
            let sizeHalf = _size / 2
            let start = _size % 2 == 1 ? -sizeHalf : -sizeHalf+1
            let end = sizeHalf
            return (start, end)
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case size
        case stride
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - size: The maximal kernel size.
    ///     - stride: Downscale factor of the resolution (height and width).
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                size: Int,
                stride: Int,
                params: MAKit.Model.Params)
    {
        _size = size
        _stride = stride
        
        let width = layerPrev.width
        let height = layerPrev.height
        let widthRes = width % stride
        let heightRes = height % stride
        let widthNew = widthRes == 0 ? width / stride : width / stride + 1
        let heightNew = heightRes == 0 ? height / stride : height / stride + 1
        
        super.init(layerPrev: layerPrev,
                   nbChannels: layerPrev.nbChannels,
                   height: heightNew,
                   width: widthNew,
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
        _size = try values.decode(Int.self, forKey: Keys.size)
        _stride = try values.decode(Int.self, forKey: Keys.stride)
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
        try container.encode(_size, forKey: Keys.size)
        try container.encode(_stride, forKey: Keys.stride)
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
            
        let layer = MaxPool2D(
            layerPrev: layerPrev,
            size: _size,
            stride: _stride,
            params: params
        )
        return layer
    }
    
    ///
    /// Clean state resources in the CPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        _indicesMax = nil
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _indicesMax = nil
    }
    
    ///
    /// Initialize state resources in the CPU execution context.
    ///
    /// We initialize the neurons' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        try super.checkStateCPU(batchSize: batchSize)
        
        if _indicesMax == nil
        {
            _indicesMax = MetalSharedBuffer<Int32>(
                batchSize * nbChannels * height * width * 2,
                deviceID: deviceID
            )
        }
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if _indicesMax == nil
        {
            _indicesMax = MetalPrivateBuffer<Int32>(
                batchSize * nbChannels * width * height,
                deviceID: deviceID
            )
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.initGC(batchSize: batchSize,
                                                      nbGC: nbGC)
                }}
            }
            
            let neuronsPrev = layerPrev.neurons
            let (start, end) = _kernelIndices
            
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var maxOutPrev: Double = -10000.0
                    for k in start...end {
                    for l in start...end
                    {
                        if let outPrev = neuronsPrev[depth]
                           .get(_stride*i+k, _stride*j+l)?.gc[batch][elem].out,
                           outPrev > maxOutPrev
                        {
                            maxOutPrev = outPrev
                        }
                    }}
                    neurons[depth].get(i, j)!.gc[batch][elem].out = maxOutPrev
                }}
            }}}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try forwardGCCPU()
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
            let (start, end) = _kernelIndices
            let indicesMaxPtr =
                (_indicesMax as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                let offsetStart = (depth + nbChannels * elem) * height
                
                for i in 0..<height {
                for j in 0..<width
                {
                    var maxI = -1
                    var maxJ = -1
                    var maxOutPrev: Double = -10000.0
                    for k in start...end {
                    for l in start...end
                    {
                        if let outPrev = neuronsPrev[depth]
                            .get(_stride*i+k, _stride*j+l)?.v[elem].out,
                           outPrev > maxOutPrev
                        {
                            maxOutPrev = outPrev
                            maxI = _stride*i+k
                            maxJ = _stride*j+l
                        }
                    }}
                    
                    if maxI >= 0 && maxJ >= 0
                    {
                        let offset = j + (offsetStart + i) * width
                        
                        neurons[depth].get(i, j)!.v[elem].out = maxOutPrev
                        indicesMaxPtr[2 * offset] = Int32(maxI)
                        indicesMaxPtr[2 * offset + 1] = Int32(maxJ)
                    }
                    else
                    {
                        fatalError("Unreachable.")
                    }
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
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let widthPrev = layerPrev.width
            let heightPrev = layerPrev.height
            let (start, end) = _kernelIndices
            
            let pStart: [Int32] = [Int32(start), Int32(end)]
            let pStride: [UInt32] = [UInt32(_stride)]
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            
            let command = MetalKernel.get.createCommand(
                "maxPoolForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pStart, atIndex: 1)
            command.setBytes(pStride, atIndex: 2)
            command.setBytes(pNbChannels, atIndex: 3)
            command.setBytes(pDimensions, atIndex: 4)
            command.setBytes(pDimensionsPrev, atIndex: 5)
            command.setBytes(pNbBatch, atIndex: 6)
            command.setBuffer(outs.metal, atIndex: 7)
            command.setBuffer(_indicesMax.metal, atIndex: 8)
            
            command.dispatchThreads(
                width: width * nbChannels,
                height: height * batchSize
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
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            let indicesMaxPtr =
                (_indicesMax as! MetalSharedBuffer<Int32>).buffer
            
            if layerPrev.dirty
            {
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    for i in 0..<heightPrev {
                    for j in 0..<widthPrev
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta = 0.0
                    }}
                }}
            }
            
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                let offsetStart = (depth + nbChannels * elem) * height
                
                for i in 0..<height {
                for j in 0..<width
                {
                    let offset = j + (offsetStart + i) * width
                    
                    let maxI = Int(indicesMaxPtr[2 * offset])
                    let maxJ = Int(indicesMaxPtr[2 * offset + 1])
                    
                    let deltaCur = neurons[depth].get(i, j)!.v[elem].delta
                    neuronsPrev[depth].get(maxI, maxJ)!.v[elem].delta +=
                        deltaCur
                }}
            }}
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
            
            let widthPrev = layerPrev.width
            let heightPrev = layerPrev.height
            let (start, end) = _kernelIndices
            
            let pStart: [Int32] = [Int32(start), Int32(end)]
            let pStride: [UInt32] = [UInt32(_stride)]
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "maxPoolBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(_indicesMax.metal, atIndex: 1)
            command.setBytes(pStart, atIndex: 2)
            command.setBytes(pStride, atIndex: 3)
            command.setBytes(pNbChannels, atIndex: 4)
            command.setBytes(pDimensions, atIndex: 5)
            command.setBytes(pDimensionsPrev, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pDirty, atIndex: 8)
            command.setBuffer(layerPrev.delta.metal, atIndex: 9)
            
            command.dispatchThreads(
                width: widthPrev * nbChannels,
                height: heightPrev * batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
