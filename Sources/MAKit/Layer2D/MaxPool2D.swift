//
// MaxPool2D.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

import MetalKit

public class MaxPool2D: Layer2D
{
    var _indicesMax: MetalBuffer<Int32>! = nil
    
    let _size: Int
    let _stride: Int
    
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
                   nbFilters: layerPrev.nbFilters,
                   height: heightNew,
                   width: widthNew,
                   params: params)
    }
    
    public required init(from decoder: Decoder) throws
    {
        let values = try decoder.container(keyedBy: Keys.self)
        _size = try values.decode(Int.self, forKey: Keys.size)
        _stride = try values.decode(Int.self, forKey: Keys.stride)
        try super.init(from: decoder)
    }
    
    public override func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        try container.encode(_size, forKey: Keys.size)
        try container.encode(_stride, forKey: Keys.stride)
        try super.encode(to: encoder)
    }
    
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
    
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        _indicesMax = nil
    }
    
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _indicesMax = nil
    }
    
    public override func checkStateCPU(batchSize: Int) throws
    {
        try super.checkStateCPU(batchSize: batchSize)
        
        if _indicesMax == nil
        {
            _indicesMax = MetalSharedBuffer<Int32>(
                batchSize * nbFilters * height * width * 2,
                deviceID: deviceID
            )
        }
    }
    
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if _indicesMax == nil
        {
            _indicesMax = MetalPrivateBuffer<Int32>(
                batchSize * nbFilters * width * height,
                deviceID: deviceID
            )
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for depth in 0..<nbFilters
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurones[depth].get(i, j)!.initGC(batchSize: batchSize,
                                                      nbGC: nbGC)
                }}
            }
            
            let neuronesPrev = layerPrev.neurones
            let (start, end) = _kernelIndices
            
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for depth in 0..<nbFilters
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var maxOutPrev: Double = -10000.0
                    for k in start...end {
                    for l in start...end
                    {
                        if let outPrev = neuronesPrev[depth]
                           .get(_stride*i+k, _stride*j+l)?.gc[batch][elem].out,
                           outPrev > maxOutPrev
                        {
                            maxOutPrev = outPrev
                        }
                    }}
                    neurones[depth].get(i, j)!.gc[batch][elem].out = maxOutPrev
                }}
            }}}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try forwardGCCPU()
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
            let (start, end) = _kernelIndices
            let indicesMaxPtr =
                (_indicesMax as! MetalSharedBuffer<Int32>).buffer
            
            for elem in 0..<batchSize {
            for depth in 0..<nbFilters
            {
                let offsetStart = (depth + nbFilters * elem) * height
                
                for i in 0..<height {
                for j in 0..<width
                {
                    var maxI = -1
                    var maxJ = -1
                    var maxOutPrev: Double = -10000.0
                    for k in start...end {
                    for l in start...end
                    {
                        if let outPrev = neuronesPrev[depth]
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
                        
                        neurones[depth].get(i, j)!.v[elem].out = maxOutPrev
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
    /// Throws an error if batch size is greater than the first batch size.
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
            let pNbFilters: [UInt32] = [UInt32(nbFilters)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            
            let command = MetalKernel.get.createCommand(
                "maxPoolForward", deviceID: deviceID)
            
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pStart, atIndex: 1)
            command.setBytes(pStride, atIndex: 2)
            command.setBytes(pNbFilters, atIndex: 3)
            command.setBytes(pDimensions, atIndex: 4)
            command.setBytes(pDimensionsPrev, atIndex: 5)
            command.setBytes(pNbBatch, atIndex: 6)
            command.setBuffer(outs.metal, atIndex: 7)
            command.setBuffer(_indicesMax.metal, atIndex: 8)
            
            let threadsPerThreadgroup = MTLSizeMake(8, 8, 8)
            let threadsPerGrid = MTLSize(width: width,
                                         height: height,
                                         depth: nbFilters * batchSize)
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
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            let indicesMaxPtr =
                (_indicesMax as! MetalSharedBuffer<Int32>).buffer
            
            if layerPrev.dirty
            {
                for elem in 0..<batchSize {
                for depth in 0..<nbFilters
                {
                    for i in 0..<heightPrev {
                    for j in 0..<widthPrev
                    {
                        neuronesPrev[depth].get(i, j)!.v[elem].delta = 0.0
                    }}
                }}
            }
            
            for elem in 0..<batchSize {
            for depth in 0..<nbFilters
            {
                let offsetStart = (depth + nbFilters * elem) * height
                
                for i in 0..<height {
                for j in 0..<width
                {
                    let offset = j + (offsetStart + i) * width
                    
                    let maxI = Int(indicesMaxPtr[2 * offset])
                    let maxJ = Int(indicesMaxPtr[2 * offset + 1])
                    
                    let deltaCur = neurones[depth].get(i, j)!.v[elem].delta
                    neuronesPrev[depth].get(maxI, maxJ)!.v[elem].delta +=
                        deltaCur
                }}
            }}
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
            
            let widthPrev = layerPrev.width
            let heightPrev = layerPrev.height
            let (start, end) = _kernelIndices
            
            let pStart: [Int32] = [Int32(start), Int32(end)]
            let pStride: [UInt32] = [UInt32(_stride)]
            let pNbFilters: [UInt32] = [UInt32(nbFilters)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "maxPoolBackward", deviceID: deviceID)
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(_indicesMax.metal, atIndex: 1)
            command.setBytes(pStart, atIndex: 2)
            command.setBytes(pStride, atIndex: 3)
            command.setBytes(pNbFilters, atIndex: 4)
            command.setBytes(pDimensions, atIndex: 5)
            command.setBytes(pDimensionsPrev, atIndex: 6)
            command.setBytes(pNbBatch, atIndex: 7)
            command.setBytes(pDirty, atIndex: 8)
            command.setBuffer(layerPrev.delta.metal, atIndex: 9)
            
            let threadsPerThreadgroup = MTLSizeMake(8, 8, 8)
            let threadsPerGrid = MTLSize(width: widthPrev,
                                         height: heightPrev,
                                         depth: nbFilters * batchSize)
            command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                               threadsPerThreadgroup: threadsPerThreadgroup)
            command.enqueue()
            
            propagateDirty()
        }
    }
}
