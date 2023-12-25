//
// AdaptiveAvgPool2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

import Foundation

///
/// Layer with a 2D shape neural structure.
///
/// This layer enables to fix a desired output grid size without having to specify any downscale factor.
/// It has an important role when `FullyConnected` layers follow: it allows to enforce the number
/// of connections whatever the size of the previous `Layer2D` layer is.
///
/// Example:
/// convolution -> adaptive_avg_pool -> fully_connected
///
/// In this example, the convolution size may be anything but the adaptive_avg_pool will output
/// a specific size which guaranties the fully_connected is safely connected to every  elements.
///
public class AdaptiveAvgPool2D: Layer2D
{
    ///
    /// Buffer containing the number of elements to average in each direction for each element in
    /// the output grid.
    /// Shape ~ (batch, nbChannels, height, width).
    ///
    var _nbElems: MetalBuffer<Int32>! = nil
    
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
                let heightPrev = (layerPrev as! Layer2D).height
                let size: Int
                
                if heightPrev >= height
                {
                    size = heightPrev / height
                }
                else
                {
                    fatalError("'strideFactor' has no meaning in this config.")
                }
                
                let value = super.strideFactor * Double(size)
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
                let layerPrev = self.layerPrev as! Layer2D
                let heightPrev = layerPrev.height
                
                let smallSize: Int
                let bigSize: Int
                if heightPrev >= height
                {
                    smallSize = height
                    bigSize = heightPrev
                }
                else
                {
                    fatalError(
                        "'receptiveField' has no meaning in this config."
                    )
                }
                
                let start = _startIndex(
                    index: 0,
                    smallSize: smallSize,
                    bigSize: bigSize
                )
                let end = _endIndex(
                    index: 0,
                    smallSize: smallSize,
                    bigSize: bigSize
                )
                let size = end - start
                
                let value = super.receptiveField +
                    (size - 1) * Int(super.strideFactor)
                receptiveFieldCache = value
                return value
            }
        }
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - size: The output fixed grid resolution (height, width).
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D, size: Int, params: GrAI.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbChannels: layerPrev.nbChannels,
                   height: size,
                   width: size,
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
    /// Clean state resources in the CPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        _nbElems = nil
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _nbElems = nil
    }
    
    ///
    /// Initialize state resources in the CPU execution context.
    ///
    /// We initialize the neurons' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        try super.checkStateCPU(batchSize: batchSize)
        
        if let layerPrev = self.layerPrev as? Layer2D
        {
            let heightPrev = layerPrev.height
            if heightPrev < height && _nbElems == nil
            {
                _nbElems = MetalSharedBuffer<Int32>(
                    batchSize * nbChannels * height * width,
                    deviceID: deviceID
                )
            }
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
        
        if let layerPrev = self.layerPrev as? Layer2D
        {
            let heightPrev = layerPrev.height
            if heightPrev < height && _nbElems == nil
            {
                _nbElems = MetalPrivateBuffer<Int32>(
                    batchSize * nbChannels * height * width,
                    deviceID: deviceID
                )
            }
        }
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
            
        let newLayer = AdaptiveAvgPool2D(
            layerPrev: layerPrev,
            size: height,
            params: params
        )
        if inPlace
        {
            newLayer.neurons = neurons
            newLayer._nbElems = _nbElems
        }
        return newLayer
    }
    
    private func _startIndex(index: Int, smallSize: Int, bigSize: Int) -> Int
    {
        return Int(floor(Double(index * bigSize) / Double(smallSize)))
    }
    
    private func _endIndex(index: Int, smallSize: Int, bigSize: Int) -> Int
    {
        return Int(ceil(Double((index + 1) * bigSize) / Double(smallSize)))
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
                    neurons[depth].get(i, j)!.initGC(
                        batchSize: batchSize,
                        nbGC: nbGC
                    )
                }}
            }
            
            let neuronsPrev = layerPrev.neurons
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            if heightPrev >= height
            {
                for batch in 0..<batchSize {
                for elem in 0..<nbGC {
                for depth in 0..<nbChannels
                {
                    for i in 0..<height
                    {
                    let startI = _startIndex(index: i,
                                             smallSize: height,
                                             bigSize: heightPrev)
                    let endI = _endIndex(index: i,
                                         smallSize: height,
                                         bigSize: heightPrev)
                    let nbElemsI = endI - startI
                        
                    for j in 0..<width
                    {
                        let startJ = _startIndex(index: j,
                                                 smallSize: width,
                                                 bigSize: widthPrev)
                        let endJ = _endIndex(index: j,
                                             smallSize: width,
                                             bigSize: widthPrev)
                        let nbElemsJ = endJ - startJ
                        
                        let nbElems = nbElemsI * nbElemsJ
                        
                        var sum = 0.0
                        for k in 0..<nbElemsI {
                        for l in 0..<nbElemsJ
                        {
                            sum += neuronsPrev[depth].get(
                                startI + k,
                                startJ + l)!.gc[batch][elem].out
                        }}
                        
                        neurons[depth].get(i, j)!.gc[batch][elem].out =
                            sum / Double(nbElems)
                    }}
                }}}
            }
            else
            {
                for batch in 0..<batchSize {
                for elem in 0..<nbGC {
                for depth in 0..<nbChannels
                {
                    for I in 0..<height {
                    for J in 0..<width
                    {
                        neurons[depth].get(I, J)!.gc[batch][elem].out = 0.0
                    }}
                }}}
                
                let nbElemsPtr: UnsafeMutableBufferPointer<Int32>
                if let sBuffer = _nbElems as? MetalSharedBuffer<Int32>
                {
                    nbElemsPtr = sBuffer.buffer
                }
                else if let pBuffer = _nbElems as? MetalPrivateBuffer
                {
                    MetalKernel.get.download([pBuffer])
                    nbElemsPtr = pBuffer.shared.buffer
                }
                else
                {
                    fatalError("Unreachable.")
                }
                
                for batch in 0..<batchSize {
                for elem in 0..<nbGC {
                for depth in 0..<nbChannels
                {
                    let offsetStart =
                        (depth + nbChannels * batch) * height
                    
                    for i in 0..<heightPrev
                    {
                    let startI = _startIndex(index: i,
                                             smallSize: heightPrev,
                                             bigSize: height)
                    let endI = _endIndex(index: i,
                                         smallSize: heightPrev,
                                         bigSize: height)
                    let nbElemsI = endI - startI
                    
                    for j in 0..<widthPrev
                    {
                        let startJ = _startIndex(index: j,
                                                 smallSize: widthPrev,
                                                 bigSize: width)
                        let endJ = _endIndex(index: j,
                                             smallSize: widthPrev,
                                             bigSize: width)
                        let nbElemsJ = endJ - startJ
                        
                        let outPrev = neuronsPrev[depth].get(i, j)!
                            .gc[batch][elem].out
                        
                        for k in 0..<nbElemsI {
                        for l in 0..<nbElemsJ
                        {
                            let offset = startJ + l +
                                (offsetStart + startI + k) * width
                            
                            neurons[depth].get(
                                startI + k, startJ + l)!.gc[batch][elem].out +=
                                outPrev / Double(nbElemsPtr[offset])
                        }}
                    }}
                }}}
            }
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
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            if heightPrev >= height
            {
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    for i in 0..<height
                    {
                    let startI = _startIndex(index: i,
                                             smallSize: height,
                                             bigSize: heightPrev)
                    let endI = _endIndex(index: i,
                                         smallSize: height,
                                         bigSize: heightPrev)
                    let nbElemsI = endI - startI
                        
                    for j in 0..<width
                    {
                        let startJ = _startIndex(index: j,
                                                 smallSize: width,
                                                 bigSize: widthPrev)
                        let endJ = _endIndex(index: j,
                                             smallSize: width,
                                             bigSize: widthPrev)
                        let nbElemsJ = endJ - startJ
                        
                        let nbElems = nbElemsI * nbElemsJ
                        
                        var sum = 0.0
                        for k in 0..<nbElemsI {
                        for l in 0..<nbElemsJ
                        {
                            sum += neuronsPrev[depth].get(
                                startI + k,
                                startJ + l)!.v[elem].out
                        }}
                        
                        neurons[depth].get(i, j)!.v[elem].out =
                            sum / Double(nbElems)
                    }}
                }}
            }
            else
            {
                let nbElemsPtr =
                    (_nbElems as! MetalSharedBuffer<Int32>).buffer
                
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    let offsetStart =
                        (depth + nbChannels * elem) * height
                    
                    for I in 0..<height {
                    for J in 0..<width
                    {
                        let offset = J +
                            (offsetStart + I) * width
                        
                        nbElemsPtr[offset] = 0
                        neurons[depth].get(I, J)!.v[elem].out = 0.0
                    }}
                }}
                
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    let offsetStart =
                        (depth + nbChannels * elem) * height
                    
                    for i in 0..<heightPrev
                    {
                    let startI = _startIndex(index: i,
                                             smallSize: heightPrev,
                                             bigSize: height)
                    let endI = _endIndex(index: i,
                                         smallSize: heightPrev,
                                         bigSize: height)
                    let nbElemsI = endI - startI
                    
                    for j in 0..<widthPrev
                    {
                        let startJ = _startIndex(index: j,
                                                 smallSize: widthPrev,
                                                 bigSize: width)
                        let endJ = _endIndex(index: j,
                                             smallSize: widthPrev,
                                             bigSize: width)
                        let nbElemsJ = endJ - startJ
                        
                        let outPrev =
                            neuronsPrev[depth].get(i, j)!.v[elem].out
                        
                        for k in 0..<nbElemsI {
                        for l in 0..<nbElemsJ
                        {
                            let offset = startJ + l +
                                (offsetStart + startI + k) * width
                            
                            neurons[depth].get(
                                startI + k,
                                startJ + l)!.v[elem].out += outPrev
                            nbElemsPtr[offset] += 1
                        }}
                    }}
                }}
                
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    let offsetStart =
                        (depth + nbChannels * elem) * height
                    
                    for I in 0..<height {
                    for J in 0..<width
                    {
                        let offset = J +
                            (offsetStart + I) * width
                        
                        neurons[depth].get(I, J)!.v[elem].out /=
                            Double(nbElemsPtr[offset])
                    }}
                }}
            }
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
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            let metalKernel = MetalKernel.get
            var command: MetalEncoder
            
            if heightPrev >= height
            {
                command = metalKernel.createEncoder(
                    "adaptiveAvgPoolForward1", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBytes(pNbChannels, atIndex: 1)
                command.setBytes(pDimensions, atIndex: 2)
                command.setBytes(pDimensionsPrev, atIndex: 3)
                command.setBytes(pNbBatch, atIndex: 4)
                command.setBuffer(outs.metal, atIndex: 5)
                
                command.dispatchThreads(
                    width: width * nbChannels,
                    height: height * batchSize
                )
                command.endEncoding()
            }
            else
            {
                var nbElems = _nbElems.nbElems
                var pNbElems: [UInt32] = [UInt32(nbElems)]
                
                command = metalKernel.createEncoder("reset", deviceID: deviceID)
                
                command.setBytes(pNbElems, atIndex: 0)
                command.setBuffer(_nbElems.metal, atIndex: 1)
                
                command.dispatchThreads(nbElems)
                command.endEncoding()
                
                nbElems = outs.nbElems
                pNbElems = [UInt32(nbElems)]
                
                command = metalKernel.createEncoder("reset", deviceID: deviceID)
                
                command.setBytes(pNbElems, atIndex: 0)
                command.setBuffer(outs.metal, atIndex: 1)
                
                command.dispatchThreads(nbElems)
                command.endEncoding()
                
                command = metalKernel.createEncoder(
                    "adaptiveAvgPoolForward2", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBytes(pNbChannels, atIndex: 1)
                command.setBytes(pDimensions, atIndex: 2)
                command.setBytes(pDimensionsPrev, atIndex: 3)
                command.setBytes(pNbBatch, atIndex: 4)
                command.setBuffer(_nbElems.metal, atIndex: 5)
                command.setBuffer(outs.metal, atIndex: 6)
                
                command.dispatchThreads(
                    width: nbChannels,
                    height: batchSize
                )
                command.endEncoding()
            }
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
            
            if layerPrev.dirty
            {
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    for I in 0..<heightPrev {
                    for J in 0..<widthPrev
                    {
                        neuronsPrev[depth].get(I, J)!.v[elem].delta = 0.0
                    }}
                }}
            }
            
            if heightPrev >= height
            {
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    for i in 0..<height
                    {
                    let startI = _startIndex(index: i,
                                             smallSize: height,
                                             bigSize: heightPrev)
                    let endI = _endIndex(index: i,
                                         smallSize: height,
                                         bigSize: heightPrev)
                    let nbElemsI = endI - startI
                        
                    for j in 0..<width
                    {
                        let startJ = _startIndex(index: j,
                                                 smallSize: width,
                                                 bigSize: widthPrev)
                        let endJ = _endIndex(index: j,
                                             smallSize: width,
                                             bigSize: widthPrev)
                        let nbElemsJ = endJ - startJ
                        
                        let nbElems = nbElemsI * nbElemsJ
                        
                        let deltaCur = neurons[depth].get(i, j)!
                            .v[elem].delta / Double(nbElems)
                        
                        for k in 0..<nbElemsI {
                        for l in 0..<nbElemsJ
                        {
                            neuronsPrev[depth].get(
                                startI + k,
                                startJ + l)!.v[elem].delta += deltaCur
                        }}
                    }}
                }}
            }
            else
            {
                let nbElemsPtr =
                    (_nbElems as! MetalSharedBuffer<Int32>).buffer
                
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    let offsetStart =
                        (depth + nbChannels * elem) * height
                    
                    for i in 0..<heightPrev
                    {
                    let startI = _startIndex(index: i,
                                             smallSize: heightPrev,
                                             bigSize: height)
                    let endI = _endIndex(index: i,
                                         smallSize: heightPrev,
                                         bigSize: height)
                    let nbElemsI = endI - startI
                        
                    for j in 0..<widthPrev
                    {
                        let startJ = _startIndex(index: j,
                                                 smallSize: widthPrev,
                                                 bigSize: width)
                        let endJ = _endIndex(index: j,
                                             smallSize: widthPrev,
                                             bigSize: width)
                        let nbElemsJ = endJ - startJ
                        
                        for k in 0..<nbElemsI {
                        for l in 0..<nbElemsJ
                        {
                            let offset = startJ + l +
                                (offsetStart + startI + k) * width
                            
                            neuronsPrev[depth].get(i, j)!
                                .v[elem].delta += neurons[depth].get(
                                startI + k,
                                startJ + l)!.v[elem].delta /
                                Double(nbElemsPtr[offset])
                        }}
                    }}
                }}
            }
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
            
            let widthPrev = layerPrev.width
            let heightPrev = layerPrev.height
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            let metalKernel = MetalKernel.get
            var command: MetalEncoder
            
            if layerPrev.dirty
            {
                let nbElems = layerPrev.delta.nbElems
                let pNbElems: [UInt32] = [UInt32(nbElems)]
                
                command = metalKernel.createEncoder("reset", deviceID: deviceID)
                
                command.setBytes(pNbElems, atIndex: 0)
                command.setBuffer(layerPrev.delta.metal, atIndex: 1)
                
                command.dispatchThreads(nbElems)
                command.endEncoding()
            }
            
            if heightPrev >= height
            {
                command = metalKernel.createEncoder(
                    "adaptiveAvgPoolBackward1", deviceID: deviceID
                )
                command.setBuffer(delta.metal, atIndex: 0)
                command.setBytes(pNbChannels, atIndex: 1)
                command.setBytes(pDimensions, atIndex: 2)
                command.setBytes(pDimensionsPrev, atIndex: 3)
                command.setBytes(pNbBatch, atIndex: 4)
                command.setBuffer(layerPrev.delta.metal, atIndex: 5)
                
                command.dispatchThreads(
                    width: nbChannels,
                    height: batchSize
                )
                command.endEncoding()
            }
            else
            {
                command = metalKernel.createEncoder(
                    "adaptiveAvgPoolBackward2", deviceID: deviceID
                )
                command.setBuffer(delta.metal, atIndex: 0)
                command.setBuffer(_nbElems.metal, atIndex: 1)
                command.setBytes(pNbChannels, atIndex: 2)
                command.setBytes(pDimensions, atIndex: 3)
                command.setBytes(pDimensionsPrev, atIndex: 4)
                command.setBytes(pNbBatch, atIndex: 5)
                command.setBuffer(layerPrev.delta.metal, atIndex: 6)
                
                command.dispatchThreads(
                    width: nbChannels,
                    height: batchSize
                )
                command.endEncoding()
            }
            propagateDirty()
        }
    }
}
