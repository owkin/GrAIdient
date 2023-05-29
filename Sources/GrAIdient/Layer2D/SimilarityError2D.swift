//
// SimilarityError2D.swift
// GrAIdient
//
//  Created by Jean-François Reboud on 29/05/2023.
//

import Foundation

///
/// Output layer with a 2D shape neural structure and a loss that computes pairwise similarity within batch
/// of previous layers.
///
public class SimilarityError2D: LayerMerge2D
{
    /// Coefficient to be applied to the loss compuptation.
    public var coeff: Double = 1.0
    
    ///
    /// Loss buffer in the GPU execution context.
    /// Shape ~ (batch, batch).
    ///
    public internal(set) var loss: MetalSharedBuffer<Float>! = nil
    
    /// Batch size sum in the previous layers.
    public var mergedBatchSize: Int
    {
        get {
            var sum = 0
            for layerPrev in _layersPrev
            {
                sum += layerPrev.batchSize
            }
            return sum
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case coeff
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layersPrev: List of previous layers that have been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layersPrev: [Layer2D], params: GrAI.Model.Params)
    {
        let layer0 = layersPrev[0]
        for layerPrev in layersPrev
        {
            if layerPrev.nbChannels != 1
            {
                fatalError("Previous layer should have only 1 channel.")
            }
            if layerPrev.height != layer0.height ||
               layerPrev.width != layer0.width
            {
                fatalError("Layer structure error.")
            }
        }
        
        super.init(layersPrev: layersPrev,
                   nbChannels: 1,
                   height: layer0.height,
                   width: layer0.width,
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
        let container = try decoder.container(keyedBy: Keys.self)
        let coeff = try container.decode(Float.self, forKey: .coeff)
        self.coeff = Double(coeff)
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
        try container.encode(Float(coeff), forKey: .coeff)
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
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        var layersPrev = [Layer2D]()
        for idPrev in _idsPrev
        {
            layersPrev.append(mapping[idPrev] as! Layer2D)
        }
        
        let layer = SimilarityError2D(layersPrev: layersPrev, params: params)
        layer.coeff = self.coeff
        
        return layer
    }
    
    ///
    /// Setup loss state  in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter batchSize: The batch size of data.
    ///
    public func checkLossGPU(batchSize: Int) throws
    {
        if loss == nil
        {
            loss = MetalSharedBuffer<Float>(
                batchSize * batchSize,
                deviceID: deviceID
            )
        }
        else if batchSize <= 0 || batchSize * batchSize > loss.nbElems
        {
            throw LayerError.BatchSize
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardGCCPU() throws
    {
        try checkStateCPU(batchSize: mergedBatchSize)
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for i in 0..<height {
        for j in 0..<width
        {
            neurons[0].get(i, j)!.initGC(
                batchSize: mergedBatchSize,
                nbGC: nbGC
            )
        }}
        
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let batchSize = _layersPrev[num].batchSize
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            
            for batch in 0..<batchSize {
            for elem in 0..<nbSameElems
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let outPrev = neuronsPrev[0].get(i, j)!.gc[batch][elem].out
                    neurons[0].get(i, j)!.gc[curElem+batch][elem].out = outPrev
                }}
            }}
            curElem += batchSize
        }
        
        curElem = 0
        for num in 0..<_layersPrev.count
        {
            let batchSize = _layersPrev[num].batchSize
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            
            for batch in 0..<batchSize {
            var offset = nbSameElems
            var nbLastElems = [Int](repeating: nbSameElems,
                                    count: _layersPrev.count)
            for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
            for elem in 0..<nbElemsTmp
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let outPrev: Double
                    if num == index
                    {
                        outPrev = neuronsPrev[0].get(i, j)!
                            .gc[batch][nbLastElems[index]+elem].out
                    }
                    else
                    {
                        outPrev = neuronsPrev[0].get(i, j)!.v[batch].out
                    }
                    
                    neurons[0].get(i, j)!.gc[curElem+batch][offset+elem].out =
                        outPrev
                }}
            }
            
            offset += nbElemsTmp
            nbLastElems[index] += nbElemsTmp
            }}
            curElem += batchSize
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardGCGPU() throws
    {
        try checkStateCPU(batchSize: mergedBatchSize)
        
        for num in 0..<_layersPrev.count
        {
            MetalKernel.get.download([(_layersPrev[num] as! Layer2D).outs])
        }
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for i in 0..<height {
        for j in 0..<width
        {
            neurons[0].get(i, j)!.initGC(
                batchSize: batchSize,
                nbGC: nbGC
            )
        }}
        
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let batchSize = _layersPrev[num].batchSize
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            
            for batch in 0..<batchSize {
            for elem in 0..<nbSameElems
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let outPrev = neuronsPrev[0].get(i, j)!.gc[batch][elem].out
                    neurons[0].get(i, j)!.gc[curElem+batch][elem].out = outPrev
                }}
            }}
            curElem += batchSize
        }
    
        curElem = 0
        for num in 0..<_layersPrev.count
        {
            let batchSize = _layersPrev[num].batchSize
            let outsPrevPtr = (_layersPrev[num] as! Layer2D).outs.shared.buffer
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            
            for batch in 0..<batchSize {
            var offset = nbSameElems
            var nbLastElems = [Int](repeating: nbSameElems,
                                    count: _layersPrev.count)
            for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
            for elem in 0..<nbElemsTmp
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let outPrev: Double
                    if num == index
                    {
                        outPrev = neuronsPrev[0].get(i, j)!
                            .gc[batch][nbLastElems[index]+elem].out
                    }
                    else
                    {
                        let offsetStart = nbChannels * batch * height
                        let offsetTmp = j + (offsetStart + i) * width
                        
                        outPrev = Double(outsPrevPtr[offsetTmp])
                    }
                    
                    neurons[0].get(i, j)!.gc[curElem+batch][offset+elem].out =
                        outPrev
                }}
            }
            
            offset += nbElemsTmp
            nbLastElems[index] += nbElemsTmp
            }}
            curElem += batchSize
        }
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardCPU() throws
    {
        try checkStateCPU(batchSize: mergedBatchSize)
        
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let batchSize = _layersPrev[num].batchSize
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                neurons[0].get(i, j)!.v[curElem+elem].out =
                    neuronsPrev[0].get(i, j)!.v[elem].out
            }}}
            curElem += batchSize
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardGPU() throws
    {
        try checkStateForwardGPU(batchSize: batchSize)
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let batchSize = _layersPrev[num].batchSize
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            
            command = metalKernel.createCommand(
                "concat02DForward", deviceID: deviceID
            )
            command.setBuffer(
                (_layersPrev[num] as! Layer2D).outs.metal, atIndex: 0
            )
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pDimensions, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBuffer(outs.metal, atIndex: 5)
            
            command.dispatchThreads(
                width: width * nbChannels,
                height: height * batchSize
            )
            command.enqueue()
            
            globalOffset += batchSize
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    open override func backwardCPU()
    {
        // Note that backward is not called except when it is
        // an intermediate layer.
        // Model.backward is only called on non dirty layers.
        
        if !mustComputeBackward
        {
            return
        }
        
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let batchSize = _layersPrev[num].batchSize
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            
            if !_layersPrev[num].computeDelta
            {
                curElem += batchSize
                continue
            }
            
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                let deltaCur = neurons[0].get(i, j)!.v[curElem+elem].delta
                
                if _layersPrev[num].dirty
                {
                    neuronsPrev[0].get(i, j)!.v[elem].delta = deltaCur
                }
                else
                {
                    neuronsPrev[0].get(i, j)!.v[elem].delta += deltaCur
                }
            }}}
            curElem += batchSize
        }
        propagateDirty()
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func backwardGPU() throws
    {
        // Note that backward is not called except when it is
        // an intermediate layer.
        // Model.backward is only called on non dirty layers.
        
        if !mustComputeBackward
        {
            return
        }
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! Layer2D
            let batchSize = layerPrev.batchSize
            
            if !_layersPrev[num].computeDelta
            {
                globalOffset += batchSize
                continue
            }
            
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            command = metalKernel.createCommand(
                "concat02DBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pDimensions, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: width * nbChannels,
                height: height * batchSize
            )
            command.enqueue()
            
            globalOffset += batchSize
        }
        propagateDirty()
    }
    
    ///
    /// Estimate the gradients of weights thanks to Gradient Checking.
    ///
    /// - Returns: The estimated gradients of weights.
    ///
    public func collectGradientsApprox<T: BinaryFloatingPoint>() -> [T]
    {
        var gradients = [T]()
        let nbGradients = neurons.first!.get(0)!.nbGC / 2
        for elem in 0..<nbGradients
        {
            let loss1: T = getLossGC(elem: 2 * elem)
            let loss2: T = getLossGC(elem: 2 * elem + 1)
            
            let gradient = (loss1 - loss2) / T(2 * Ɛ)
            gradients.append(gradient)
        }
        return gradients
    }
    
    ///
    /// Get the loss consecutive of a modified weights during the Gradient Checking process.
    ///
    /// - Parameter elem: The modified weight for which we collect the resulting loss.
    /// - Returns: The loss value.
    ///
    func getLossGC<T: BinaryFloatingPoint>(elem: Int) -> T
    {
        var loss = T(0.0)
        for batch1 in 0..<mergedBatchSize {
        for batch2 in 0..<mergedBatchSize
        {
            if batch1 == batch2
            {
                continue
            }
            
            for i in 0..<height {
            for j in 0..<width
            {
                let out1 = T(neurons[0].get(i, j)!.gc[batch1][elem].out)
                let out2 = T(neurons[0].get(i, j)!.gc[batch2][elem].out)
                
                loss += out1 * out2
            }}
        }}
        return T(coeff) * loss / T(mergedBatchSize)
    }
    
    ///
    /// Get loss in the CPU execution context.
    ///
    /// - Returns: The loss value.
    ///
    public func getLossCPU<T: BinaryFloatingPoint>() -> T
    {
        var loss = T(0.0)
        for elem1 in 0..<mergedBatchSize {
        for elem2 in 0..<mergedBatchSize
        {
            if elem1 == elem2
            {
                continue
            }
            
            for i in 0..<height {
            for j in 0..<width
            {
                let out1 = T(neurons[0].get(i, j)!.v[elem1].out)
                let out2 = T(neurons[0].get(i, j)!.v[elem2].out)
                
                loss += out1 * out2
            }}
        }}
        return T(coeff) * loss / T(mergedBatchSize)
    }
    
    ///
    /// Get loss in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Returns: The loss value.
    ///
    public func getLossGPU() throws -> Float
    {
        try checkLossGPU(batchSize: mergedBatchSize)
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pNbBatch: [UInt32] = [UInt32(mergedBatchSize)]
        
        let command = MetalKernel.get.createCommand(
            "similarBatchError2DLoss", deviceID: deviceID
        )
        command.setBuffer(outs.metal, atIndex: 0)
        command.setBytes(pNbChannels, atIndex: 1)
        command.setBytes(pDimensions, atIndex: 2)
        command.setBytes(pNbBatch, atIndex: 3)
        command.setBuffer(loss.metal, atIndex: 4)
        
        command.dispatchThreads(width: batchSize, height: batchSize)
        command.enqueue()
        
        MetalKernel.get.download([loss])
        var loss: Float = 0.0
        let lossPtr = self.loss.buffer
        for elem1 in 0..<mergedBatchSize {
        for elem2 in 0..<mergedBatchSize
        {
            if elem1 == elem2
            {
                continue
            }
            loss += lossPtr[elem2 + mergedBatchSize * elem1]
        }}
        return Float(coeff) * loss / Float(mergedBatchSize)
    }
    
    ///
    /// Compute the derivative of the loss in the CPU execution context.
    ///
    /// This function is necessary to initialize the backward pass !
    /// In a way, it plays a similar role as the `setData` of the first layer.
    ///
    public func lossDerivativeCPU()
    {
        if !mustComputeBackward
        {
            return
        }
        
        var curElem = 0
        for num in 0..<_layersPrev.count
        {
            let batchSize = _layersPrev[num].batchSize
            let neuronsPrev = (_layersPrev[num] as! Layer2D).neurons
            
            if !_layersPrev[num].computeDelta
            {
                curElem += batchSize
                continue
            }
            
            for elem in 0..<batchSize {
            for i in 0..<height {
            for j in 0..<width
            {
                var sum = 0.0
                for elem1 in 0..<mergedBatchSize
                {
                    if elem1 == elem+curElem
                    {
                        continue
                    }
                    sum += 2 * neurons[0].get(i, j)!.v[elem1].out
                }
                
                if _layersPrev[num].dirty
                {
                    neuronsPrev[0].get(i, j)!.v[elem].delta =
                        coeff / Double(mergedBatchSize) * sum
                }
                else
                {
                    neuronsPrev[0].get(i, j)!.v[elem].delta +=
                        coeff / Double(mergedBatchSize) * sum
                }
            }}}
            curElem += batchSize
        }
        propagateDirty()
    }
    
    ///
    /// Compute the derivative of the loss in the GPU execution context.
    ///
    /// This function is necessary to initialize the backward pass !
    /// In a way, it plays a similar role as the `setData` of the first layer.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    public func lossDerivativeGPU() throws
    {
        if !mustComputeBackward
        {
            return
        }
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pCoeff: [Float] = [Float(coeff)]
        let pNbBatch: [UInt32] = [UInt32(mergedBatchSize)]
        
        let metalKernel = MetalKernel.get
        var command: MetalCommand
        
        var globalOffset = 0
        for num in 0..<_layersPrev.count
        {
            let layerPrev = _layersPrev[num] as! Layer2D
            let batchSize = layerPrev.batchSize
            
            if !layerPrev.computeDelta
            {
                globalOffset += batchSize
                continue
            }
            
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pGlobalOffset: [UInt32] = [UInt32(globalOffset)]
            let pNbBatchPrev: [UInt32] = [UInt32(batchSize)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            command = metalKernel.createCommand(
                "similarError2DLossDerivative", deviceID: deviceID
            )
            command.setBuffer(outs.metal, atIndex: 0)
            command.setBytes(pGlobalOffset, atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pDimensions, atIndex: 3)
            command.setBytes(pCoeff, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pNbBatchPrev, atIndex: 6)
            command.setBytes(pDirty, atIndex: 7)
            command.setBuffer(layerPrev.delta.metal, atIndex: 8)
            
            command.dispatchThreads(
                width: width * height,
                height: batchSize
            )
            command.enqueue()
            
            globalOffset += batchSize
        }
        propagateDirty()
    }
}
