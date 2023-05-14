//
// Normalize2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 14/05/2023.
//

import Foundation

///
/// Layer with a 2D shape neural structure.
///
/// This layer normalizes thanks to an aggregation on the channel axis of the previous layer.
///
public class Normalize12D: Layer2D
{
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D, params: GrAI.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbChannels: layerPrev.nbChannels,
                   height: layerPrev.height,
                   width: layerPrev.width,
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
            
        let layer = Normalize12D(
            layerPrev: layerPrev,
            params: params
        )
        return layer
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
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var norm = 0.0
                    for depth in 0..<nbChannels
                    {
                        let outPrev = neuronsPrev[depth].get(i, j)!
                            .gc[batch][elem].out
                        norm += outPrev * outPrev
                    }
                    norm = sqrt(norm)
                    
                    for depth in 0..<nbChannels
                    {
                        let outPrev = neuronsPrev[depth].get(i, j)!
                            .gc[batch][elem].out
                        neurons[depth].get(i, j)!.gc[batch][elem].out =
                            outPrev / max(norm, 1e-12)
                    }
                }}
            }}
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
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var norm = 0.0
                    for depth in 0..<nbChannels
                    {
                        let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                        norm += outPrev * outPrev
                    }
                    norm = sqrt(norm)
                    
                    for depth in 0..<nbChannels
                    {
                        let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                        neurons[depth].get(i, j)!.v[elem].out =
                            outPrev / max(norm, 1e-12)
                    }
                }}
            }
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    /*public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let nbChannelsPrev = layerPrev.nbChannels
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            let pNbChannelsPrev: [UInt32] = [UInt32(nbChannelsPrev)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            
            let command = MetalKernel.get.createCommand(
                "selfCorrelate2DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbChannelsPrev, atIndex: 1)
            command.setBytes(pDimensionsPrev, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBuffer(outs.metal, atIndex: 4)
            
            command.dispatchThreads(
                width: nbChannelsPrev * nbChannelsPrev,
                height: batchSize
            )
            command.enqueue()
        }
    }*/
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var normTmp = 0.0
                    for depth in 0..<nbChannels
                    {
                        let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                        normTmp += outPrev * outPrev
                    }
                    let norm = sqrt(normTmp)
                    normTmp = pow(norm, 3)
                    
                    for depth in 0..<nbChannels
                    {
                        let outPrev =
                            neuronsPrev[depth].get(i, j)!.v[elem].out
                        let deltaCur = neurons[depth].get(i, j)!.v[elem].delta
                        
                        let newValue: Double
                        if norm > 1e-12
                        {
                            var sum = 0.0
                            for depth1 in 0..<nbChannels
                            {
                                let deltaCur1 = neurons[depth1]
                                    .get(i, j)!.v[elem].delta
                                let outPrev1 = neuronsPrev[depth1]
                                    .get(i, j)!.v[elem].out
                                
                                sum -= outPrev1 * outPrev / normTmp * deltaCur1
                            }
                            
                            sum += deltaCur / norm
                            newValue = sum
                        }
                        else
                        {
                            newValue = deltaCur / 1e-12
                        }
                        
                        if layerPrev.dirty
                        {
                            neuronsPrev[depth].get(i, j)!.v[elem].delta =
                                newValue
                        }
                        else
                        {
                            neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                                newValue
                        }
                    }
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
    /*public override func backwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let nbChannelsPrev = layerPrev.nbChannels
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            let pNbChannelsPrev: [UInt32] = [UInt32(nbChannelsPrev)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "selfCorrelate2DBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(layerPrev.outs.metal, atIndex: 1)
            command.setBytes(pNbChannelsPrev, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: widthPrev * nbChannelsPrev,
                height: heightPrev * batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }*/
}

///
/// Layer with a 2D shape neural structure.
///
/// This layer normalizes thanks to an aggregation on the channel and the the spatial axes
/// of the previous layer.
///
public class Normalize122D: Layer2D
{
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D, params: GrAI.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbChannels: layerPrev.nbChannels,
                   height: layerPrev.height,
                   width: layerPrev.width,
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
            
        let layer = Normalize122D(
            layerPrev: layerPrev,
            params: params
        )
        return layer
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
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                var norm = 0.0
                for i in 0..<height {
                for j in 0..<width {
                for depth in 0..<nbChannels
                {
                    let outPrev = neuronsPrev[depth].get(i, j)!
                        .gc[batch][elem].out
                    norm += outPrev * outPrev
                }}}
                norm = sqrt(norm)
                
                for i in 0..<height {
                for j in 0..<width {
                for depth in 0..<nbChannels
                {
                    let outPrev = neuronsPrev[depth].get(i, j)!
                        .gc[batch][elem].out
                    neurons[depth].get(i, j)!.gc[batch][elem].out =
                        outPrev / max(norm, 1e-12)
                }}}
            }}
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
            for elem in 0..<batchSize
            {
                var norm = 0.0
                for i in 0..<height {
                for j in 0..<width {
                for depth in 0..<nbChannels
                {
                    let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                    norm += outPrev * outPrev
                }}}
                norm = sqrt(norm)
                
                for i in 0..<height {
                for j in 0..<width {
                for depth in 0..<nbChannels
                {
                    let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                    neurons[depth].get(i, j)!.v[elem].out =
                        outPrev / max(norm, 1e-12)
                }}}
            }
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                var normTmp = 0.0
                for i in 0..<height {
                for j in 0..<width {
                for depth in 0..<nbChannels
                {
                    let outPrev = neuronsPrev[depth].get(i, j)!.v[elem].out
                    normTmp += outPrev * outPrev
                }}}
                let norm = sqrt(normTmp)
                normTmp = pow(norm, 3)
                    
                for i in 0..<height {
                for j in 0..<width {
                for depth in 0..<nbChannels
                {
                    let outPrev =
                        neuronsPrev[depth].get(i, j)!.v[elem].out
                    let deltaCur = neurons[depth].get(i, j)!.v[elem].delta
                    
                    let newValue: Double
                    if norm > 1e-12
                    {
                        var sum = 0.0
                        for i1 in 0..<height {
                        for j1 in 0..<width {
                        for depth1 in 0..<nbChannels
                        {
                            let deltaCur1 = neurons[depth1]
                                .get(i1, j1)!.v[elem].delta
                            let outPrev1 = neuronsPrev[depth1]
                                .get(i1, j1)!.v[elem].out
                            
                            sum -= outPrev1 * outPrev / normTmp * deltaCur1
                        }}}
                        
                        sum += deltaCur / norm
                        newValue = sum
                    }
                    else
                    {
                        newValue = deltaCur / 1e-12
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta =
                            newValue
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            newValue
                    }
                }}}
            }
            propagateDirty()
        }
    }
}
