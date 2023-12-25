//
// SelfCorrelate2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 13/05/2023.
//

///
/// Layer with a 2D shape neural structure.
///
/// This layer computes correlation of previous channels.
///
public class SelfCorrelate2D: Layer2D
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
        let nbChannels = layerPrev.nbChannels
        super.init(layerPrev: layerPrev,
                   nbChannels: 1,
                   height: nbChannels,
                   width: nbChannels,
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
            
        let layer = SelfCorrelate2D(
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
            for i in 0..<height {
            for j in 0..<width
            {
                neurons[0].get(i, j)!.initGC(
                    batchSize: batchSize,
                    nbGC: nbGC
                )
            }}
            
            let neuronsPrev = layerPrev.neurons
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for channel1 in 0..<height {
                for channel2 in 0..<width
                {
                    var correlation = 0.0
                    for i in 0..<heightPrev {
                    for j in 0..<widthPrev
                    {
                        correlation +=
                            neuronsPrev[channel1].get(i, j)!.gc[batch][elem].out
                            *
                            neuronsPrev[channel2].get(i, j)!.gc[batch][elem].out
                    }}
                    
                    neurons[0].get(channel1, channel2)!
                        .gc[batch][elem].out = correlation
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
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            for elem in 0..<batchSize
            {
                for channel1 in 0..<height {
                for channel2 in 0..<width
                {
                    var correlation = 0.0
                    for i in 0..<heightPrev {
                    for j in 0..<widthPrev
                    {
                        correlation +=
                            neuronsPrev[channel1].get(i, j)!.v[elem].out *
                            neuronsPrev[channel2].get(i, j)!.v[elem].out
                    }}
                    
                    neurons[0].get(channel1, channel2)!.v[elem].out =
                        correlation
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
            
            let nbChannelsPrev = layerPrev.nbChannels
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            let pNbChannelsPrev: [UInt32] = [UInt32(nbChannelsPrev)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            
            let command = MetalKernel.get.createEncoder(
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
            command.endEncoding()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            let nbChannelsPrev = layerPrev.nbChannels
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            for elem in 0..<batchSize {
            for depthPrev in 0..<nbChannelsPrev {
            for i in 0..<heightPrev {
            for j in 0..<widthPrev
            {
                var correlation = 0.0
                for col in 0..<width
                {
                    correlation +=
                        neurons[0].get(depthPrev, col)!.v[elem].delta *
                        neuronsPrev[col].get(i, j)!.v[elem].out
                }
                for row in 0..<height
                {
                    correlation +=
                        neurons[0].get(row, depthPrev)!.v[elem].delta *
                        neuronsPrev[row].get(i, j)!.v[elem].out
                }
                
                if layerPrev.dirty
                {
                    neuronsPrev[depthPrev].get(i, j)!.v[elem].delta =
                        correlation
                }
                else
                {
                    neuronsPrev[depthPrev].get(i, j)!.v[elem].delta +=
                        correlation
                }
            }}}}
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
            
            let nbChannelsPrev = layerPrev.nbChannels
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            let pNbChannelsPrev: [UInt32] = [UInt32(nbChannelsPrev)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createEncoder(
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
            command.endEncoding()
            
            propagateDirty()
        }
    }
}
