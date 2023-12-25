//
// IRDFT2RGB.swift
// GrAIdient
//
// Created by Jean-François Reboud on 25/11/2022.
//

import Foundation

///
/// Layer with a 2D shape neural structure.
///
/// This layer computes Inverse Real Discrete Fourier Transform to output a RGB image.
///
public class IRDFT2RGB: Layer2D
{
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D, params: GrAI.Model.Params) throws
    {
        let width = layerPrev.width
        let height = layerPrev.height
        let nbChannels = layerPrev.nbChannels
        
        if nbChannels != 6
        {
            throw LayerError.Init(
                message: "IRDFT2RGB input channels should be 6."
            )
        }
        super.init(layerPrev: layerPrev,
                   nbChannels: nbChannels / 2,
                   height: height,
                   width: width,
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
            
        let layer = try! IRDFT2RGB(
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
            for elem in 0..<nbGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var sum: Double = 0.0
                    for k in 0..<height {
                    for l in 0..<width
                    {
                        let realPrev = neuronsPrev[2 * depth].get(k, l)!
                            .gc[batch][elem].out
                        let imPrev = neuronsPrev[2 * depth + 1].get(k, l)!
                            .gc[batch][elem].out
                        var angle = 2.0 * Double.pi
                        angle *=
                            (Double(i) / Double(height) * Double(k) +
                             Double(j) / Double(width) * Double(l))
                        sum += realPrev * cos(angle) - imPrev * sin(angle)
                    }}
                    sum /= Double(height * width)
                    neurons[depth].get(i, j)!.gc[batch][elem].out = sum
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
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var sum: Double = 0.0
                    for k in 0..<height {
                    for l in 0..<width
                    {
                        let realPrev =
                            neuronsPrev[2 * depth].get(k, l)!.v[elem].out
                        let imPrev =
                            neuronsPrev[2 * depth + 1].get(k, l)!.v[elem].out
                        var angle = 2.0 * Double.pi
                        angle *=
                            (Double(i) / Double(height) * Double(k) +
                             Double(j) / Double(width) * Double(l))
                        sum += realPrev * cos(angle) - imPrev * sin(angle)
                    }}
                    sum /= Double(height * width)
                    neurons[depth].get(i, j)!.v[elem].out = sum
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
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            
            let command = MetalKernel.get.createEncoder(
                "IRDFT2RGBForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBuffer(outs.metal, atIndex: 4)
            
            command.dispatchThreads(
                width: width * nbChannels,
                height: height * batchSize
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
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    var sum1: Double = 0.0
                    var sum2: Double = 0.0
                    for k in 0..<height {
                    for l in 0..<width
                    {
                        let delta =
                            neurons[depth].get(k, l)!.v[elem].delta
                        var angle = 2.0 * Double.pi
                        angle *=
                            (Double(i) / Double(height) * Double(k) +
                             Double(j) / Double(width) * Double(l))
                        sum1 += delta * cos(angle)
                        sum2 -= delta * sin(angle)
                    }}
                    sum1 /= Double(height * width)
                    sum2 /= Double(height * width)
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev[2 * depth].get(i, j)!
                            .v[elem].delta = sum1
                        neuronsPrev[2 * depth + 1].get(i, j)!
                            .v[elem].delta = sum2
                    }
                    else
                    {
                        neuronsPrev[2 * depth].get(i, j)!
                            .v[elem].delta += sum1
                        neuronsPrev[2 * depth + 1].get(i, j)!
                            .v[elem].delta += sum2
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
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createEncoder(
                "IRDFT2RGBBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pDirty, atIndex: 4)
            command.setBuffer(layerPrev.delta.metal, atIndex: 5)
            
            command.dispatchThreads(
                width: width * nbChannels,
                height: height * batchSize
            )
            command.endEncoding()
            
            propagateDirty()
        }
    }
}
