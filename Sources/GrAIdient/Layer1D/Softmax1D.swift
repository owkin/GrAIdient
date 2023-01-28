//
// Softmax1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 28/01/2023.
//

import Foundation

public class Softmax1D: Layer1D
{
    public init(layerPrev: Layer1D, params: GrAI.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbNeurons: layerPrev.nbNeurons,
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
        let layerPrev = mapping[idPrev] as! Layer1D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = Softmax1D(
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
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for j in 0..<nbNeurons
            {
                neurons.get(j)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                var sum = 0.0
                for depth in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(depth)!.gc[batch][elem].out
                    sum += exp(outPrev)
                }
                
                for depth in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(depth)!.gc[batch][elem].out
                    neurons.get(depth)!.gc[batch][elem].out = exp(outPrev)/sum
                }
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
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                var sum1 = 0.0
                for depth1 in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(depth1)!.v[elem].out
                    sum1 += exp(outPrev)
                }
                
                for depth1 in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(depth1)!.v[elem].out
                    neurons.get(depth1)!.v[elem].out = exp(outPrev) / sum1
                }
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
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            
            let command = MetalKernel.get.createCommand(
                "softmax1DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pNbBatch, atIndex: 2)
            command.setBuffer(outs.metal, atIndex: 3)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                var sum1: Double = 0.0
                for depth1 in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev.get(depth1)!.v[elem].out
                    sum1 += exp(outPrev)
                }
                    
                for depth1 in 0..<nbNeurons
                {
                    let outPrev1 = neuronsPrev.get(depth1)!.v[elem].out
                    let deltaCur1 = neurons.get(depth1)!.v[elem].delta
                    
                    var sum2: Double = 0.0
                    for depth2 in 0..<nbNeurons
                    {
                        let deltaCur2 = neurons.get(depth2)!.v[elem].delta
                        let outPrev2 = neuronsPrev.get(depth2)!.v[elem].out
                        sum2 += exp(outPrev1 + outPrev2) * deltaCur2
                    }
                    
                    let tmp = exp(outPrev1) * deltaCur1 / sum1
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(depth1)!.v[elem].delta =
                            -sum2 / (sum1 * sum1) + tmp
                    }
                    else
                    {
                        neuronsPrev.get(depth1)!.v[elem].delta +=
                            -sum2 / (sum1 * sum1) + tmp
                    }
                }
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
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "softmax1DBackward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pDirty, atIndex: 4)
            command.setBuffer(layerPrev.delta.metal, atIndex: 5)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
