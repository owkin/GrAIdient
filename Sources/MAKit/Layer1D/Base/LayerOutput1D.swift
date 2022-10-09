//
// LayerOutput1D.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

import MetalKit

open class LayerOutput1D: Layer1D
{
    public var coeff: Double = 1.0
    
    public internal(set) var loss: MetalSharedBuffer<Float>! = nil
    
    private enum Keys: String, CodingKey
    {
        case coeff
    }
    
    public init(layerPrev: Layer1D, params: MAKit.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbNeurones: layerPrev.nbNeurones,
                   params: params)
    }
    
    public required init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        let coeff = try container.decode(Float.self, forKey: .coeff)
        self.coeff = Double(coeff)
        try super.init(from: decoder)
    }
    
    public override func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        try container.encode(Float(coeff), forKey: .coeff)
        try super.encode(to: encoder)
    }
    
    open override func resetKernelGPU()
    {
        super.resetKernelGPU()
        loss = nil
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    open override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for depth in 0..<nbNeurones
            {
                neurones.get(depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronesPrev = layerPrev.neurones
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for j in 0..<nbNeurones
            {
                neurones.get(j)!.gc[batch][elem].out =
                    neuronesPrev.get(j)!.gc[batch][elem].out
            }}}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    open override func forwardGCGPU() throws
    {
        try forwardGCCPU()
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    open override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize {
            for j in 0..<nbNeurones
            {
                neurones.get(j)!.v[elem].out = neuronesPrev.get(j)!.v[elem].out
            }}
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    open override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
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
    open override func backwardCPU()
    {
        // Note that backward is not called except when it is
        // an intermediate layer.
        // Model.backward is only called on not dirty layers.
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize {
            for depth in 0..<nbNeurones
            {
                let delta = neurones.get(depth)!.v[elem].delta
                if layerPrev.dirty
                {
                    neuronesPrev.get(depth)!.v[elem].delta = delta
                }
                else
                {
                    neuronesPrev.get(depth)!.v[elem].delta += delta
                }
            }}
            propagateDirty()
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    open override func backwardGPU() throws
    {
        // Note that backward is not called except when it is
        // an intermediate layer.
        // Model.backward is only called on not dirty layers.
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
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
}
