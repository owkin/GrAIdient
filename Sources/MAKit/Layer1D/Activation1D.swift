//
// Activation1D.swift
// MAKit
//
// Created by Jean-François Reboud on 10/10/2022.
//

import MetalKit

public class Activation1D: Layer1D
{
    let _activation: ActivationFunction?
    
    var _tmp: MetalPrivateBuffer<Float>! = nil
    
    private enum Keys: String, CodingKey
    {
        case activation
    }
    
    public init(layerPrev: Layer1D,
                activation: String,
                params: MAKit.Model.Params)
    {
        _activation = MAKit.Model.Activation.build(activation)
        
        super.init(layerPrev: layerPrev,
                   nbNeurones: layerPrev.nbNeurones,
                   params: params)
    }
    
    public init(layerPrev: Layer?,
                nbNeurones: Int, activation: String?,
                params: MAKit.Model.Params)
    {
        if let activationStr = activation
        {
            _activation = MAKit.Model.Activation.build(activationStr)
        }
        else
        {
            _activation = nil
        }
        
        super.init(layerPrev: layerPrev, nbNeurones: nbNeurones, params: params)
    }
    
    public required init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        _activation =
            try container.decodeIfPresent(ActivationContainer.self,
                                          forKey: .activation)?.activation
        try super.init(from: decoder)
    }
    
    public override func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        if let activation = _activation
        {
            try container.encode(ActivationContainer(activation),
                                 forKey: Keys.activation)
        }
        try super.encode(to: encoder)
    }
    
    public override func copy(
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = mapping[idPrev] as! Layer1D
        
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
        
        let layer = Activation1D(
            layerPrev: layerPrev,
            activation: _activation!.name,
            params: params
        )
        return layer
    }
    
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _tmp = nil
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        try _forwardGC()
        _activation!.forwardGC(self)
    }
    
    private func _forwardGC() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for j in 0..<nbNeurones
            {
                neurones.get(j)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronesPrev = layerPrev.neurones
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbNeurones
                {
                    neurones.get(depth)!.gc[batch][elem].out =
                        neuronesPrev.get(depth)!.gc[batch][elem].out
                }
            }}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try _forwardGC()
        _activation!.forwardGC(self)
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurones
                {
                    neurones.get(depth)!.v[elem].out =
                        neuronesPrev.get(depth)!.v[elem].out
                }
            }
            
            _activation!.forwardCPU(self)
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
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
            let threadsPerGrid = MTLSize(width: nbElems,
                                         height: 1,
                                         depth: 1)
            command.dispatchThreads(
                threadsPerGrid: threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            command.enqueue()
            
            _activation!.forwardGPU(self)
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        _activation!.backwardCPU(self)
        
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurones
                {
                    if layerPrev.dirty
                    {
                        neuronesPrev.get(depth)!.v[elem].delta =
                            neurones.get(depth)!.v[elem].delta
                    }
                    else
                    {
                        neuronesPrev.get(depth)!.v[elem].delta +=
                            neurones.get(depth)!.v[elem].delta
                    }
                }
            }
            propagateDirty()
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        _activation!.backwardGPU(self)
        
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
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
}
