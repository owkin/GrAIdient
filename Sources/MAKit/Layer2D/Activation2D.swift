//
// Activation2D.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

import MetalKit

/// Layer with a 2D shape neural structure and an activation function.
public class Activation2D: Layer2D
{
    /// The activation function.
    let _activation: ActivationFunction?
    
    /// Pre output buffer (result of the forward pass before applying activation)
    /// used in the GPU execution context.
    var _tmp: MetalPrivateBuffer<Float>! = nil
    
    private enum Keys: String, CodingKey
    {
        case activation
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - activation: The activation function.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D, activation: String,
                params: MAKit.Model.Params)
    {
        _activation = MAKit.Model.Activation.build(activation)
        
        super.init(layerPrev: layerPrev,
                   nbFilters: layerPrev.nbFilters,
                   height: layerPrev.height,
                   width: layerPrev.width,
                   params: params)
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - nbFilters: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - activation: The activation function.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer?,
                nbFilters: Int, height: Int, width: Int, activation: String?,
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
        
        super.init(layerPrev: layerPrev,
                   nbFilters: nbFilters,
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
        let container = try decoder.container(keyedBy: Keys.self)
        _activation =
            try container.decodeIfPresent(ActivationContainer.self,
                                          forKey: .activation)?.activation
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
        if let activation = _activation
        {
            try container.encode(ActivationContainer(activation),
                                 forKey: Keys.activation)
        }
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
        
        let layer = Activation2D(
            layerPrev: layerPrev,
            activation: _activation!.name,
            params: params
        )
        return layer
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// State resources are the resources that are dependent on the batch size.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _tmp = nil
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        try _forwardGC()
        _activation!.forwardGC(self)
    }
    
    ///
    /// Apply the forward pass (until the activation function) of the Gradient Checking.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    private func _forwardGC() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for j in 0..<nbFilters
            {
                neurons[j].initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbFilters {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.gc[batch][elem].out =
                        neuronsPrev[depth].get(i, j)!.gc[batch][elem].out
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
        try _forwardGC()
        _activation!.forwardGC(self)
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
                for depth in 0..<nbFilters {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.v[elem].out =
                        neuronsPrev[depth].get(i, j)!.v[elem].out
                }}}
            }
            
            _activation!.forwardCPU(self)
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
        
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for depth in 0..<nbFilters {
                for i in 0..<height {
                for j in 0..<width
                {
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta =
                            neurons[depth].get(i, j)!.v[elem].delta
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            neurons[depth].get(i, j)!.v[elem].delta
                    }
                }}}
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
        _activation!.backwardGPU(self)
        
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
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
