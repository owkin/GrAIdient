//
// SelectNeurons1D.swift
// MAKit
//
// Created by Jean-François Reboud on 10/10/2022.
//

import MetalKit

///
/// Layer with a 1D shape neural structure.
///
/// This layer enables to reduce the number of neurons of a 1D layer.
/// 
public class SelectNeurons1D: Layer1D
{
    /// List of neurons to select.
    let _neurons: [Int]
    /// List of coefficients to scale each selected neuron.
    let _coeffs: [Double]
    
    private enum Keys: String, CodingKey
    {
        case neurons
        case coeffs
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - neurons: The list of neurons to select.
    ///     - coeffs: The list of coefficients to scale each selected neuron.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer1D,
                neurons: [Int],
                coeffs: [Double],
                params: MAKit.Model.Params)
    {
        _neurons = neurons
        _coeffs = coeffs
        super.init(layerPrev: layerPrev,
                   nbNeurons: _neurons.count,
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
        _neurons = try container.decode([Int].self, forKey: .neurons)
        _coeffs = try container.decode([Double].self, forKey: .coeffs)
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
        try container.encode(_neurons, forKey: Keys.neurons)
        try container.encode(_coeffs, forKey: Keys.coeffs)
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
        let layerPrev = mapping[idPrev] as! Layer1D
        
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
            
        let layer = SelectNeurons1D(
            layerPrev: layerPrev,
            neurons: _neurons,
            coeffs: _coeffs,
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
                for depth in 0..<nbNeurons
                {
                    neurons.get(depth)!.gc[batch][elem].out =
                        _coeffs[depth] *
                        neuronsPrev.get(_neurons[depth])!.gc[batch][elem].out
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
                for depth in 0..<nbNeurons
                {
                    neurons.get(depth)!.v[elem].out =
                        _coeffs[depth] *
                        neuronsPrev.get(_neurons[depth])!.v[elem].out
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
            let pNbNeuronsPrev: [UInt32] = [UInt32(layerPrev.nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            var pNeurons = [UInt32]()
            for neuron in _neurons
            {
                pNeurons.append(UInt32(neuron))
            }
            var pCoeffs = [Float]()
            for coeff in _coeffs
            {
                pCoeffs.append(Float(coeff))
            }
            
            let command = MetalKernel.get.createCommand(
                "selectNeurons1DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pNbNeuronsPrev, atIndex: 2)
            command.setBytes(pNeurons, atIndex: 3)
            command.setBytes(pCoeffs, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            
            let threadsPerThreadgroup = MTLSizeMake(8, 8, 1)
            let threadsPerGrid = MTLSize(width: nbNeurons,
                                         height: batchSize,
                                         depth: 1)
            command.dispatchThreads(
                threadsPerGrid: threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
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
            if layerPrev.dirty
            {
                for elem in 0..<batchSize {
                for depth in 0..<layerPrev.nbNeurons
                {
                    neuronsPrev.get(depth)!.v[elem].delta = 0.0
                }}
            }
            
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    neuronsPrev.get(_neurons[depth])!.v[elem].delta +=
                        _coeffs[depth] * neurons.get(depth)!.v[elem].delta
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
            let pNbNeuronsPrev: [UInt32] = [UInt32(layerPrev.nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            var pNeurons = [UInt32]()
            for neuron in _neurons
            {
                pNeurons.append(UInt32(neuron))
            }
            var pCoeffs = [Float]()
            for coeff in _coeffs
            {
                pCoeffs.append(Float(coeff))
            }
            
            var command: MetalCommand
            
            if layerPrev.dirty
            {
                let nbElems = layerPrev.delta.nbElems
                let pNbElems: [UInt32] = [UInt32(nbElems)]
                
                command = MetalKernel.get.createCommand(
                    "reset", deviceID: deviceID
                )
                command.setBytes(pNbElems, atIndex: 0)
                command.setBuffer(layerPrev.delta.metal, atIndex: 1)
                
                let threads = command.maxThreadsPerThreadgroup >= nbElems ?
                    command.threadExecutionWidth :
                    command.maxThreadsPerThreadgroup
                let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
                let threadsPerGrid = MTLSize(width: nbElems,
                                             height: 1,
                                             depth: 1)
                command.dispatchThreads(
                    threadsPerGrid: threadsPerGrid,
                    threadsPerThreadgroup: threadsPerThreadgroup
                )
                command.enqueue()
            }
            
            command = MetalKernel.get.createCommand(
                "selectNeurons1DBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbNeurons, atIndex: 1)
            command.setBytes(pNbNeuronsPrev, atIndex: 2)
            command.setBytes(pNeurons, atIndex: 3)
            command.setBytes(pCoeffs, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            let threadsPerThreadgroup = MTLSizeMake(8, 8, 1)
            let threadsPerGrid = MTLSize(width: nbNeurons,
                                         height: batchSize,
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