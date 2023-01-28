//
// Softmax1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 28/01/2023.
//

import Foundation

public class Softmax1D: Layer1D
{
    let _size: Int
    
    private enum Keys: String, CodingKey
    {
        case size
    }
    
    public init(layerPrev: Layer1D, size: Int, params: GrAI.Model.Params)
    {
        let nbNeurons = layerPrev.nbNeurons
        if nbNeurons % size != 0
        {
            fatalError(
                "'nbNeurons' (\(nbNeurons) " +
                "should be a multiple of size (\(size))."
            )
        }
        
        _size = size
        super.init(layerPrev: layerPrev,
                   nbNeurons: nbNeurons,
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
        let values = try decoder.container(keyedBy: Keys.self)
        _size = try values.decode(Int.self, forKey: Keys.size)
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
        try container.encode(_size, forKey: Keys.size)
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
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = Softmax1D(
            layerPrev: layerPrev,
            size: _size,
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
            
            let nbBlocks = nbNeurons / _size
            let neuronsPrev = layerPrev.neurons
            
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for block in 0..<nbBlocks
                {
                    var sum1 = 0.0
                    for j1 in 0..<_size
                    {
                        let outPrev = neuronsPrev.get(
                            j1 + block * _size)!.gc[batch][elem].out
                        sum1 += exp(outPrev)
                    }
                    
                    for j1 in 0..<_size
                    {
                        let outPrev = neuronsPrev.get(
                            j1 + block * _size)!.gc[batch][elem].out
                        neurons.get(j1 + block * _size)!.gc[batch][elem].out =
                            exp(outPrev) / sum1
                    }
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
            
            let nbBlocks = nbNeurons / _size
            let neuronsPrev = layerPrev.neurons
            
            for elem in 0..<batchSize
            {
                for block in 0..<nbBlocks
                {
                    var sum1 = 0.0
                    for j1 in 0..<_size
                    {
                        let outPrev = neuronsPrev.get(
                            j1 + block * _size)!.v[elem].out
                        sum1 += exp(outPrev)
                    }
                    
                    for j1 in 0..<_size
                    {
                        let outPrev = neuronsPrev.get(
                            j1 + block * _size)!.v[elem].out
                        neurons.get(j1 + block * _size)!.v[elem].out =
                            exp(outPrev) / sum1
                    }
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
            
            let pSize: [UInt32] = [UInt32(_size)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            
            let command = MetalKernel.get.createCommand(
                "softmax1DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pSize, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBuffer(outs.metal, atIndex: 4)
            
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
            let nbBlocks = nbNeurons / _size
            let neuronsPrev = layerPrev.neurons
            
            for elem in 0..<batchSize
            {
                for block in 0..<nbBlocks
                {
                    var sum1: Double = 0.0
                    for j1 in 0..<_size
                    {
                        let outPrev = neuronsPrev.get(
                            j1 + block * _size)!.v[elem].out
                        sum1 += exp(outPrev)
                    }
                    
                    for j1 in 0..<_size
                    {
                        let outPrev1 = neuronsPrev.get(
                            j1 + block * _size)!.v[elem].out
                        let deltaCur1 = neurons.get(
                            j1 + block * _size)!.v[elem].delta
                        
                        var sum2: Double = 0.0
                        for j2 in 0..<_size
                        {
                            let deltaCur2 = neurons.get(
                                j2 + block * _size)!.v[elem].delta
                            let outPrev2 = neuronsPrev.get(
                                j2 + block * _size)!.v[elem].out
                            sum2 += exp(outPrev1 + outPrev2) * deltaCur2
                        }
                        
                        let tmp = exp(outPrev1) * deltaCur1 / sum1
                        if layerPrev.dirty
                        {
                            neuronsPrev.get(
                                j1 + block * _size)!.v[elem].delta =
                                -sum2 / (sum1 * sum1) + tmp
                        }
                        else
                        {
                            neuronsPrev.get(
                                j1 + block * _size)!.v[elem].delta +=
                                -sum2 / (sum1 * sum1) + tmp
                        }
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
            
            let pSize: [UInt32] = [UInt32(_size)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "softmax1DBackward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pSize, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
