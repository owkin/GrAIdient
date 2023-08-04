//
// Softmax1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 28/01/2023.
//

import Foundation

///
/// Layer with a 1D shape neural structure.
///
/// This layer computes the Softmax function per block of neurons of a 1D layer.
///
public class Softmax1D: Layer1D
{
    let _nbHeads: Int
    
    private enum Keys: String, CodingKey
    {
        case nbHeads
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - nbHeads: Number of heads (groups) of neurons.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer1D,
                nbHeads: Int,
                params: GrAI.Model.Params) throws
    {
        let nbNeurons = layerPrev.nbNeurons
        if nbNeurons % nbHeads != 0
        {
            throw LayerError.Init(
                message: "`nbNeurons` (\(nbNeurons) " +
                         "should be a multiple of nbHeads (\(nbHeads))."
            )
        }
        
        _nbHeads = nbHeads
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
        _nbHeads = try values.decode(Int.self, forKey: Keys.nbHeads)
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
        try container.encode(_nbHeads, forKey: Keys.nbHeads)
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
            
        let layer = try! Softmax1D(
            layerPrev: layerPrev,
            nbHeads: _nbHeads,
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
            
            let size = nbNeurons / _nbHeads
            let neuronsPrev = layerPrev.neurons
            
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for head in 0..<_nbHeads
                {
                    var cMax = neuronsPrev
                        .get(0 + head * size)!.gc[batch][elem].out
                    for j in 0..<size
                    {
                        let outPrev = neuronsPrev
                            .get(j + head * size)!.gc[batch][elem].out
                        if outPrev > cMax
                        {
                            cMax = outPrev
                        }
                    }
                    
                    var sum1 = 0.0
                    for j in 0..<size
                    {
                        let outPrev = neuronsPrev
                            .get(j + head * size)!.gc[batch][elem].out
                        sum1 += exp(outPrev - cMax)
                    }
                    
                    for j in 0..<size
                    {
                        let outPrev = neuronsPrev
                            .get(j + head * size)!.gc[batch][elem].out
                        neurons.get(j + head * size)!.gc[batch][elem].out =
                            exp(outPrev - cMax) / sum1
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
            
            let size = nbNeurons / _nbHeads
            let neuronsPrev = layerPrev.neurons
            
            for elem in 0..<batchSize {
            for head in 0..<_nbHeads
            {
                var cMax = neuronsPrev.get(0 + head * size)!.v[elem].out
                for j in 0..<size
                {
                    let outPrev = neuronsPrev.get(j + head * size)!.v[elem].out
                    if outPrev > cMax
                    {
                        cMax = outPrev
                    }
                }
                
                var sum1 = 0.0
                for j in 0..<size
                {
                    let outPrev = neuronsPrev.get(j + head * size)!.v[elem].out
                    sum1 += exp(outPrev - cMax)
                }
                
                for j in 0..<size
                {
                    let outPrev = neuronsPrev.get(j + head * size)!.v[elem].out
                    neurons.get(j + head * size)!.v[elem].out =
                        exp(outPrev - cMax) / sum1
                }
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
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            
            let command = MetalKernel.get.createCommand(
                "softmax1DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbHeads, atIndex: 1)
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
            let size = nbNeurons / _nbHeads
            let neuronsPrev = layerPrev.neurons
            
            for elem in 0..<batchSize {
            for head in 0..<_nbHeads
            {
                for j in 0..<size
                {
                    let outCur = neurons.get(j + head * size)!.v[elem].out
                    let deltaCur = neurons.get(j + head * size)!.v[elem].delta
                    
                    var sum1: Double = 0.0
                    for j1 in 0..<size
                    {
                        let deltaCur1 = neurons
                            .get(j1 + head * size)!.v[elem].delta
                        let outCur1 = neurons.get(j1 + head * size)!.v[elem].out
                        sum1 += outCur1 * deltaCur1
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(j + head * size)!.v[elem].delta =
                            outCur * (deltaCur - sum1)
                    }
                    else
                    {
                        neuronsPrev.get(j + head * size)!.v[elem].delta +=
                            outCur * (deltaCur - sum1)
                    }
                }
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
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "softmax1DBackward", deviceID: deviceID
            )
            command.setBuffer(outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
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
