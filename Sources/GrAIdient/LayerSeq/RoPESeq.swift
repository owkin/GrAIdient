//
// RoPESeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 16/06/2024.
//

import Foundation

///
/// Layer with a sequential shape neural structure.
///
/// This layer computes Rotary Positional Embedding (RoPE) of a sequential layer.
///
public class RoPESeq: LayerSeq
{
    /// Number of heads (groups) of neurons.
    let _nbHeads: Int
    
    /// List of positions to encode in the sequential axis.
    var _seqPositions: [Int]
    /// Whether positions in the sequential axis have just been set or not.
    var _dirtySeqPositions: Bool
    
    /// List of positions to encode in the sequential axis.
    public var seqPositions: [Int]
    {
        get {
            return _seqPositions
        }
        set {
            _seqPositions = newValue
            _dirtySeqPositions = true
        }
    }
    
    /// Rotary matrix.
    var _rotationMatrix: FloatBuffer! = nil
    
    private enum Keys: String, CodingKey
    {
        case nbHeads
        case seqPositions
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - seqPositions: List of positions to encode in the sequential axis.
    ///     - nbHeads: Number of heads (groups) of neurons.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq,
                seqPositions: [Int],
                nbHeads: Int,
                params: GrAI.Model.Params) throws
    {
        let nbNeurons = layerPrev.nbNeurons
        if nbNeurons % nbHeads != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(nbNeurons) " +
                "should be a multiple of nbHeads (\(nbHeads))."
            )
        }
        let size = nbNeurons / nbHeads
        if size % 2 != 0
        {
            throw LayerError.Init(message:
                "`size` (\(size) should be a multiple of 2."
            )
        }
        
        self._nbHeads = nbHeads
        self._seqPositions = seqPositions
        self._dirtySeqPositions = true
        
        super.init(layerPrev: layerPrev,
                   sequence: layerPrev.sequence,
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
        let values = try decoder.container(keyedBy: Keys.self)
        _seqPositions = try values.decode([Int].self, forKey: Keys.seqPositions)
        _nbHeads = try values.decode(Int.self, forKey: Keys.nbHeads)
        self._dirtySeqPositions = true
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
        try container.encode(seqPositions, forKey: Keys.seqPositions)
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
        let layerPrev = mapping[idPrev] as! LayerSeq
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = try! RoPESeq(
            layerPrev: layerPrev,
            seqPositions: seqPositions,
            nbHeads: _nbHeads,
            params: params
        )
        return layer
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We first clean the neurons' state (forward and backward).
    /// We do not clean weights and biases but must reset their delta (dependent on batch size) and
    /// momentum state.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        _rotationMatrix = nil
    }
    
    ///
    /// Initialize state resources in the CPU execution context.
    ///
    /// We initialize the neurons' state (forward and backward).
    ///
    public override func checkStateCPU(batchSize: Int) throws
    {
        if seqPositions.count != sequence
        {
            throw LayerError.Init(message:
                "`seqPositions` should contain \(sequence) elements but " +
                "it contains \(seqPositions) elements."
            )
        }
        try super.checkStateCPU(batchSize: batchSize)
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    /// We initialize the weights and biases' delta.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        if seqPositions.count != sequence
        {
            throw LayerError.Init(message:
                "`seqPositions` should contain \(sequence) elements but " +
                "it contains \(seqPositions) elements."
            )
        }
        
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if _rotationMatrix == nil || _dirtySeqPositions
        {
            let size = nbNeurons / _nbHeads
            let nbBlocks = size / 2
            
            _rotationMatrix = FloatBuffer(
                nbElems: sequence * size, deviceID: deviceID
            )
            
            let seqPositions32: [Int32] = seqPositions.map { Int32($0) }
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            
            let command = MetalKernel.get.createCommand(
                "createRoPESeqMatrix", deviceID: deviceID
            )
            command.setBytes(seqPositions32, atIndex: 0)
            command.setBytes(pNbHeads, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pSequence, atIndex: 3)
            command.setBuffer(_rotationMatrix.metal, atIndex: 4)
            
            command.dispatchThreads(
                width: nbBlocks,
                height: sequence
            )
            command.enqueue()
            
            _dirtySeqPositions = false
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for seq in 0..<sequence {
            for depth in 0..<nbNeurons
            {
                neurons.get(seq, depth)!.initGC(
                    batchSize: batchSize, nbGC: nbGC
                )
            }}
            
            let size = nbNeurons / _nbHeads
            let nbBlocks = size / 2
            let neuronsPrev = layerPrev.neurons!
            
            for batch in 0..<batchSize {
            for seq in 0..<sequence {
            for head in 0..<_nbHeads {
            for elem in 0..<nbGC
            {
                let position = seqPositions[seq]
                for block in 0..<nbBlocks
                {
                    let offset = 2 * block + head * size
                    
                    let theta = pow(
                        10000.0,
                        -2.0 * Double(block) / Double(size)
                    )
                    let mTheta = Double(position) * theta
                    let cosVal = cos(mTheta)
                    let sinVal = sin(mTheta)
                    
                    let in1 = neuronsPrev.get(seq, 0 + offset)!
                        .gc[batch][elem].out
                    let in2 = neuronsPrev.get(seq, 1 + offset)!
                        .gc[batch][elem].out
                    
                    let out1 = in1 * cosVal - in2 * sinVal
                    let out2 = in1 * sinVal + in2 * cosVal
                    
                    neurons.get(seq, 0 + offset)!.gc[batch][elem].out = out1
                    neurons.get(seq, 1 + offset)!.gc[batch][elem].out = out2
                }
            }}}}
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
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let size = nbNeurons / _nbHeads
            let nbBlocks = size / 2
            let neuronsPrev = layerPrev.neurons!
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence {
            for head in 0..<_nbHeads
            {
                let position = seqPositions[seq]
                for block in 0..<nbBlocks
                {
                    let offset = 2 * block + head * size
                    
                    let theta = pow(
                        10000.0,
                        -2.0 * Double(block) / Double(size)
                    )
                    let mTheta = Double(position) * theta
                    let cosVal = cos(mTheta)
                    let sinVal = sin(mTheta)
                    
                    let in1 = neuronsPrev.get(seq, 0 + offset)!.v[elem].out
                    let in2 = neuronsPrev.get(seq, 1 + offset)!.v[elem].out
                    
                    let out1 = in1 * cosVal - in2 * sinVal
                    let out2 = in1 * sinVal + in2 * cosVal
                    
                    neurons.get(seq, 0 + offset)!.v[elem].out = out1
                    neurons.get(seq, 1 + offset)!.v[elem].out = out2
                }
            }}}
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let size = nbNeurons / _nbHeads
            let nbBlocks = size / 2
            
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            
            let command = MetalKernel.get.createCommand(
                "RoPESeqForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(_rotationMatrix.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: _nbHeads * nbBlocks,
                height: batchSize * sequence
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            let size = nbNeurons / _nbHeads
            let nbBlocks = size / 2
            let neuronsPrev = layerPrev.neurons!
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence {
            for head in 0..<_nbHeads
            {
                let position = seqPositions[seq]
                for block in 0..<nbBlocks
                {
                    let offset = 2 * block + head * size
                    
                    let theta = pow(
                        10000.0,
                        -2.0 * Double(block) / Double(size)
                    )
                    let mTheta = Double(position) * theta
                    let cosVal = cos(mTheta)
                    let sinVal = sin(mTheta)
                    
                    let out1 = neurons.get(seq, 0 + offset)!.v[elem].delta
                    let out2 = neurons.get(seq, 1 + offset)!.v[elem].delta
                    
                    let in1 = out1 * cosVal + out2 * sinVal
                    let in2 = -out1 * sinVal + out2 * cosVal
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(seq, 0 + offset)!.v[elem].delta = in1
                        neuronsPrev.get(seq, 1 + offset)!.v[elem].delta = in2
                    }
                    else
                    {
                        neuronsPrev.get(seq, 0 + offset)!.v[elem].delta += in1
                        neuronsPrev.get(seq, 1 + offset)!.v[elem].delta += in2
                    }
                }
            }}}
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
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let size = nbNeurons / _nbHeads
            let nbBlocks = size / 2
            
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "RoPESeqSeqBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(_rotationMatrix.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: _nbHeads * nbBlocks,
                height: batchSize * sequence
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
