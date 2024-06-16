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
    var _rotaryMatrix: FloatBuffer! = nil
    
    private enum Keys: String, CodingKey
    {
        case seqPositions
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - seqPositions: List of positions to encode in the sequential axis.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq,
                seqPositions: [Int],
                params: GrAI.Model.Params) throws
    {
        self._seqPositions = seqPositions
        self._dirtySeqPositions = true
        
        super.init(layerPrev: layerPrev,
                   sequence: layerPrev.sequence,
                   nbNeurons: layerPrev.nbNeurons,
                   params: params)
        
        if layerPrev.nbNeurons % 2 != 0
        {
            throw LayerError.Init(message:
                "`nbNeurons` (\(nbNeurons) should be a multiple of 2."
            )
        }
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
        _rotaryMatrix = nil
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
        
        if _rotaryMatrix == nil || _dirtySeqPositions
        {
            let nbBlocks = nbNeurons / 2
            _rotaryMatrix = FloatBuffer(nbElems:
                4 * nbBlocks, deviceID: deviceID
            )
            
            // TODO: Update rotaryMatrix.
            
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
            
            let nbBlocks = nbNeurons / 2
            let neuronsPrev = layerPrev.neurons!
            
            for batch in 0..<batchSize {
            for seq in 0..<sequence {
            for elem in 0..<nbGC
            {
                let position = seqPositions[seq]
                for block in 0..<nbBlocks
                {
                    let theta = pow(
                        10000.0,
                        -2.0 * Double(block) / Double(nbNeurons)
                    )
                    let mTheta = Double(position) * theta
                    let cosVal = cos(mTheta)
                    let sinVal = sin(mTheta)
                    
                    let in1 = neuronsPrev.get(seq, 0 + 2 * block)!
                        .gc[batch][elem].out
                    let in2 = neuronsPrev.get(seq, 1 + 2 * block)!
                        .gc[batch][elem].out
                    
                    let out1 = in1 * cosVal - in2 * sinVal
                    let out2 = in1 * sinVal + in2 * cosVal
                    
                    neurons.get(seq, 0 + 2 * block)!.gc[batch][elem].out = out1
                    neurons.get(seq, 1 + 2 * block)!.gc[batch][elem].out = out2
                }
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
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbBlocks = nbNeurons / 2
            let neuronsPrev = layerPrev.neurons!
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence 
            {
                let position = seqPositions[seq]
                for block in 0..<nbBlocks
                {
                    let theta = pow(
                        10000.0,
                        -2.0 * Double(block) / Double(nbNeurons)
                    )
                    let mTheta = Double(position) * theta
                    let cosVal = cos(mTheta)
                    let sinVal = sin(mTheta)
                    
                    let in1 = neuronsPrev.get(seq, 0 + 2 * block)!.v[elem].out
                    let in2 = neuronsPrev.get(seq, 1 + 2 * block)!.v[elem].out
                    
                    let out1 = in1 * cosVal - in2 * sinVal
                    let out2 = in1 * sinVal + in2 * cosVal
                    
                    neurons.get(seq, 0 + 2 * block)!.v[elem].out = out1
                    neurons.get(seq, 1 + 2 * block)!.v[elem].out = out2
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
        /*if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            
            let kernel = (nbNeurons / _nbHeads) % 4 == 0 ?
                "softmaxSeq4Forward" : "softmaxSeqForward"
            let coeff = (nbNeurons / _nbHeads) % 4 == 0 ? 4 : 1
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbHeads, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pSequence, atIndex: 4)
            command.setBuffer(outs.metal, atIndex: 5)
            
            command.dispatchThreads(
                width: nbNeurons / coeff,
                height: batchSize * sequence
            )
            command.enqueue()
        }*/
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            let nbBlocks = nbNeurons / 2
            let neuronsPrev = layerPrev.neurons!
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                let position = seqPositions[seq]
                for block in 0..<nbBlocks
                {
                    let theta = pow(
                        10000.0,
                        -2.0 * Double(block) / Double(nbNeurons)
                    )
                    let mTheta = Double(position) * theta
                    let cosVal = cos(mTheta)
                    let sinVal = sin(mTheta)
                    
                    let out1 = neuronsPrev.get(seq, 0 + 2 * block)!
                        .v[elem].delta
                    let out2 = neuronsPrev.get(seq, 1 + 2 * block)!
                        .v[elem].delta
                    
                    let in1 = out1 * cosVal + out2 * sinVal
                    let in2 = -out1 * sinVal + out2 * cosVal
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(seq, 0 + 2 * block)!.v[elem].delta = in1
                        neuronsPrev.get(seq, 1 + 2 * block)!.v[elem].delta = in2
                    }
                    else
                    {
                        neuronsPrev.get(seq, 0 + 2 * block)!.v[elem].delta += 
                            in1
                        neuronsPrev.get(seq, 1 + 2 * block)!.v[elem].delta += 
                            in2
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
        /*if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbHeads: [UInt32] = [UInt32(_nbHeads)]
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let kernel = (nbNeurons / _nbHeads) % 4 == 0 ?
                "softmaxSeq4Backward" : "softmaxSeqBackward"
            let coeff = (nbNeurons / _nbHeads) % 4 == 0 ? 4 : 1
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(outs.metal, atIndex: 0)
            command.setBuffer(delta.metal, atIndex: 1)
            command.setBytes(pNbHeads, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: nbNeurons / coeff,
                height: batchSize * sequence
            )
            command.enqueue()
            
            propagateDirty()
        }*/
    }
}
