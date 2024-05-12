//
// LayerOutput1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// Loss layer of a model with a 1D shape neural structure.
open class LayerOutput1D: Layer1D
{
    /// Coefficient to be applied to the loss computation.
    public var coeff: Double = 1.0
    
    ///
    /// Ground truth buffer in the GPU execution context.
    /// Shape ~ (batch, nbNeurons).
    ///
    public internal(set) var groundTruth: FloatBuffer! = nil
    
    ///
    /// Loss buffer in the GPU execution context.
    /// Shape ~ (batch,).
    ///
    public internal(set) var loss: FloatBuffer! = nil
    
    private enum Keys: String, CodingKey
    {
        case coeff
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - params: Contextual parameters linking to the model.
    ///
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
        let container = try decoder.container(keyedBy: Keys.self)
        let coeff = try container.decode(Float.self, forKey: .coeff)
        self.coeff = Double(coeff)
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
        try container.encode(Float(coeff), forKey: .coeff)
        try super.encode(to: encoder)
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We clean the neurons' state (forward and backward).
    ///
    open override func resetKernelGPU()
    {
        super.resetKernelGPU()
        groundTruth = nil
        loss = nil
    }
    
    ///
    /// Check and setup ground truth in the CPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    ///
    public func checkGroundTruthCPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]],
        batchSize: Int,
        nbNeurons: Int) throws
    {
        if groundTruth.count != batchSize ||
           groundTruth.first!.count != nbNeurons
        {
            throw LayerError.DataSize
        }
        if batchSize != self.batchSize ||
           nbNeurons != self.nbNeurons
        {
            throw LayerError.DataSize
        }
        if batchSize <= 0 || batchSize > neurons.get(0)!.v.count
        {
            throw LayerError.BatchSize
        }
    }
    
    ///
    /// Check and setup ground truth in the GPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    ///
    public func checkGroundTruthGPU<T: BinaryFloatingPoint>(
        _ groundTruth: [[T]],
        batchSize: Int,
        nbNeurons: Int) throws
    {
        if groundTruth.count != batchSize ||
           groundTruth.first!.count != nbNeurons
        {
            throw LayerError.DataSize
        }
        if batchSize != self.batchSize ||
           nbNeurons != self.nbNeurons
        {
            throw LayerError.DataSize
        }
        
        if self.groundTruth == nil
        {
            self.groundTruth = FloatBuffer(
                nbElems: batchSize * nbNeurons,
                deviceID: deviceID,
                shared: true
            )
        }
        else if batchSize <= 0 ||
                batchSize * nbNeurons > self.groundTruth.nbElems
        {
            throw LayerError.BatchSize
        }
        
        var buffer = [Float](repeating: 0.0, count: batchSize * nbNeurons)
        for (i, dataI) in groundTruth.enumerated()
        {
            if dataI.count != nbNeurons
            {
                throw LayerError.DataSize
            }
            for (j, dataIJ) in dataI.enumerated()
            {
                buffer[j + i * nbNeurons] = Float(dataIJ)
            }
        }
        self.groundTruth.initialize(array: &buffer)
    }
    
    ///
    /// Check and setup ground truth in the GPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    ///
    public func checkGroundTruthGPU(
        _ groundTruth: FloatBuffer,
        batchSize: Int,
        nbNeurons: Int) throws
    {
        if batchSize <= 0 ||
           batchSize * nbNeurons > groundTruth.nbElems
        {
            throw LayerError.BatchSize
        }
        if batchSize != self.batchSize ||
           nbNeurons != self.nbNeurons
        {
            throw LayerError.DataSize
        }
    }
    
    ///
    /// Setup loss state  in the GPU execution context.
    ///
    /// Throw an error if batch size or ground truth are incoherent.
    ///
    /// - Parameter batchSize: The batch size of data.
    ///
    public func checkLossGPU(batchSize: Int) throws
    {
        if loss == nil
        {
            loss = FloatBuffer(
                nbElems: batchSize, deviceID: deviceID, shared: true
            )
        }
        else if batchSize > loss.nbElems
        {
            throw LayerError.BatchSize
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for depth in 0..<nbNeurons
            {
                neurons.get(depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for j in 0..<nbNeurons
            {
                neurons.get(j)!.gc[batch][elem].out =
                    neuronsPrev.get(j)!.gc[batch][elem].out
            }}}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardGCGPU() throws
    {
        try forwardGCCPU()
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for j in 0..<nbNeurons
            {
                neurons.get(j)!.v[elem].out = neuronsPrev.get(j)!.v[elem].out
            }}
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer1D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let nbElems = outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let kernel = nbElems % 4 == 0 ? "sum14" : "sum1"
            let coeff = nbElems % 4 == 0 ? 4 : 1
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems / coeff)
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    open override func backwardCPU()
    {
        // Note that backward is not called except when it is
        // an intermediate layer.
        // Model.backward is only called on non dirty layers.
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbNeurons
            {
                let delta = neurons.get(depth)!.v[elem].delta
                if layerPrev.dirty
                {
                    neuronsPrev.get(depth)!.v[elem].delta = delta
                }
                else
                {
                    neuronsPrev.get(depth)!.v[elem].delta += delta
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
    open override func backwardGPU() throws
    {
        // Note that backward is not called except when it is
        // an intermediate layer.
        // Model.backward is only called on non dirty layers.
        if let layerPrev = self.layerPrev as? Layer1D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let kernel: String
            let coeff = nbElems % 4 == 0 ? 4 : 1
            if layerPrev.dirty
            {
                kernel = nbElems % 4 == 0 ? "sum14" : "sum1"
            }
            else
            {
                kernel = nbElems % 4 == 0 ? "sum24" : "sum2"
            }
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(layerPrev.delta.metal, atIndex: 2)
            
            command.dispatchThreads(nbElems / coeff)
            command.enqueue()
            
            propagateDirty()
        }
    }
}
