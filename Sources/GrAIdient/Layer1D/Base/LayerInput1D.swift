//
// LayerInput1D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// Input layer of a model.
open class LayerInput1D: Layer1D
{
    ///
    /// Whether we must call the forward for this layer or not.
    ///
    /// When this layer is the first of a model, user is supposed to pass data with an explicit API.
    /// When this layer is an intermediate layer of the model, user expects to rely to the forward API:
    /// in that case, set `computeForward` to true.
    ///
    public var computeForward = false
    
    /// Number of different weigths for which we are estimating the gradient during Gradient Checking.
    public override var nbGC: Int
    {
        get {
            return 0
        }
    }
    
    ///
    /// Find the `layerPrev` associated to the layer's `idPrev`.
    ///
    /// - Parameter layers: The potential layers where to find the layer's `idPrev`.
    ///
    public override func initLinks(_ layers: [Layer])
    {
        super.initLinks(layers)
        if layerPrev != nil
        {
            computeDelta = true
        }
    }
    
    ///
    /// Check and setup input in the CPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The input data.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    ///
    public func checkInputCPU<T: BinaryFloatingPoint>(
        _ data: [[T]],
        batchSize: Int,
        nbNeurons: Int) throws
    {
        if data.count != batchSize || data.first!.count != nbNeurons
        {
            throw LayerError.DataSize
        }
        if nbNeurons != self.nbNeurons
        {
            throw LayerError.DataSize
        }
        try checkStateCPU(batchSize: batchSize)
        
        for (elem, sample) in data.enumerated()
        {
            if sample.count != nbNeurons
            {
                throw LayerError.DataSize
            }
            
            for (i, feature) in sample.enumerated() {
            if let neuron = neurons.get(i)
            {
                neuron.v[elem].out = Double(feature)
            }}
        }
    }
    
    ///
    /// Check and setup input in the GPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The input data.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    ///
    public func checkInputGPU<T: BinaryFloatingPoint>(
        _ data: [[T]],
        batchSize: Int,
        nbNeurons: Int) throws
    {
        if data.count != batchSize || data.first!.count != nbNeurons
        {
            throw LayerError.DataSize
        }
        if nbNeurons != self.nbNeurons
        {
            throw LayerError.DataSize
        }
        try checkStateForwardGPU(batchSize: batchSize)
        
        // Wait for previous loop to end to avoid race condition with
        // download in the following example:
        // FullyConnected.backwardWeightsGPU accesses layerPrev.outs.
        _ = outs.download()
        
        var buffer = [Float](repeating: 0.0, count: batchSize * nbNeurons)
        for elem in 0..<batchSize
        {
            for depth in 0..<nbNeurons
            {
                let offset = depth + nbNeurons * elem
                buffer[offset] = Float(data[elem][depth])
            }
        }
        outs.initialize(array: &buffer)
    }
    
    ///
    /// Check and setup input in the GPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The input data.
    ///     - batchSize: The batch size of data.
    ///     - nbNeurons: Number of neurons.
    ///
    public func checkInputGPU(
        _ data: FloatBuffer,
        batchSize: Int,
        nbNeurons: Int) throws
    {
        if batchSize * nbNeurons > data.nbElems
        {
            throw LayerError.DataSize
        }
        if nbNeurons != self.nbNeurons
        {
            throw LayerError.DataSize
        }
        try checkStateForwardGPU(batchSize: batchSize)
        outs = data
    }
}
