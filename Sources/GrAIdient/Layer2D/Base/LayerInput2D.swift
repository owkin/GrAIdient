//
// LayerInput2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// Input layer of a model.
open class LayerInput2D: Layer2D
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
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - format: The data format.
    ///
    public func checkInputCPU<T: BinaryFloatingPoint>(
        _ data: [T],
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int,
        format: ImageFormat) throws
    {
        if data.count != batchSize * nbChannels * height * width
        {
            throw LayerError.DataSize
        }
        if nbChannels != self.nbChannels ||
           height != self.height ||
           width != self.width
        {
            throw LayerError.DataSize
        }
        try checkStateCPU(batchSize: batchSize)
        
        switch format
        {
        case .RGB:
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let offset = j + (elem * height + i) * width
                    for depth in 0..<nbChannels
                    {
                        neurons[depth].get(i, j)!.v[elem].out =
                            Double(data[nbChannels * offset + depth])
                    }
                }}
            }
        case .Neuron:
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    for depth in 0..<nbChannels
                    {
                        let offsetStart = (depth + nbChannels * elem) * height
                        let offset = j + (offsetStart + i) * width
                        
                        neurons[depth].get(i, j)!.v[elem].out =
                            Double(data[offset])
                    }
                }}
            }
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
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - format: The data format.
    ///
    public func checkInputGPU<T: BinaryFloatingPoint>(
        _ data: [T],
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int,
        format: ImageFormat) throws
    {
        if data.count != batchSize * nbChannels * height * width
        {
            throw LayerError.DataSize
        }
        if nbChannels != self.nbChannels ||
           height != self.height ||
           width != self.width
        {
            throw LayerError.DataSize
        }
        try checkStateForwardGPU(batchSize: batchSize)
        
        // Wait for previous loop to end to avoid race condition with
        // didModifyRange in the following example:
        // Convolution.backwardWeightsGPU accesses layerPrev.outs.
        MetalKernel.get.download([outs])
        
        var buffer = [Float](
            repeating: 0.0, count: batchSize * nbChannels * height * width
        )
        
        switch format
        {
        case .RGB:
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let offsetGet = j + (elem * height + i) * width
                    for depth in 0..<nbChannels
                    {
                        let offsetStartSet =
                            (depth + nbChannels * elem) * height
                        let offsetSet = j + (offsetStartSet + i) * width
                        
                        buffer[offsetSet] =
                            Float(data[nbChannels * offsetGet + depth])
                    }
                }}
            }
        case .Neuron:
            for elem in 0..<batchSize
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    for depth in 0..<nbChannels
                    {
                        let offsetStart = (depth + nbChannels * elem) * height
                        let offset = j + (offsetStart + i) * width
                        
                        buffer[offset] = Float(data[offset])
                    }
                }}
            }
        }
        
        setupHalfBuffer(
            array: &buffer,
            out: self.outs,
            start: 0,
            nbElems: batchSize * nbChannels * height * width,
            deviceID: deviceID
        )
    }
    
    ///
    /// Check and setup input in the GPU execution context.
    ///
    /// Throw an error if data size is not coherent.
    ///
    /// - Parameters:
    ///     - data: The input data.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - format: The data format.
    ///
    public func checkInputGPU(
        _ data: MetalPrivateBuffer<UInt16>,
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int) throws
    {
        if batchSize * nbChannels * height * width > data.nbElems
        {
            throw LayerError.DataSize
        }
        if nbChannels != self.nbChannels ||
           height != self.height ||
           width != self.width
        {
            throw LayerError.DataSize
        }
        try checkStateForwardGPU(batchSize: batchSize)
        outs = data
    }
}
