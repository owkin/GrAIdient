//
// BCESigmoid2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 07/07/2023.
//

import Foundation

///
/// Output layer with a 2D shape neural structure and a loss that computes binary cross entropy on top
/// of a sigmoid activation.
///
public class BCESigmoid2D: LayerOutput2D
{
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
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = try! BCESigmoid2D(layerPrev: layerPrev, params: params)
        layer.coeff = self.coeff
        
        return layer
    }
    
    ///
    /// Estimate the gradients of weights thanks to Gradient Checking.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - format: TThe data format.
    /// - Returns: The estimated gradients of weights.
    ///
    public func collectGradientsApprox<T: BinaryFloatingPoint>(
        _ groundTruth: [T],
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int,
        format: ImageFormat) throws -> [T]
    {
        var gradients = [T]()
        let nbGradients = neurons.first!.get(0)!.nbGC / 2
        for elem in 0..<nbGradients
        {
            let loss1 = try getLossGC(
                groundTruth,
                batchSize: batchSize,
                nbChannels: nbChannels, height: height, width: width,
                elem: 2 * elem,
                format: format
            )
            let loss2 = try getLossGC(
                groundTruth,
                batchSize: batchSize,
                nbChannels: nbChannels, height: height, width: width,
                elem: 2 * elem + 1,
                format: format
            )
            
            let gradient = (loss1 - loss2) / T(2 * Ɛ)
            gradients.append(gradient)
        }
        return gradients
    }
    
    ///
    /// Get the loss consecutive of a modified weights during the Gradient Checking process.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - elem: The modified weight for which we collect the resulting loss.
    ///     - format: The data format.
    /// - Returns: The loss value.
    ///
    func getLossGC<T: BinaryFloatingPoint>(
        _ groundTruth: [T],
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int,
        elem: Int,
        format: ImageFormat) throws -> T
    {
        try checkGroundTruthCPU(
            groundTruth,
            batchSize: batchSize,
            nbChannels: nbChannels, height: height, width: width
        )
        
        var losses = [T](repeating: 0.0, count: batchSize)
        switch format
        {
        case .RGB:
            for batch in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let offset = j + (batch * height + i) * width
                    
                    let out = neurons[depth].get(i, j)!.gc[batch][elem].out
                    let gt = groundTruth[nbChannels * offset + depth]
                    var value: T
                    
                    if out > 0
                    {
                        value = T(1 - gt) * T(out)
                        value += T(log(1 + exp(-out)))
                    }
                    else
                    {
                        value = -T(out) * T(gt)
                        value += T(log(exp(out) + 1))
                    }
                    
                    losses[batch] += value
                }}
            }}
        case .Neuron:
            for batch in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                let offsetStart = (depth + nbChannels * batch) * height
                
                for i in 0..<height {
                for j in 0..<width
                {
                    let offset = j + (offsetStart + i) * width
                    
                    let out = neurons[depth].get(i, j)!.gc[batch][elem].out
                    let gt = groundTruth[offset]
                    var value: T
                    
                    if out > 0
                    {
                        value = T(1 - gt) * T(out)
                        value += T(log(1 + exp(-out)))
                    }
                    else
                    {
                        value = -T(out) * T(gt)
                        value += T(log(exp(out) + 1))
                    }
                    
                    losses[batch] += value
                }}
            }}
        }
        return T(coeff) * losses.reduce(0, +) /
               T(batchSize * nbChannels * height * width)
    }
    
    ///
    /// Get loss in the CPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - format: The data format.
    /// - Returns: The loss value.
    ///
    public func getLossCPU<T: BinaryFloatingPoint>(
        _ groundTruth: [T],
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int,
        format: ImageFormat) throws -> T
    {
        try checkGroundTruthCPU(
            groundTruth,
            batchSize: batchSize,
            nbChannels: nbChannels, height: height, width: width
        )
        
        var losses = [T](repeating: 0.0, count: batchSize)
        switch format
        {
        case .RGB:
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let offset = j + (elem * height + i) * width
                    
                    let out = neurons[depth].get(i, j)!.v[elem].out
                    let gt = groundTruth[nbChannels * offset + depth]
                    var value: T
                    
                    if out > 0
                    {
                        value = T(1 - gt) * T(out)
                        value += T(log(1 + exp(-out)))
                    }
                    else
                    {
                        value = -T(out) * T(gt)
                        value += T(log(exp(out) + 1))
                    }
                    
                    losses[elem] += value
                }}
            }}
        case .Neuron:
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                let offsetStart = (depth + nbChannels * elem) * height
                
                for i in 0..<height {
                for j in 0..<width
                {
                    let offset = j + (offsetStart + i) * width
                    
                    let out = neurons[depth].get(i, j)!.v[elem].out
                    let gt = groundTruth[offset]
                    var value: T
                    
                    if out > 0
                    {
                        value = T(1 - gt) * T(out)
                        value += T(log(1 + exp(-out)))
                    }
                    else
                    {
                        value = -T(out) * T(gt)
                        value += T(log(exp(out) + 1))
                    }
                    
                    losses[elem] += value
                }}
            }}
        }
        return T(coeff) * losses.reduce(0, +) /
               T(batchSize * nbChannels * height * width)
    }
    
    ///
    /// Get loss in the GPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - format: The data format.
    /// - Returns: The loss value.
    ///
    public func getLossGPU<T: BinaryFloatingPoint>(
        _ groundTruth: [T],
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int,
        format: ImageFormat) throws -> T
    {
        try checkGroundTruthGPU(
            groundTruth,
            batchSize: batchSize,
            nbChannels: nbChannels, height: height, width: width,
            format: format
        )
        return try T(getLossGPU(
            self.groundTruth,
            batchSize: batchSize,
            nbChannels: nbChannels, height: height, width: width
        ))
    }
    
    ///
    /// Get loss in the GPU execution context.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    /// - Returns: The loss value.
    ///
    public func getLossGPU(
        _ groundTruth: MetalBuffer<UInt16>,
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int) throws -> Float
    {
        try checkGroundTruthGPU(
            groundTruth,
            batchSize: batchSize,
            nbChannels: nbChannels, height: height, width: width
        )
        try checkLossGPU(batchSize: batchSize)
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        
        let command = MetalKernel.get.createCommand(
            "BCESigmoid2DLoss", deviceID: deviceID
        )
        command.setBuffer(outs.metal(), atIndex: 0)
        command.setBuffer(groundTruth.metal(), atIndex: 1)
        command.setBytes(pNbChannels, atIndex: 2)
        command.setBytes(pDimensions, atIndex: 3)
        command.setBytes(pNbBatch, atIndex: 4)
        command.setBuffer(loss.metal(), atIndex: 5)
        
        command.dispatchThreads(batchSize)
        command.enqueue()
        
        var loss: Float = 0.0
        let lossPtr = getHalfBuffer(self.loss).array
        for i in 0..<batchSize
        {
            loss += lossPtr[i]
        }
        return Float(coeff) * loss /
               Float(batchSize * nbChannels * height * width)
    }
    
    ///
    /// Compute the derivative of the loss in the CPU execution context.
    ///
    /// This function is necessary to initialize the backward pass !
    /// In a way, it plays a similar role as the `setData` of the first layer.
    ///
    /// The `setData` API sets data to the first layer to initialize the forward pass.
    /// Here we use the `groundTruth` to initialize the backward pass.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - format: The data format.
    ///
    public func lossDerivativeCPU<T: BinaryFloatingPoint>(
        _ groundTruth: [T],
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int,
        format: ImageFormat) throws
    {
        try checkGroundTruthCPU(
            groundTruth,
            batchSize: batchSize,
            nbChannels: nbChannels, height: height, width: width
        )
        
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            switch format
            {
            case .RGB:
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    for i in 0..<height {
                    for j in 0..<width
                    {
                        let offset = j + (elem * height + i) * width
                        
                        let out = neurons[depth].get(i, j)!.v[elem].out
                        let gt = groundTruth[nbChannels * offset + depth]
                        let value: Double
                        
                        if out >= 0
                        {
                            value = 1.0 / (1.0 + exp(-out))
                        }
                        else
                        {
                            value = exp(out) / (1.0 + exp(out))
                        }
                        
                        if layerPrev.dirty
                        {
                            neuronsPrev[depth].get(i, j)!.v[elem].delta =
                                coeff * (value - Double(gt)) /
                                Double(batchSize * nbChannels * height * width)
                        }
                        else
                        {
                            neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                                coeff * (value - Double(gt)) /
                                Double(batchSize * nbChannels * height * width)
                        }
                    }}
                }}
            case .Neuron:
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    let offsetStart = (depth + nbChannels * elem) * height
                    
                    for i in 0..<height {
                    for j in 0..<width
                    {
                        let offset = j + (offsetStart + i) * width
                        
                        let out = neurons[depth].get(i, j)!.v[elem].out
                        let gt = groundTruth[offset]
                        let value: Double
                        
                        if out >= 0
                        {
                            value = 1.0 / (1.0 + exp(-out))
                        }
                        else
                        {
                            value = exp(out) / (1.0 + exp(out))
                        }
                        
                        if layerPrev.dirty
                        {
                            neuronsPrev[depth].get(i, j)!.v[elem].delta =
                                coeff * (value - Double(gt)) /
                                Double(batchSize * nbChannels * height * width)
                        }
                        else
                        {
                            neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                                coeff * (value - Double(gt)) /
                                Double(batchSize * nbChannels * height * width)
                        }
                    }}
                }}
            }
            propagateDirty()
        }
    }
    
    ///
    /// Compute the derivative of the loss in the GPU execution context.
    ///
    /// This function is necessary to initialize the backward pass !
    /// In a way, it plays a similar role as the `setData` of the first layer.
    ///
    /// The `setData` API sets data to the first layer to initialize the forward pass.
    /// Here we use the `groundTruth` to initialize the backward pass.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - format: The data format.
    ///
    public func lossDerivativeGPU<T: BinaryFloatingPoint>(
        _ groundTruth: [T],
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int,
        format: ImageFormat) throws
    {
        try checkGroundTruthGPU(
            groundTruth,
            batchSize: batchSize,
            nbChannels: nbChannels, height: height, width: width,
            format: format
        )
        try lossDerivativeGPU(
            self.groundTruth,
            batchSize: batchSize,
            nbChannels: nbChannels, height: height, width: width
        )
    }
    
    ///
    /// Compute the derivative of the loss in the GPU execution context.
    ///
    /// This function is necessary to initialize the backward pass !
    /// In a way, it plays a similar role as the `setData` of the first layer.
    ///
    /// The `setData` API sets data to the first layer to initialize the forward pass.
    /// Here we use the `groundTruth` to initialize the backward pass.
    ///
    /// Throw an error if data size is incoherent.
    ///
    /// - Parameters:
    ///     -  groundTruth: The ground truth.
    ///     - batchSize: The batch size of data.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///
    public func lossDerivativeGPU(
        _ groundTruth: MetalBuffer<UInt16>,
        batchSize: Int,
        nbChannels: Int, height: Int, width: Int) throws
    {
        try checkGroundTruthGPU(
            groundTruth,
            batchSize: batchSize,
            nbChannels: nbChannels, height: height, width: width
        )
        
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pCoeff: [Float] = [Float(coeff)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "BCESigmoid2DLossDerivative", deviceID: deviceID
            )
            command.setBuffer(outs.metal(), atIndex: 0)
            command.setBuffer(groundTruth.metal(), atIndex: 1)
            command.setBytes(pNbChannels, atIndex: 2)
            command.setBytes(pDimensions, atIndex: 3)
            command.setBytes(pCoeff, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal(), atIndex: 7)
            
            command.dispatchThreads(
                width: nbChannels * width,
                height: batchSize * height
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
