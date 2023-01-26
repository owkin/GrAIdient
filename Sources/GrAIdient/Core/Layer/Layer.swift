//
// Layer.swift
// GrAIdient
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation

/// Error occuring during the layer forward or backward propagation.
public enum LayerError: Error
{
    /// Data has not the correct dimensions.
    case DataSize
    /// Batch size is not coherent.
    case BatchSize
}

extension LayerError: CustomStringConvertible
{
    public var description: String
    {
        switch self
        {
        case .DataSize:
            return "The parameters do not have the expected number of elements."
        case .BatchSize:
            return "Batch size is not coherent with internal state."
        }
    }
}

/// A layer that is composed of multiple operations, especially an activation.
public protocol LayerWithActivation: Layer
{
    ///
    /// Extract main operation of this layer without the activation part.
    ///
    /// This API will create a new layer in the same context as this.
    ///
    /// - Parameter inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new layer. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    func extractActivation(inPlace: Bool) -> Layer
    ///
    /// Extract main operation of this layer without the activation part.
    ///
    /// - Parameter params: Contextual parameters linking to the model.
    ///
    /// - Returns: A new layer.
    ///
    func extractActivation(params: GrAI.Model.Params) -> Layer
    
    ///
    /// Create the activation part ot this layer.
    ///
    /// This API will create a new layer in the same context as this.
    ///
    /// - Returns: A new activation layer.
    ///
    func createActivation() -> Layer
    ///
    /// Create the activation part ot this layer.
    ///
    /// - Parameter params: Contextual parameters linking to the model.
    ///
    /// - Returns: A new activation layer.
    ///
    func createActivation(params: GrAI.Model.Params) -> Layer
}

/// A layer that needs image size information.
public protocol LayerResize: Layer
{
    ///
    /// Resize this layer.
    ///
    /// - Parameters:
    ///     - imageWidth: New size width.
    ///     - imageHeight: New size height.
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///
    /// - Returns: A new layer. When `inPlace` is false, `initKernel` is
    ///  necessary in order to recreate hard resources.
    ///
    func resize(
        imageWidth: Int,
        imageHeight: Int,
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool
    ) -> Layer
}

/// Abstract layer of a deep learning model.
open class Layer: Codable
{
    /// Identifier of the layer. Should be unique in a given model.
    public internal(set) var id: Int = 0
    /// Identifier of the previous layer in the model.
    public var idPrev: Int
    
    /// Link to the previous layer in the model.
    public weak var layerPrev: Layer? = nil
    
    /// Whether the gradient has been updated or not.
    public var dirty = true
    /// Whether to compute gradients of not.
    public var computeDelta = true
    
    /// Slight modification to use during gradient checking.
    public let Ɛ = 0.000001
    
    /// GPU device on which model is executed.
    public var deviceID: Int = 0
    /// Batch size of data.
    public var batchSize: Int = 0
    /// Running phase of a model: Training or Inference.
    public var phase: Phase? = nil
    
    /// Description of the Layer.
    var description: String
    {
        get {
            return String(describing: type(of: self))
        }
    }
    
    /// Whether backward pass should continue backward or not.
    public var mustComputeBackward: Bool
    {
        get {
            if let layerPrev = self.layerPrev, layerPrev.computeDelta
            {
                // Cannot test dirty: some layers are included in others.
                // The latter may be dirty when the first ones compute
                // their backward pass.
                return true
            }
            return false
        }
    }
    
    /// Number of different weigths for which we are estimating the gradient during Gradient Checking.
    public var nbGC: Int
    {
        get {
            return 0
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case id
        case idPrev
    }
    
    ///
    /// Create a layer.
    ///
    /// - Parameters:
    ///    - layerPrev: Previous layer that has been queued to the model.
    ///    - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer?, params: GrAI.Model.Params)
    {
        if let layerPrevTemp = layerPrev
        {
            idPrev = layerPrevTemp.id
        }
        else
        {
            idPrev = -1
        }
        
        id = params.context.curID
        params.context.update(layer: self, hidden: params.hidden)
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
        id = try container.decode(Int.self, forKey: .id)
        idPrev = try container.decode(Int.self, forKey: .idPrev)
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
    public func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        try container.encode(id, forKey: .id)
        try container.encode(idPrev, forKey: .idPrev)
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
    open func copy(
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// Find the `layerPrev` associated to the layer's `idPrev`.
    ///
    /// - Parameter layers: The potential layers where to find the layer's `idPrev`.
    ///
    open func initLinks(_ layers: [Layer])
    {
        for testLayer in layers
        {
            if testLayer.id == idPrev
            {
                layerPrev = testLayer
                break
            }
        }
    }
    
    ///
    /// Clean state resources in the CPU execution context.
    ///
    /// State resources are the resources that are dependent on the batch size.
    ///
    open func resetKernelCPU() {}
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// State resources are the resources that are dependent on the batch size.
    ///
    open func resetKernelGPU() {}
    
    ///
    /// Initialize hard resources in the CPU execution context.
    ///
    /// Hard resources are the resources that are not dependent on the batch size.
    /// Example: the batch normalization layer.
    /// Note that the weights are not initialized here, they have a dedicated API (initWeightsCPU).
    ///
    open func initKernelCPU() {}
    ///
    /// Initialize hard resources in the GPU execution context.
    ///
    /// Hard resources are the resources that are not dependent on the batch size.
    /// Example: the batch normalization layer.
    /// Note that the weights are not initialized here, they have a dedicated API (initWeightsGPU).
    ///
    open func initKernelGPU() {}
    
    ///
    /// Update the backward dirty flag for `layerPrev` instance.
    ///
    /// - Parameter dirty: The boolean value for the dirty flag.
    ///
    public func propagateDirty(_ dirty: Bool = false)
    {
        if let layerPrev = self.layerPrev
        {
            layerPrev.dirty = dirty
        }
    }
    
    ///
    /// Get every layers (a "graph") between the very first of the `Model` and this.
    ///
    /// - Parameter layerPrev: The different layers found in the "graph".
    ///
    public func getGraph(_ layerPrev: inout [Layer])
    {
        layerPrev.append(self)
        if let layerPrevTmp = self.layerPrev
        {
            layerPrevTmp.getGraph(&layerPrev)
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open func forwardGCCPU() throws {}
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open func forwardGCGPU() throws {}
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open func forwardCPU() throws {}
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open func forwardGPU() throws {}
    
    /// Apply the backward pass in the CPU execution context.
    open func backwardCPU() {}
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    open func backwardGPU() throws {}
}
