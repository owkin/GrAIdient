//
// Serialization.swift
// GrAIdient
//
// Created by Jean-François Reboud on 05/10/2022.
//

/// Error occuring while decoding from the disk.
public enum SerializationError: Error
{
    /// Trying to decode a class which has not been registered.
    case TypeNotFound
}

extension SerializationError: CustomStringConvertible
{
    public var description: String
    {
        switch self
        {
        case .TypeNotFound:
            return "Type has not been registered anywhere: " +
                   "use GrAI.Model.*.append APIs."
        }
    }
}

/// Registry of layer types.
let LAYER_REGISTRY: [String: Codable.Type] = buildRegistry(
[
    Activation1D.self,
    Activation2D.self,
    AdaIN.self,
    AdaptiveAvgPool2D.self,
    AvgPool2D.self,
    AvgPoolSeq.self,
    BN2D.self,
    Concat1D.self,
    Concat2D.self,
    Concat1Seq.self,
    Concat2Seq.self,
    Constant1D.self,
    Constant2D.self,
    Constant12Seq.self,
    Constant2Seq.self,
    Convolution2D.self,
    Crop2D.self,
    Deconvolution2D.self,
    DecorrelateRGB.self,
    DotProduct1D.self,
    FTFrequences2D.self,
    FullyConnected.self,
    FullyConnectedPatch.self,
    FullyConnectedSeq.self,
    Input1D.self,
    Input2D.self,
    IRDFT2RGB.self,
    InstanceNorm2D.self,
    LayerNormSeq.self,
    LinearError1D.self,
    LinearScale2D.self,
    MaxPool2D.self,
    Normalize12D.self,
    Normalize122D.self,
    MSE1D.self,
    MSE2D.self,
    Multiply2D.self,
    Pad2D.self,
    QuerySeq.self,
    ResizeBilinear.self,
    ResizeBilinearCrop.self,
    ResizeBilinearPad.self,
    Rotate2D.self,
    SelfCorrelate2D.self,
    Softmax1D.self,
    SoftmaxSeq.self,
    Sum1D.self,
    Sum2D.self,
    SumSeq.self,
    SelectNeurons1D.self,
    SelectNeurons2D.self,
    SelectSeq.self,
    SimilarityBatchError2D.self,
    ValueSeq.self,
    VQ2D.self,
])

///
/// Build a registry of types.
///
/// - Parameter listTypes: List of types to register.
/// - Returns: A dictionary allowing to find the available types when loading from the disk.
///
public func buildRegistry(_ listTypes: [Codable.Type]) -> [String: Codable.Type]
{
    var registry = [String:Codable.Type]()
    for type in listTypes
    {
        registry[getStr(type)] = type
    }
    return registry
}

///
/// Convert a type to a string.
///
/// - Parameter type: The type to convert to string.
/// - Returns: The string describing the type.
///
public func getStr(_ type: Codable.Type) -> String
{
    return String(describing: type)
}

/// Serialize and deserialize a layer thanks to its string type.
class LayerContainer: Codable
{
    /// The layer to serialize/deserialize.
    let layer: Layer
    
    enum Keys: String, CodingKey
    {
        case type
        case layer
    }
    
    ///
    /// Create a serializable container.
    ///
    /// - Parameter layer: The layer to wrap.
    ///
    init(_ layer: Layer)
    {
        self.layer = layer
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    required init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        let layerType = try container.decode(String.self, forKey: .type)
        
        let subContainer = try container.superDecoder(forKey: .layer)
        if let layerType = GrAI.Model.Layer.getType(layer: layerType)
        {
            self.layer = try layerType.init(from: subContainer) as! Layer
        }
        else
        {
            throw SerializationError.TypeNotFound
        }
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
    func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        let typeName = getStr(type(of: self.layer))
        try container.encode(typeName, forKey: .type)
        try container.encode(self.layer, forKey: .layer)
    }
}

/// Serialize and deserialize an activation function thanks to its string type.
class ActivationContainer: Codable
{
    /// The activatoin function to serialize/deserialize.
    let activation: ActivationFunction
    
    enum Keys: String, CodingKey
    {
        case type
        case activation
    }
    
    ///
    /// Create a seerializable container.
    ///
    /// - Parameter activation: The activation function to wrap.
    ///
    required init(_ activation: ActivationFunction)
    {
        self.activation = activation
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    required init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        let activationType = try container.decode(String.self, forKey: .type)
        
        let subContainer = try container.superDecoder(forKey: .activation)
        if let activationType =
            GrAI.Model.Activation.getType(activation: activationType)
        {
            self.activation = try activationType.init(from: subContainer)
                as! ActivationFunction
        }
        else
        {
            throw SerializationError.TypeNotFound
        }
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
    func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        let typeName = getStr(type(of: self.activation))
        try container.encode(typeName, forKey: .type)
        try container.encode(self.activation, forKey: .activation)
    }
}
