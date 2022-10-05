//
// Serialization.swift
// MAKit
//
// Created by Jean-François Reboud on 05/10/2022.
//

public enum SerializationError: Error
{
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
                   "use JLearn.Model.*.append APIs."
        }
    }
}

let LAYER_REGISTRY: [String: Codable.Type] = buildRegistry(
[
    // TODO: add elements here.
])

public func buildRegistry(_ listTypes: [Codable.Type]) -> [String: Codable.Type]
{
    var registry = [String:Codable.Type]()
    for type in listTypes
    {
        registry[getStr(type)] = type
    }
    return registry
}

public func getStr(_ type: Codable.Type) -> String
{
    return String(describing: type)
}

class LayerContainer: Codable
{
    let layer: Layer
    
    enum Keys: String, CodingKey
    {
        case type
        case layer
    }
    
    init(_ layer: Layer)
    {
        self.layer = layer
    }
    
    required init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        let layerType = try container.decode(String.self, forKey: .type)
        
        let subContainer = try container.superDecoder(forKey: .layer)
        if let layerType = MAKit.Model.Layer.getType(layer: layerType)
        {
            self.layer = try layerType.init(from: subContainer) as! Layer
        }
        else
        {
            throw SerializationError.TypeNotFound
        }
    }
    
    func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        let typeName = getStr(type(of: self.layer))
        try container.encode(typeName, forKey: .type)
        try container.encode(self.layer, forKey: .layer)
    }
}

let ACTIVATION_REGISTRY: [String: Codable.Type] = buildRegistry(
[
    // TODO: add elements here.
])

class ActivationContainer: Codable
{
    let activation: ActivationFunction
    
    enum Keys: String, CodingKey
    {
        case type
        case activation
    }
    
    required init(_ activation: ActivationFunction)
    {
        self.activation = activation
    }
    
    required init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        let activationType = try container.decode(String.self, forKey: .type)
        
        let subContainer = try container.superDecoder(forKey: .activation)
        if let activationType =
            MAKit.Model.Activation.getType(activation: activationType)
        {
            self.activation = try activationType.init(from: subContainer)
                as! ActivationFunction
        }
        else
        {
            throw SerializationError.TypeNotFound
        }
    }
    
    func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        let typeName = getStr(type(of: self.activation))
        try container.encode(typeName, forKey: .type)
        try container.encode(self.activation, forKey: .activation)
    }
}
