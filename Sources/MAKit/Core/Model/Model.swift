//
// Model.swift
// MAKit
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation

public class BaseModel: Codable
{
    public var name = ""
    
    public var layers = [Layer]()
    
    private enum Keys: String, CodingKey
    {
        case name
        case layers
    }
    
    public init(name: String)
    {
        self.name = name
    }
    
    public init(model: BaseModel)
    {
        name = model.name
        layers = model.layers
    }
    
    public required init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        
        name = try container.decode(String.self, forKey: .name)
        
        let listLayerContainer = try container.decode([LayerContainer].self,
                                                      forKey: .layers)
        var layers = [Layer]()
        for layerContainer in listLayerContainer
        {
            layers.append(layerContainer.layer)
        }
        self.layers = layers
    }
    
    public func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        
        try container.encode(name, forKey: .name)
        try container.encode(layers, forKey: .layers)
        
        var listLayerContainer = [LayerContainer]()
        for layer in layers
        {
            listLayerContainer.append(LayerContainer(layer))
        }
        try container.encode(listLayerContainer, forKey: .layers)
    }
    
    func append(_ layer: Layer)
    {
        layers.append(layer)
    }
    
    func copy(
        mapping: inout Dictionary<Int, Layer>,
        inPlace: Bool) -> BaseModel
    {
        // inPLace allows to give resources without having to re build them.
        let newModel = BaseModel(name: name)
        var newLayers = [Layer]()
        
        for layer in layers
        {
            let newLayer = layer.copy(mapping: mapping, inPlace: inPlace)
            newLayers.append(newLayer)
            mapping[layer.id] = newLayer
        }
        
        newModel.layers = newLayers
        return newModel
    }
    
    func resize(
        mapping: inout Dictionary<Int, Layer>,
        inPlace: Bool,
        imageWidth: Int,
        imageHeight: Int) -> BaseModel
    {
        // inPLace allows to give resources without having to re build them.
        let newModel = BaseModel(name: name)
        var newLayers = [Layer]()
        
        for layer in layers
        {
            let newLayer: Layer
            if let layerResize = layer as? LayerResize
            {
                newLayer = layerResize.resize(
                    imageWidth: imageWidth,
                    imageHeight: imageHeight,
                    mapping: mapping,
                    inPlace: inPlace
                )
            }
            else
            {
                newLayer = layer.copy(mapping: mapping, inPlace: inPlace)
            }
            newLayers.append(newLayer)
            mapping[layer.id] = newLayer
        }
        
        newModel.layers = newLayers
        return newModel
    }
}

public class Model: BaseModel
{
    // TODO: add elements here.
    
    public var weights: [[Float]]
    {
        get {
            var weightsList = [[Float]]()
            if MAKit.Opti.GPU
            {
                for layer in layers
                {
                    if let layerUpdate = layer as? LayerUpdate
                    {
                        weightsList.append(layerUpdate.weightsGPU)
                    }
                    else
                    {
                        weightsList.append([])
                    }
                }
            }
            else
            {
                for layer in layers
                {
                    if let layerUpdate = layer as? LayerUpdate
                    {
                        weightsList.append(layerUpdate.weightsCPU)
                    }
                    else
                    {
                        weightsList.append([])
                    }
                }
            }
            return weightsList
        }
        set {
            if newValue.count != layers.count
            {
                fatalError(
                    "Weights do not have the expected number of elements."
                )
            }
            if MAKit.Opti.GPU
            {
                for (i, layer) in layers.enumerated()
                {
                    if let layerUpdate = layer as? LayerUpdate
                    {
                        layerUpdate.weightsGPU = newValue[i]
                    }
                }
            }
            else
            {
                for (i, layer) in layers.enumerated()
                {
                    if let layerUpdate = layer as? LayerUpdate
                    {
                        layerUpdate.weightsCPU = newValue[i]
                    }
                }
            }
        }
    }
    
    public var computeDeltaWeights: Bool
    {
        get {
            for layer in layers.reversed()
            {
                if let layerUpdate = layer as? LayerUpdate
                {
                    return layerUpdate.computeDeltaWeights
                }
            }
            return false
        }
        set {
            for layer in layers
            {
                if let layerUpdate = layer as? LayerUpdate
                {
                    layerUpdate.computeDeltaWeights = newValue
                }
            }
        }
    }
    
    public var accumulateDeltaWeights: Bool
    {
        get {
            for layer in layers.reversed()
            {
                if let layerUpdate = layer as? LayerUpdate
                {
                    return layerUpdate.accumulateDeltaWeights
                }
            }
            return false
        }
        set {
            for layer in layers
            {
                if let layerUpdate = layer as? LayerUpdate
                {
                    layerUpdate.accumulateDeltaWeights = newValue
                }
            }
        }
    }
    
    var dirty: Bool
    {
        get {
            return layers.first!.dirty
        }
        set {
            for layer in layers
            {
                layer.dirty = newValue
            }
        }
    }
    
    public var deviceID: Int
    {
        get {
            return layers.first!.deviceID
        }
        set {
            for layer in layers
            {
                layer.deviceID = newValue
            }
        }
    }
    
    public var phase: Phase?
    {
        get {
            return layers.first!.phase
        }
        set {
            for layer in layers
            {
                layer.phase = newValue
            }
        }
    }
    
    var batchSise: Int
    {
        get {
            return layers.first!.batchSize
        }
        set {
            for layer in layers
            {
                layer.batchSize = newValue
            }
        }
    }
    
    public init(model: BaseModel, modelsPrev: [BaseModel])
    {
        super.init(model: model)
        
        var layers = [Layer]()
        for modelPrev in modelsPrev
        {
            layers += modelPrev.layers
        }
        
        layers += self.layers
        for layer in self.layers
        {
            layer.initLinks(layers)
        }
    }
    
    public init(model: BaseModel, layersPrev: [Layer])
    {
        super.init(model: model)
        
        let layers = layersPrev + self.layers
        for layer in self.layers
        {
            layer.initLinks(layers)
        }
    }
    
    public override init(name: String)
    {
        super.init(name: name)
    }
    
    public required init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    // TODO: add elements here.
    
    ///
    /// Clean state resources.
    ///
    /// State resources are the resources that are dependent on the batch size.
    ///
    public func resetKernel()
    {
        if MAKit.Opti.GPU
        {
            for layer in layers
            {
                layer.resetKernelGPU()
            }
        }
        else
        {
            for layer in layers
            {
                layer.resetKernelCPU()
            }
        }
    }
    
    ///
    /// Initialize hard resources.
    ///
    /// Hard resources are the resources that are not dependent on the batch size.
    /// Example: the weights, the batch normalization layer.
    ///
    public func initKernel(phase: Phase? = nil, deviceID: Int = 0)
    {
        self.phase = phase
        if phase != nil && phase! == .Inference
        {
            self.computeDeltaWeights = false
        }
        
        if MAKit.Opti.GPU
        {
            self.deviceID = deviceID
            DispatchQueue.concurrentPerform(iterations: layers.count)
            {
                (layerNum: Int) in
                
                let layer = self.layers[layerNum]
                layer.resetKernelGPU()
                
                if MAKit.Loop.gradientChecking
                {
                    layer.initKernelCPU() // initialize BN
                }
                layer.initKernelGPU()
                
                if let layerUpdate = layer as? LayerUpdate
                {
                    layerUpdate.initWeightsGPU()
                    
                    if MAKit.Loop.gradientChecking
                    {
                        layerUpdate.initWeightsCPU() // initialize BN weights
                    }
                }
            }
        }
        else
        {
            DispatchQueue.concurrentPerform(iterations: layers.count)
            {
                (layerNum: Int) in
                
                let layer = self.layers[layerNum]
                
                layer.resetKernelCPU()
                layer.initKernelCPU()
                
                if let layerUpdate = layer as? LayerUpdate
                {
                    layerUpdate.initWeightsCPU()
                }
            }
        }
    }
    
    /// Initialize the weights.
    public func initWeights()
    {
        if MAKit.Opti.GPU
        {
            for layer in layers
            {
                if let layerUpdate = layer as? LayerUpdate
                {
                    layerUpdate.initWeightsGPU()
                }
            }
        }
        else
        {
            for layer in layers
            {
                if let layerUpdate = layer as? LayerUpdate
                {
                    layerUpdate.initWeightsCPU()
                }
            }
        }
    }
    
    // TODO: add elements here.
    
    public func updateKernel(batchSize: Int)
    {
        self.batchSise = batchSize
        self.dirty = true
    }
    
    public static func copy(models: [BaseModel], inPlace: Bool) -> [Model]
    {
        // inPLace allows to give resources without having to re build them.
        var mapping = Dictionary<Int, Layer>()
        
        var newModels = [Model]()
        for model in models
        {
            let newBaseModel = model.copy(mapping: &mapping, inPlace: inPlace)
            let newModel = Model(model: newBaseModel, modelsPrev: newModels)
            newModels.append(newModel)
        }
        
        return newModels
    }
    
    public static func resize(
        models: [BaseModel],
        imageWidth: Int,
        imageHeight: Int,
        inPlace: Bool) -> [Model]
    {
        // inPLace allows to give resources without having to re build them.
        var mapping = Dictionary<Int, Layer>()
        
        var newModels = [Model]()
        for model in models
        {
            let newBaseModel = model.resize(
                mapping: &mapping,
                inPlace: inPlace,
                imageWidth: imageWidth,
                imageHeight: imageHeight
            )
            let newModel = Model(model: newBaseModel, modelsPrev: newModels)
            newModels.append(newModel)
        }
        
        return newModels
    }
    
    // TODO: add elements here.
    
    ///
    /// Apply the forward pass of the Gradient Checking.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public func forwardGradients() throws
    {
        if MAKit.Opti.GPU
        {
            for layer in layers
            {
                try layer.forwardGCGPU()
            }
        }
        else
        {
            for layer in layers
            {
                try layer.forwardGCCPU()
            }
        }
    }
    
    ///
    /// Apply the forward pass.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public func forward() throws
    {
        if MAKit.Opti.GPU
        {
            for layer in layers
            {
                let desc = layer.description + String(layer.id)
                MAKit.Time.start()
                
                try layer.forwardGPU()
                
                try MAKit.Time.stop(id: "ForwardGPU", description: desc)
            }
        }
        else
        {
            for layer in layers
            {
                let desc = layer.description + String(layer.id)
                MAKit.Time.start()
                
                try layer.forwardCPU()
                
                try MAKit.Time.stop(id: "ForwardCPU", description: desc)
            }
        }
    }
    
    ///
    /// Apply the backward pass.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public func backward() throws
    {
        if MAKit.Opti.GPU
        {
            for layer in layers.reversed()
            {
                // Note that there are two steps in backward:
                // previous.backward and current.backwardWeights.
                // We could set dirty inside mustComputeBackward in order
                // to remove the check below...
                // But then some dirty layers could try to compute their
                // backwardWeights while we wanted to stop back propagation.
                // Plus some layers are included in others.
                // The latter may be dirty when the first ones compute
                // their backward pass.
                if !layer.dirty
                {
                    let desc = layer.description + String(layer.id)
                    MAKit.Time.start()
                    
                    try layer.backwardGPU()
                    if layer.mustComputeBackward
                    {
                        layer.propagateDirty()
                    }
                    
                    try MAKit.Time.stop(id: "BackwardGPU", description: desc)
                }
            }
        }
        else
        {
            for layer in layers.reversed()
            {
                // Same as above.
                if !layer.dirty
                {
                    let desc = layer.description + String(layer.id)
                    MAKit.Time.start()
                    
                    layer.backwardCPU()
                    if layer.mustComputeBackward
                    {
                        layer.propagateDirty()
                    }
                    
                    try MAKit.Time.stop(id: "BackwardCPU", description: desc)
                }
            }
        }
    }
    
    private func _checkLayers(layers: [Layer])
    {
        for layer1 in layers
        {
            var found = false
            for layer2 in self.layers
            {
                if layer2 === layer1
                {
                    found = true
                    break
                }
            }
            if !found
            {
                fatalError("Did not find layer \(layer1.id).")
            }
        }
    }
    
    // TODO: add elements here.
    
    public static func synchronizeWeights(_ models: [Model],
                                          _ modelCoeffs: [Float])
    {
        for indexLayer in 0..<models[0].layers.count
        {
            var listWeights = [[Float]]()
            for (model, modelCoeff) in zip(models, modelCoeffs)
            {
                var weights = [Float]()
                if let layerUpdate = model.layers[indexLayer]
                   as? LayerUpdate
                {
                    for weight in layerUpdate.weightsGPU
                    {
                        weights.append(weight * modelCoeff)
                    }
                }
                
                listWeights.append(weights)
            }
            
            var newWeights = listWeights[0]
            for indexWeights in 1..<listWeights.count
            {
                for i in 0..<newWeights.count
                {
                    newWeights[i] += listWeights[indexWeights][i]
                }
            }
            
            for model in models
            {
                if let layerUpdate = model.layers[indexLayer] as? LayerUpdate
                {
                    layerUpdate.weightsGPU = newWeights
                }
            }
        }
        
        for model in models
        {
            model.initWeights()
        }
    }
    
    public func getGraph(_ endpoint: Layer) -> [Layer]
    {
        var layers = [Layer]()
        endpoint.getGraph(&layers)
        layers = layers.reversed()
        return layers
    }
}
