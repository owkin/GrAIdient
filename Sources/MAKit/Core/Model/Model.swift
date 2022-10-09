//
// Model.swift
// MAKit
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation

///
/// A model containing layers with partial chaining information
/// (`Layer.idPrev` is initialized but not `Layer.layerPrev`).
///
/// The interest of this base model is to be decodable from the disk. Hence, the different `Layer.idPrev`
/// are retrieved and a further call to `Model` allows to update the different `Layer.layerPrev`
/// thanks to the different `Layer.idPrev`.
///
public class BaseModel: Codable
{
    /// Name of the model.
    public var name = ""
    
    /// The different layers composing the model.
    public var layers = [Layer]()
    
    private enum Keys: String, CodingKey
    {
        case name
        case layers
    }
    
    ///
    /// Create a model with a name.
    ///
    /// - Parameter name: Name of the model.
    ///
    public init(name: String)
    {
        self.name = name
    }
    
    ///
    /// Create a model out of another model.
    ///
    /// - Parameter model: The model to get the layers from.
    ///
    public init(model: BaseModel)
    {
        name = model.name
        layers = model.layers
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or ortherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
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
        
        try container.encode(name, forKey: .name)
        try container.encode(layers, forKey: .layers)
        
        var listLayerContainer = [LayerContainer]()
        for layer in layers
        {
            listLayerContainer.append(LayerContainer(layer))
        }
        try container.encode(listLayerContainer, forKey: .layers)
    }
    
    ///
    /// Add a layer to the queue of the model.
    ///
    /// - Parameter layer: The layer to add to the model.
    ///
    func append(_ layer: Layer)
    {
        layers.append(layer)
    }
    
    ///
    /// Create a model by copying the inner layers.
    ///
    /// - Parameters:
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///     - inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new model. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    func copy(
        mapping: inout Dictionary<Int, Layer>,
        inPlace: Bool) -> BaseModel
    {
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
    
    ///
    /// Resize the model, creating a new one.
    ///
    /// - Parameters:
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///     - inPlace: Whether hard resources should be copied as is.
    ///     - imageWidth: New size width.
    ///     - imageHeight: New size height.
    ///
    /// - Returns: A new model. When `inPlace` is false, `initKernel` is
    ///  necessary in order to recreate hard resources.
    ///
    func resize(
        mapping: inout Dictionary<Int, Layer>,
        inPlace: Bool,
        imageWidth: Int,
        imageHeight: Int) -> BaseModel
    {
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

///
/// A model containing layers with full chaining information
/// (`Layer.idPrev` AND `Layer.layerPrev` are initialized).
///
public class Model: BaseModel
{
    /// The state of the optimization process.
    var _kernel = OptimizerKernel()
    
    /// Get optimizer parameters.
    public var optimizerParams: MAKit.Optimizer.Params
    {
        get {
            return _kernel.params
        }
    }
    
    /// Get alpha (learning rate) value in the optimizer.
    public var alpha: Double
    {
        get {
            return _kernel.algo.alpha
        }
    }
    /// Get alpha (learning rate) percent value in the optimizer.
    public var alphaPercent: Double
    {
        get {
            return _kernel.algo.alphaPercent
        }
    }
    
    /// Get lambda (weight decay) value in the optimizer.
    public var lambda: Double?
    {
        get {
            return _kernel.algo.lambda
        }
    }
    /// Get lambda (weight decay) percent value in the optimizer.
    public var lambdaPercent: Double?
    {
        get {
            return _kernel.algo.lambdaPercent
        }
    }
    
    /// Get lower bound value in the optimizer.
    public var lowerBound: Double?
    {
        get {
            return _kernel.algo.lowerBound
        }
    }
    /// Get lower bound percent value in the optimizer.
    public var lowerBoundPercent: Double?
    {
        get {
            return _kernel.algo.lowerBoundPercent
        }
    }
    
    /// Get upper bound  value in the optimizer.
    public var upperBound: Double?
    {
        get {
            return _kernel.algo.upperBound
        }
    }
    /// Get upper bound percent value in the optimizer.
    public var upperBoundPercent: Double?
    {
        get {
            return _kernel.algo.upperBoundPercent
        }
    }
    
    // TODO: add elements here.
    
    /// Get/Set the weights of the different layers.
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
    
    /// Get/Set the need to compute the gradients of the weights for the different layers.
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
    
    /// Get/Set: if true: the gradients of the weights are accumulated in place.
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
    
    /// Get/Set: whether layer has been visited by the backward pass of its following layer in the queue.
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
    
    /// Get/Set the GPU ID where the model is sent.
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
    
    /// Get/Set the learning mode: Training or Inference.
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
    
    /// Get/Set the batch size of data.
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
    
    ///
    /// Create a model ouf of a base model with partial chaining information, resolving the chaining
    /// thanks to `Layer.idPrev`.
    ///
    /// - Parameters:
    ///     - model: The base model with partial chaining information.
    ///     - modelsPrev: A list of models where layers may appear as `layerPrev` of some layers of
    ///     model.
    ///
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
    
    ///
    /// Create a model ouf of a base model with partial chaining information, resolving the chaining
    /// thanks to `Layer.idPrev`.
    ///
    /// - Parameters:
    ///     - model: The base model with partial chaining information.
    ///     - layersPrev: A list of layers that may appear as `layerPrev` of some layers of model.
    ///
    public init(model: BaseModel, layersPrev: [Layer])
    {
        super.init(model: model)
        
        let layers = layersPrev + self.layers
        for layer in self.layers
        {
            layer.initLinks(layers)
        }
    }
    
    ///
    /// Create a model with a name.
    ///
    /// - Parameter name: Name of the model.
    ///
    public override init(name: String)
    {
        super.init(name: name)
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or ortherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    public required init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    ///
    /// Set the parameters for the optimizer.
    ///
    /// - Parameter params: The parameters to set.
    ///
    public func setupOptimizers(params: MAKit.Optimizer.Params)
    {
        _kernel.params = params
    }
    
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
    /// Example: the batch normalization layer.
    /// Note that the weights are not initialized here, they have a dedicated API (initWeights).
    ///
    /// - Parameters:
    ///     - phase: Running phase of the model: `Training` or `Inference`.
    ///     - deviceID: GPU device where the model will be executed.
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
    
    ///
    /// Initialize hard resources and set the parameters for the optimizer.
    ///
    /// Hard resources are the resources that are not dependent on the batch size.
    /// Example: the batch normalization layer.
    /// Note that the weights are not initialized here, they have a dedicated API (initWeights).
    ///
    /// - Parameters:
    ///     - params: The parameters for the `Optimizer`.
    ///     - phase: Running phase of the model: `Training` or `Inference`.
    ///     - deviceID: GPU device where the model will be executed.
    ///
    public func initialize(params: MAKit.Optimizer.Params,
                           phase: Phase,
                           deviceID: Int = 0)
    {
        setupOptimizers(params: params)
        initKernel(phase: phase, deviceID: deviceID)
    }
    
    ///
    /// Reset the state status before the forward.
    ///
    /// The batch size may be change during the different steps. Generally speaking, the hypothesis is that
    /// it is possible to reverse memory for the max batch size during the first step. Some later steps
    /// (especially the last one) may have a smaller batch size.
    /// The different layers must also update their dirty state so that we can trace whether every layer
    /// has been visited by at least one backward during a Training phase.
    ///
    /// - Parameter batchSize: The batch size of data.
    ///
    public func updateKernel(batchSize: Int)
    {
        self.batchSise = batchSize
        self.dirty = true
    }
    
    ///
    /// Create a list of copied models (copy their their inner layers).
    ///
    /// - Parameters:
    ///     - models: The different models to copy.
    ///     - inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: The list of copied models. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    public static func copy(models: [BaseModel], inPlace: Bool) -> [Model]
    {
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
    
    ///
    /// Resize a list of models, creating a list of new ones.
    ///
    /// - Parameters:
    ///     - models: The different models to resize.
    ///     - imageWidth: New size width.
    ///     - imageHeight: New size height.
    ///     - inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: The list of resized models. When `inPlace` is false, `initKernel` is
    ///  necessary in order to recreate hard resources.
    ///
    public static func resize(
        models: [BaseModel],
        imageWidth: Int,
        imageHeight: Int,
        inPlace: Bool) -> [Model]
    {
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
    
    /// Notify optimizer that a step has been completed.
    public func incStep()
    {
        _kernel.incStep()
    }
    
    /// Notify optimizer tthat a new epoch has begun.
    public func incEpoch()
    {
        _kernel.incEpoch()
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking.
    ///
    /// Throw an error if batch size is greater than the first batch size.
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
    /// Throw an error if batch size is greater than the first batch size.
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
    /// Throw an error if batch size is greater than the first batch size.
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
    
    ///
    /// Update the weights of a model.
    ///
    /// Throw an error when layers have not been visited by any  backward pass.
    ///
    /// - Parameters:
    ///     - gradientNorm: A norm to scale the weights' gradients.
    ///     - layers: The list of layers that potentially contain weights to update.
    ///
    public func update(gradientNorm: Double? = nil,
                       layers: [Layer] = []) throws
    {
        _checkLayers(layers: layers)
        
        let myLayers = layers.count > 0 ? layers : self.layers
        if MAKit.Opti.GPU
        {
            let gNorm: Float? = gradientNorm != nil ?
                                Float(gradientNorm!) : nil
            try _kernel.algo.udpateGPU(layers: myLayers,
                                       gradientNorm: gNorm)
        }
        else
        {
            try _kernel.algo.udpateCPU(layers: myLayers,
                                       gradientNorm: gradientNorm)
        }
    }
    
    ///
    /// Multiply weights' gradients by a scalar.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameters:
    ///     - factor: The coefficient to multiply the weights' gradients by.
    ///     - layers: The list of layers to consider.
    ///
    public func multiplyGradient(factor: Double,
                                 layers: [Layer] = []) throws
    {
        _checkLayers(layers: layers)
        
        let myLayers = layers.count > 0 ? layers : self.layers
        if MAKit.Opti.GPU
        {
            try _kernel.algo.multiplyGradientGPU(layers: myLayers,
                                                 factor: Float(factor))
        }
        else
        {
            try _kernel.algo.multiplyGradientCPU(layers: myLayers,
                                                 factor: factor)
        }
    }
    
    ///
    /// Get gradient norm.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameter layers: The list of layers to consider.
    /// - Returns: The gradient norm.
    ///
    public func getGradientNorm(layers: [Layer] = []) throws -> Double
    {
        _checkLayers(layers: layers)
        
        let myLayers = layers.count > 0 ? layers : self.layers
        if MAKit.Opti.GPU
        {
            return try Double(_kernel.algo.getGradientNormGPU(myLayers))
        }
        else
        {
            return try _kernel.algo.getGradientNormCPU(myLayers)
        }
    }
    
    ///
    /// Get the weights'  gradients.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameter layers: The list of layers to consider.
    /// - Returns: The list of weights' gradients.
    ///
    public func collectGradients(layers: [Layer]) throws -> [Double]
    {
        _checkLayers(layers: layers)
        
        if MAKit.Opti.GPU
        {
            var gradients = [Double]()
            for gradient in try _kernel.algo.getGradientsGPU(layers)
            {
                gradients.append(Double(gradient))
            }
            return gradients
        }
        else
        {
            return try _kernel.algo.getGradientsCPU(layers)
        }
    }
    
    ///
    /// Average the weights of the different layers of a model that has been trained in similar conditions.
    ///
    /// Particularly usefull considering models that are being trained in parallel on several GPU,
    /// in a federated learning style.
    ///
    /// - Parameters:
    ///     - models: The list of "mirrored" models.
    ///     - modelCoeffs: A list of coefficients to ponder the average of the weights (especially when
    ///     the different GPU do not have same capability).
    ///
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
    
    ///
    /// Get the chain of layers that go from the "first one" of the model to an endpoint, following the
    /// `Layer.layerPrev` chaining.
    ///
    /// - Parameter endpoint: The final layer of our chain.
    /// - Returns: The list of layers between the "first one" and `endpoint`.
    ///
    public func getGraph(_ endpoint: Layer) -> [Layer]
    {
        var layers = [Layer]()
        endpoint.getGraph(&layers)
        layers = layers.reversed()
        return layers
    }
}
