//
// Model.swift
// GrAITorchTests
//
// Created by Jean-François Reboud on 19/10/2022.
//

import GrAIdient
import PythonKit

/// Model to test against PyTorch.
class ModelTestConv1
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Convolution with stride and biases
    ///   - MaxPool
    ///   - FullyConnected with biases
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConv1", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 7, nbChannels: 5, stride: 2,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        
        layer = MaxPool2D(
            layerPrev: layer, size: 2, stride: 2, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 1, nbChannels: 10, stride: 2,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        
        layer = AdaptiveAvgPool2D(layerPrev: layer, size: 7, params: params)
        
        var head: Layer1D = FullyConnected(
            layerPrev: layer,
            nbNeurons: 10,
            activation: ReLU.str,
            biases: true,
            params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv1_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        
        // Apply weights on the `GrAIdient` model's layers.
        var cur = 0
        for num_layer in 0..<model.layers.count
        {
            // Load weights and biases.
            if let convLayer = model.layers[num_layer] as? Convolution2D
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                convLayer.weightsCPU = weightsTmp + biases
            }
            // Load weights and biases.
            else if let flLayer = model.layers[num_layer] as? FullyConnected
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
        }
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestConv2
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Convolution with batch normalization and no biases
    ///   - MaxPool with overlapping
    ///   - ResNet like shortcut
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConv2", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        // Note that as we use batch normalization, the biases are
        // redundant: do not update them.
        layer = Convolution2D(
            layerPrev: layer,
            size: 7, nbChannels: 5, stride: 2,
            activation: ReLU.str, biases: false, bn: true,
            params: params
        )
        
        layer = MaxPool2D(
            layerPrev: layer, size: 3, stride: 2, params: params
        )
        
        let layer1 = layer
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 5, stride: 1,
            activation: ReLU.str, biases: false, bn: true,
            params: params
        )
        
        layer = Sum2D(layersPrev: [layer, layer1], params: params)
        
        layer = AdaptiveAvgPool2D(layerPrev: layer, size: 7, params: params)
        
        var head: Layer1D = FullyConnected(
            layerPrev: layer,
            nbNeurons: 10,
            activation: ReLU.str,
            biases: true,
            params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv2_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        
        // Apply weights on the `GrAIdient` model's layers.
        var cur = 0
        for num_layer in 0..<model.layers.count
        {
            // Load weights, the batch normalization's weights and biases
            // and the batch normalization's running average and deviation.
            if let convLayer = model.layers[num_layer] as? Convolution2D
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let Ɣ: [Float] = weights[cur]
                cur += 1
                let β: [Float] = weights[cur]
                cur += 1
                let Eμ: [Float] = weights[cur]
                cur += 1
                let Eσ2: [Float] = weights[cur]
                cur += 1
                cur += 1
                
                convLayer.weightsCPU = weightsTmp + Ɣ + β
                convLayer.statsCPU = Eμ + Eσ2
            }
            // Load weights and biases.
            else if let flLayer = model.layers[num_layer] as? FullyConnected
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
        }
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestFFT
{
    ///
    /// Create the model.
    ///
    /// Principle features:
    ///   - 2D Frequences & scale
    ///   - IRDFT
    ///   - Decorrelate color
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestFFT", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 6, width: size, height: size,
            params: params
        )
        let firstLayer: Layer2D = layer
        
        layer = FTFrequences2D(
            nbChannels: 6, dimension: size,
            params: params
        )
        
        layer = Multiply2D(
            layersPrev: [firstLayer, layer], params: params
        )
        
        layer = IRDFT2RGB(
            layerPrev: layer, params: params
        )
        
        layer = LinearScale2D(
            layerPrev: layer,
            weight: 1.0 / 4.0, bias: 0.0,
            params: params
        )
        
         layer = DecorrelateRGB(
            layerPrev: layer,
            correlation: [
                0.26, 0.09, 0.02,
                0.27, 0.00, -0.05,
                0.27, -0.09, 0.03
            ].map { $0 / 0.4619524 },
            params: params
        )
        
        layer = Activation2D(
            layerPrev: layer,
            activation: Sigmoid.str,
            params: params
        )
        
        layer = LinearScale2D(
            layerPrev: layer,
            weight: 2.0, bias: -1.0,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = SelectNeurons1D(
            layerPrev: head, neurons: [0], coeffs: [1.0], params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        return model
    }
}

/// Generic model to test against PyTorch.
class ModelTestConv
{
    ///
    /// Load weights in the model.
    ///
    /// - Parameters:
    ///     - model: The model.
    ///     - weights: The weights.
    ///
    static func initWeights(model: Model, weights: [[Float]])
    {
        // Apply weights on the `GrAIdient` model's layers.
        var cur = 0
        for num_layer in 0..<model.layers.count
        {
            // Load weights and biases.
            if let convLayer = model.layers[num_layer] as? Convolution2D
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                convLayer.weightsCPU = weightsTmp + biases
            }
            // Load weights and biases.
            else if let flLayer = model.layers[num_layer] as? FullyConnected
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
        }
    }
}

/// Model to test against PyTorch.
class ModelTestConvS1K2: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConvS1K2", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 5, stride: 1,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv3_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestConvS1K3: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConvS1K3", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 5, stride: 1,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv4_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestConvS1K4: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConvS1K4", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 4, nbChannels: 5, stride: 1,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv5_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestConvS2K2: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConvS2K2", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 5, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv3_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestConvS2K3: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConvS2K3", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 5, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv4_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestConvS2K4: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConvS2K4", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 4, nbChannels: 5, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv5_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestConvS4K2: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConvS4K2", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 5, stride: 4,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv3_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestConvS4K3: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConvS4K3", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 5, stride: 4,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv4_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestConvS4K4: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestConvS4K4", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 4, nbChannels: 5, stride: 4,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_conv5_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestDeConvS1K2: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestDeConvS1K2", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 5, stride: 1,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_deconv1_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestDeConvS1K3: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestDeConvS1K3", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 5, stride: 1,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_deconv2_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestDeConvS1K4: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestDeConvS1K4", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 4, nbChannels: 5, stride: 1,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_deconv3_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestDeConvS2K2: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestDeConvS2K2", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 5, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_deconv1_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestDeConvS2K3: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestDeConvS2K3", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 5, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_deconv2_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestDeConvS2K4: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestDeConvS2K4", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 4, nbChannels: 5, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_deconv3_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestDeConvS4K2: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestDeConvS4K2", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 5, stride: 4,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_deconv1_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestDeConvS4K3: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestDeConvS4K3", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 5, stride: 4,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_deconv2_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestDeConvS4K4: ModelTestConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestDeConvS4K4", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 4, nbChannels: 5, stride: 4,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_deconv3_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestCat
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Concat2D
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestCat", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        let layer1: Layer2D = Convolution2D(
            layerPrev: layer,
            size: 1, nbChannels: 6, stride: 1,
            activation: nil, biases: true, bn: false,
            params: params
        )
        let layer2: Layer2D = Convolution2D(
            layerPrev: layer,
            size: 1, nbChannels: 9, stride: 1,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        layer = Concat2D(
            layersPrev: [layer1, layer2], params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_cat_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        
        // Apply weights on the `GrAIdient` model's layers.
        var cur = 0
        for num_layer in 0..<model.layers.count
        {
            // Load weights and biases.
            if let convLayer = model.layers[num_layer] as? Convolution2D
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                convLayer.weightsCPU = weightsTmp + biases
            }
            // Load weights and biases.
            else if let flLayer = model.layers[num_layer] as? FullyConnected
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
        }
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestResize
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Deconvolution with even kernel and no stride
    ///
    /// - Parameters:
    ///     - sizeInput: The size of the input data.
    ///     - sizeOutput: The output size of the resize operation.
    /// - Returns: The built model.
    ///
    static func build(sizeInput: Int, sizeOutput: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestResize", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: sizeInput,
            height: sizeInput,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 1, nbChannels: 5, stride: 1,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        layer = ResizeBilinearCrop(
            layerPrev: layer,
            scale: Double(sizeOutput) / Double(sizeInput),
            offsetI: 0, offsetJ: 0,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(
            layerPrev: layer, params: params
        )
        
        head = FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: nil,
            biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_resize_weights(sizeOutput)
        
        let weights = [[Float]](data.tuple2.0)!
        
        // Apply weights on the `GrAIdient` model's layers.
        var cur = 0
        for num_layer in 0..<model.layers.count
        {
            // Load weights and biases.
            if let convLayer = model.layers[num_layer] as? Convolution2D
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                convLayer.weightsCPU = weightsTmp + biases
            }
            // Load weights and biases.
            else if let flLayer = model.layers[num_layer] as? FullyConnected
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
        }
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestPatchConv
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - FullyConnectedPatch
    ///   - AvgPoolSeq
    ///
    /// - Parameters:
    ///     - size: The size of the input data.
    ///     - patch: The kernel split size of the input data.
    /// - Returns: The built model.
    ///
    static func build(size: Int, patch: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestPatchConv", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        let layerSeq: LayerSeq = FullyConnectedPatch(
            layerPrev: layer, patch: patch, nbNeurons: 5,
            activation: nil, biases: true,
            params: params
        )
        
        var head: Layer1D = AvgPoolSeq(
            layerPrev: layerSeq, params: params
        )
        
        head = FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: nil, biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_patch_conv_weights(size, patch)
        
        let weights = [[Float]](data.tuple2.0)!
        
        // Apply weights on the `GrAIdient` model's layers.
        var cur = 0
        for num_layer in 0..<model.layers.count
        {
            // Load weights and biases.
            if let flLayer = model.layers[num_layer] as? FullyConnectedPatch
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
            else if let flLayer = model.layers[num_layer] as? FullyConnected
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
        }
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestAttention1
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - FullyConnectedSeq
    ///   - QuerySeq
    ///   - SoftmaxSeq
    ///   - ValueSeq
    ///
    /// - Parameters:
    ///     - size: The size of the input data.
    ///     - patch: The kernel split size of the input data.
    /// - Returns: The built model.
    ///
    static func build(size: Int, patch: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestAttention1", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        var layerSeq: LayerSeq = FullyConnectedPatch(
            layerPrev: layer, patch: patch, nbNeurons: 5,
            activation: nil, biases: true,
            params: params
        )
        
        let query = FullyConnectedSeq(
            layerPrev: layerSeq, nbNeurons: 5,
            activation: nil, biases: true,
            params: params
        )
        let key = FullyConnectedSeq(
            layerPrev: layerSeq, nbNeurons: 5,
            activation: nil, biases: true,
            params: params
        )
        let value = FullyConnectedSeq(
            layerPrev: layerSeq, nbNeurons: 5,
            activation: nil, biases: true,
            params: params
        )
        
        var score: LayerSeq = QuerySeq(
            query: query, key: key, nbHeads: 1, params: params
        )
        score = SoftmaxSeq(layerPrev: score, nbHeads: 1, params: params)
        
        layerSeq = ValueSeq(
            value: value, score: score, nbHeads: 1,
            params: params
        )
        
        layerSeq = FullyConnectedSeq(
            layerPrev: layerSeq, nbNeurons: 5,
            activation: nil, biases: true,
            params: params
        )
        
        var head: Layer1D = AvgPoolSeq(
            layerPrev: layerSeq, params: params
        )
        
        head = FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: nil, biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_attention1_weights(size, patch)
        
        let weights = [[Float]](data.tuple2.0)!
        
        // Apply weights on the `GrAIdient` model's layers.
        var cur = 0
        for num_layer in 0..<model.layers.count
        {
            // Load weights and biases.
            if let flLayer = model.layers[num_layer] as? FullyConnectedPatch
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
            else if let flLayer = model.layers[num_layer] as? FullyConnectedSeq
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
            else if let flLayer = model.layers[num_layer] as? FullyConnected
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
        }
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestAttention2
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - FullyConnectedSeq
    ///   - QuerySeq
    ///   - SoftmaxSeq
    ///   - ValueSeq
    ///
    /// - Parameters:
    ///     - size: The size of the input data.
    ///     - patch: The kernel split size of the input data.
    /// - Returns: The built model.
    ///
    static func build(size: Int, patch: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestAttention2", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        var layerSeq: LayerSeq = FullyConnectedPatch(
            layerPrev: layer, patch: patch, nbNeurons: 6,
            activation: nil, biases: true,
            params: params
        )
        
        let query: LayerSeq = FullyConnectedSeq(
            layerPrev: layerSeq, nbNeurons: 6,
            activation: nil, biases: true,
            params: params
        )
        let key: LayerSeq = FullyConnectedSeq(
            layerPrev: layerSeq, nbNeurons: 6,
            activation: nil, biases: true,
            params: params
        )
        let value: LayerSeq = FullyConnectedSeq(
            layerPrev: layerSeq, nbNeurons: 6,
            activation: nil, biases: true,
            params: params
        )
        
        let nbHeads = 3
        layerSeq = QuerySeq(
            query: query, key: key, nbHeads: nbHeads,
            params: params
        )
        layerSeq = SoftmaxSeq(
            layerPrev: layerSeq, nbHeads: nbHeads,
            params: params
        )
            
        layerSeq = ValueSeq(
            value: value, score: layerSeq, nbHeads: nbHeads,
            params: params
        )
        
        layerSeq = FullyConnectedSeq(
            layerPrev: layerSeq, nbNeurons: 6,
            activation: nil, biases: true,
            params: params
        )
        
        var head: Layer1D = AvgPoolSeq(
            layerPrev: layerSeq, params: params
        )
        
        head = FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: nil, biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_attention2_weights(size, patch)
        
        let weights = [[Float]](data.tuple2.0)!
        
        // Apply weights on the `GrAIdient` model's layers.
        var cur = 0
        for num_layer in 0..<model.layers.count
        {
            // Load weights and biases.
            if let flLayer = model.layers[num_layer] as? FullyConnectedPatch
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
            else if let flLayer = model.layers[num_layer] as? FullyConnectedSeq
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
            else if let flLayer = model.layers[num_layer] as? FullyConnected
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
        }
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestLayerNorm
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - LayerNormSeq
    ///
    /// - Parameters:
    ///     - size: The size of the input data.
    ///     - patch: The kernel split size of the input data.
    /// - Returns: The built model.
    ///
    static func build(size: Int, patch: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestLayerNorm", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        var layerSeq: LayerSeq = FullyConnectedPatch(
            layerPrev: layer, patch: patch, nbNeurons: 5,
            activation: nil, biases: true,
            params: params
        )
        
        layerSeq = LayerNormSeq(
            layerPrev: layerSeq, activation: nil, params: params
        )
        
        var head: Layer1D = AvgPoolSeq(
            layerPrev: layerSeq, params: params
        )
        
        head = FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: nil, biases: true,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_layer_norm_weights(size, patch)
        
        let weights = [[Float]](data.tuple2.0)!
        
        // Apply weights on the `GrAIdient` model's layers.
        var cur = 0
        for num_layer in 0..<model.layers.count
        {
            // Load weights and biases.
            if let flLayer = model.layers[num_layer] as? FullyConnectedPatch
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
            else if let layer = model.layers[num_layer] as? LayerNormSeq
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                layer.weightsCPU = weightsTmp + biases
            }
            else if let flLayer = model.layers[num_layer] as? FullyConnected
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
        }
        
        return model
    }
}

/// Generic model to test against PyTorch.
class ModelTestAutoEncoder
{
    ///
    /// Load weights in the model.
    ///
    /// - Parameters:
    ///     - model: The model.
    ///     - weights: The weights.
    ///
    static func initWeights(model: Model, weights: [[Float]])
    {
        // Apply weights on the `GrAIdient` model's layers.
        var cur = 0
        for num_layer in 0..<model.layers.count
        {
            // Load weights and biases.
            if let convLayer = model.layers[num_layer] as? Convolution2D
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                convLayer.weightsCPU = weightsTmp + biases
            }
            // Load weights and biases.
            else if let flLayer = model.layers[num_layer] as? FullyConnected
            {
                let weightsTmp: [Float] = weights[cur]
                cur += 1
                let biases: [Float] = weights[cur]
                cur += 1
                
                flLayer.weightsCPU = weightsTmp + biases
            }
        }
    }
}

/// Model to test against PyTorch.
class ModelTestAutoEncoder1: ModelTestAutoEncoder
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Convolution
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestAutoEncoder1", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 5, stride: 2,
            activation: nil,
            biases: true, bn: false,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 3, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_auto_encoder1_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}

/// Model to test against PyTorch.
class ModelTestAutoEncoder2: ModelTestAutoEncoder
{
    ///
    /// Create the model and import weights from PyTorch.
    ///
    /// Principle features:
    ///   - Convolution
    ///   - Deconvolution
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTestAutoEncoder2", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 5, stride: 2,
            activation: nil,
            biases: true, bn: false,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 3, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_auto_encoder2_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        super.initWeights(model: model, weights: weights)
        
        return model
    }
}
