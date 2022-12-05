//
// Model.swift
// MATorchTests
//
// Created by Jean-François Reboud on 19/10/2022.
//

import MAKit
import PythonKit

/// Model to test against PyTorch.
class ModelTest1
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
        let context = ModelContext(name: "ModelTest1", curID: 0)
        let params = MAKit.Model.Params(context: context)
        
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
        let data = pythonLib.load_test1_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        
        // Apply weights on the `MAKit` model's layers.
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
class ModelTest2
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
        let context = ModelContext(name: "ModelTest2", curID: 0)
        let params = MAKit.Model.Params(context: context)
        
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
        let data = pythonLib.load_test2_weights()
        
        let weights = [[Float]](data.tuple2.0)!
        
        // Apply weights on the `MAKit` model's layers.
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
class ModelTest3
{
    ///
    /// Create the model.
    ///
    /// Principle features:
    ///   - 2D Frequences & scale
    ///   - IRDFT
    ///   - (Decorrelate color)
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "ModelTest2", curID: 0)
        let params = MAKit.Model.Params(context: context)
        
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
        
        /*
         Introduces float rounding approximation issues!
         layer = DecorrelateRGB(
            layerPrev: layer,
            correlation: [
                0.26, 0.26, 0.27,
                0.09, 0.00, -0.09,
                0.02, -0.05, 0.03
            ].map { $0 / 0.4619524 },
            params: params
        )
        */
        
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
