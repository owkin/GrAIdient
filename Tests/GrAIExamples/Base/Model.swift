//
// Model.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 21/05/2023.
//

import GrAIdient
import PythonKit

/// Simple auto encoder model.
class SimpleAutoEncoder
{
    ///
    /// Create a simple auto encoder model and import weights from PyTorch.
    ///
    /// - Parameter size: The size of the input data.
    /// - Returns: The built model.
    ///
    static func build(_ size: Int) -> Model
    {
        let context = ModelContext(name: "SimpleAutoEncoder", curID: 0)
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
            size: 3, nbChannels: 12, stride: 2,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 24, stride: 2,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 48, stride: 2,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 24, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 12, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        layer = Deconvolution2D(
            layerPrev: layer,
            size: 2, nbChannels: 3, stride: 2,
            activation: Sigmoid.str, biases: true, bn: false,
            params: params
        )
        
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_simple_auto_encoder_weights()
        
        let weightsNumpy = [PythonObject](data.tuple2.0)!
        var weights = [[Float]]()
        for weightsNP in weightsNumpy
        {
            if let weightsTmp = Array<Float>(numpy: weightsNP)
            {
                weights.append(weightsTmp)
            }
        }
        
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
        }
        return model
    }
}
