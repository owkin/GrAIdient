//
// AutoEncoderExample.swift
// GrAIExamples
//
// Created by Aurélien PEDEN on 23/03/2023.
// Modified by Jean-François Reboud on 21/05/2023.
//

import XCTest
import GrAIdient

/// Train a simple Auto Encoder model on the CIFAR dataset.
final class AutoEncoderExample: XCTestCase
{
    /// Directory to dump outputs from the tests.
    let _outputDir = NSTemporaryDirectory()
    
    /// Batch size of data.
    let _batchSize = 16
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
    }
    
    ///
    /// Get optimizer parameters for model training.
    ///
    /// - Parameter nbLoops: Number of steps per epoch.
    /// - Returns: The optimizer parameters.
    ///
    func _getOptimizerParams(nbLoops: Int) -> GrAI.Optimizer.Params
    {
        var optimizerParams = GrAI.Optimizer.Params()
        optimizerParams.nbLoops = nbLoops
        
        // Simple optimizer scheduler: always the same optimizer during
        // the training.
        optimizerParams.optimizer = ConstEpochsScheduler(
            GrAI.Optimizer.Class.AdamRectified
        )
        
        // Simple variable scheduler: always the same variable during
        // the training.
        optimizerParams.variables["alpha"] = ConstEpochsVar(
            value: ConstVal(1e-3)
        )
        optimizerParams.variables["lambda"] = ConstEpochsVar(
            value: ConstVal(1e-6)
        )
        
        // Other schedulers can be built thanks to `GrAI.Optimizer.Params`.
        return optimizerParams
    }
    
    ///
    /// Build an encoder branch with `nbBlock` blocks of dimension reduction (factor of 2).
    ///
    /// - Parameters:
    ///     - size: Size of one image (height and width are the same) after resize.
    ///     - nbBlocks: Number of reduction blocks.
    ///     - params: Contextual parameters linking to the model.
    /// - Returns: A list of layers at different image resolutions.
    ///
    func _buildEncoder(
        size: Int,
        nbBlocks: Int,
        params: GrAI.Model.Params) -> [Layer2D]
    {
        var layer: Layer2D
        var layers = [Layer2D]()
        
        layer = Input2D(
            nbChannels: 3,
            width: size, height: size,
            params: params
        )
        
        for _ in 0..<nbBlocks
        {
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 8, stride: 2,
                activation: ReLU.str, biases: true, bn: false,
                params: params
            )
            layers.insert(layer, at: 0)
        }
        return layers
    }
    
    ///
    /// Build a UNet like decoder branch.
    ///
    /// - Parameters:
    ///     - layersPrev: A list of layers at different image resolutions.
    ///     - params: Contextual parameters linking to the model.
    /// - Returns: The last layer of the decoder branch.
    ///
    func _buildUNetDecoder(
        layersPrev: [Layer2D],
        params: GrAI.Model.Params) -> Layer2D
    {
        var layer: Layer2D = layersPrev.first!
        var numLayer = 0
        
        while numLayer < layersPrev.count
        {
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 8, stride: 2,
                activation: nil, biases: true, bn: false,
                params: params
            )
            
            if numLayer + 1 < layersPrev.count
            {
                layer = try! Concat2D(
                    layersPrev: [layersPrev[numLayer + 1], layer],
                    params: params
                )
                layer = Convolution2D(
                    layerPrev: layer,
                    size: 3, nbChannels: 8, stride: 1,
                    activation: ReLU.str, biases: true, bn: false,
                    params: params
                )
            }
            else
            {
                layer = Convolution2D(
                    layerPrev: layer,
                    size: 3, nbChannels: 3, stride: 1,
                    activation: Sigmoid.str, biases: true, bn: false,
                    params: params
                )
            }
            numLayer += 1
        }
        return layer
    }
    
    ///
    /// Build a style controller branch.
    ///
    /// - Parameters:
    ///     - layersPrev: A tuple of layers at different image resolutions.
    ///     - params: Contextual parameters linking to the model.
    /// - Returns: The last layer of the style branch.
    ///
    func _buildStyleMapping(
        layersPrev: [Layer2D],
        params: GrAI.Model.Params) -> Layer1D
    {
        var layers = [Layer1D]()
        for layerPrev in layersPrev
        {
            layers.append(
                AvgPool2D(layerPrev: layerPrev, params: params)
            )
        }
        var layer: Layer1D = Concat1D(
            layersPrev: layers, params: params
        )
        for _ in 0..<8
        {
            layer = try! FullyConnected(
                layerPrev: layer, nbNeurons: 8,
                activation: ReLU.str, biases: true,
                params: params
            )
        }
        return layer
    }
    
    ///
    /// Build a StyleGAN like decoder branch with `nbBlock` blocks
    /// of dimension augmentation (factor of 2).
    ///
    /// - Parameters:
    ///     - nbBlocks: Number of augmentation blocks.
    ///     - style: The last layer of the style branch.
    ///     - params: Contextual parameters linking to the model.
    /// - Returns: The last layer of the decoder branch.
    ///
    func _buildStyleDecoder(
        nbBlocks: Int,
        style: Layer1D,
        params: GrAI.Model.Params) -> Layer2D
    {
        var layer: Layer2D
        layer = Constant2D(
            nbChannels: 8, height: 2, width: 2,
            params: params
        )
        layer = try! AdaIN(
            layersPrev: [
                layer,
                FullyConnected(
                    layerPrev: style, nbNeurons: 2 * 8,
                    activation: ReLU.str, biases: true,
                    params: params
                )
            ],
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        layer = try! AdaIN(
            layersPrev: [
                layer,
                FullyConnected(
                    layerPrev: style, nbNeurons: 2 * 8,
                    activation: ReLU.str, biases: true,
                    params: params
                )
            ],
            params: params
        )
        
        for _ in 0..<nbBlocks-1
        {
            layer = try! ResizeBilinearPad(
                layerPrev: layer, scalesList: [2], padValue: 0.0,
                params: params
            )
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
                activation: ReLU.str, biases: true, bn: false,
                params: params
            )
            layer = try! AdaIN(
                layersPrev: [
                    layer,
                    FullyConnected(
                        layerPrev: style, nbNeurons: 2 * 8,
                        activation: ReLU.str, biases: true,
                        params: params
                    )
                ],
                params: params
            )
        }
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 3, stride: 1,
            activation: Sigmoid.str, biases: true, bn: false,
            params: params
        )
        return layer
    }
    
    /// Model architecture choice.
    enum ModelClass
    {
        case UNet, Style
    }
    
    ///
    /// Build the final model that is composed of `nbBlocks` blocks of dimension reduction
    /// followed by the same number of blocks of dimension augmentation.
    ///
    /// - Parameters:
    ///     - nbBlocks: Number blocks (reduction, augmentation, factor of 2).
    ///     - size: Size of one image (height and width are the same) after resize.
    ///     - modelType: The model to build.
    /// - Returns: The model built.
    ///
    func _buildModel(
        modelType: ModelClass,
        size: Int,
        nbBlocks: Int) -> Model
    {
        // Create the context to build a graph of layers where
        // there is no previous model dependency: layer id starts at 0.
        let context = ModelContext(name: "AutoEncoder", models: [])
        let params = GrAI.Model.Params(context: context)
        
        let layersPrev = _buildEncoder(
            size: size,
            nbBlocks: nbBlocks,
            params: params
        )
        
        switch modelType
        {
        case .Style:
            _ = _buildStyleDecoder(
                nbBlocks: nbBlocks,
                style: _buildStyleMapping(
                    layersPrev: layersPrev,
                    params: params
                ),
                params: params
            )
        case .UNet:
            _ = _buildUNetDecoder(
                layersPrev: layersPrev,
                params: params
            )
        }
        return Model(model: context.model, modelsPrev: [])
    }
    
    ///
    /// Train the model.
    ///
    /// - Parameters:
    ///     - model: The model to train.
    ///     - size: Size of one image (height and width are the same) after resize.
    ///
    func _trainModel(model: Model, size: Int)
    {
        let trainer = try! CIFARAutoEncoderTrainer(
            model: model, size: size
        )
        trainer.run(
            batchSize: _batchSize,
            label: 8,
            nbEpochs: 2,
            keep: 1000
        )
    }
    
    /// Test1: train a simple auto encoder model.
    func test1_TrainSimpleModel()
    {
        let size = 32
        
        // Build a model with randomly initialized weights.
        let model = SimpleAutoEncoder.build(size)
        
        // Train model.
        _trainModel(
            model: model,
            size: size
        )
    }
    
    /// Test2: train a UNet like auto encoder model.
    func test2_TrainUNetModel()
    {
        let nbBlocks = 5
        let size = min(Int(pow(2.0, Double(nbBlocks))), 32)
        
        // Build a model with randomly initialized weights.
        let model = _buildModel(
            modelType: .UNet,
            size: size,
            nbBlocks: nbBlocks
        )
        
        // Train model.
        _trainModel(
            model: model,
            size: size
        )
    }
    
    /// Test3: train a StyleGAN like auto encoder model.
    func test3_TrainStyleModel()
    {
        let nbBlocks = 5
        let size = min(Int(pow(2.0, Double(nbBlocks))), 32)
        
        // Build a model with randomly initialized weights.
        let model = _buildModel(
            modelType: .Style,
            size: size,
            nbBlocks: nbBlocks
        )
        
        // Train model.
        _trainModel(
            model: model,
            size: size
        )
    }
}
