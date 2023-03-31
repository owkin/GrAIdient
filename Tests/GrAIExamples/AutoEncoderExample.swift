//
// AutoEncoderExample.swift
// GrAIExamples
//
// Created by Aurélien PEDEN on 23/03/2023.
//

import XCTest
import GrAIdient

/// Test that we can train a simple Auto Encoder model on the CIFAR dataset.
final class AutoEncoderExample: XCTestCase
{
    /// Directory to dump outputs from the tests.
    let _outputDir = NSTemporaryDirectory()
    
    /// Batch size of data.
    let _batchSize = 64
    
    /// Size of one image (height and width are the same).
    let _size = 32
    
    /// Mean of the preprocessing to apply to data.
    let _mean = (125.3, 123.0, 113.9)
    /// Deviation of the preprocessing to apply to data.
    let _std = (63.0, 62.1, 66.7)
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
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
    /// Build an encoder branch.
    ///
    /// - Parameter params: Contextual parameters linking to the model.
    /// - Returns: A tuple of layers at different image resolutions.
    ///
    func _buildEncoder(params: GrAI.Model.Params)
        -> (Layer2D, Layer2D, Layer2D, Layer2D, Layer2D)
    {
        var layer, layer1, layer2, layer3, layer4: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: _size, height: _size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 2,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        layer4 = layer
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 2,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        layer3 = layer
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 2,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        layer2 = layer
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 2,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        layer1 = layer
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 2,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        return (layer, layer1, layer2, layer3, layer4)
    }
    
    ///
    /// Build a UNet like decoder branch.
    ///
    /// - Parameters:
    ///     - layersPrev: A tuple of layers at different image resolutions.
    ///     - params: Contextual parameters linking to the model.
    /// - Returns: The last layer of the decoder branch.
    ///
    func _buildUNetDecoder(
        layersPrev: (Layer2D, Layer2D, Layer2D, Layer2D, Layer2D),
        params: GrAI.Model.Params) -> Layer2D
    {
        var (layer, layer1, layer2, layer3, layer4) = layersPrev
        
        layer = Deconvolution2D(
            layerPrev: layer, size: 2, nbChannels: 8, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        layer = Concat2D(
            layersPrev: [layer1, layer],
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer, size: 2, nbChannels: 8, stride: 2,
            activation: nil, biases: true, bn: true,
            params: params
        )
        layer = Concat2D(
            layersPrev: [layer2, layer],
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer, size: 2, nbChannels: 8, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        layer = Concat2D(
            layersPrev: [layer3, layer],
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer, size: 2, nbChannels: 8, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        layer = Concat2D(
            layersPrev: [layer4, layer],
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: false,
            params: params
        )
        
        layer = Deconvolution2D(
            layerPrev: layer, size: 2, nbChannels: 8, stride: 2,
            activation: nil, biases: true, bn: false,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 3, stride: 1,
            activation: Sigmoid.str, biases: true, bn: false,
            params: params
        )
        return layer
    }
    
    ///
    /// Build a style controler branch.
    ///
    /// - Parameters:
    ///     - layersPrev: A tuple of layers at different image resolutions.
    ///     - params: Contextual parameters linking to the model.
    /// - Returns: The last layer of the style branch.
    ///
    func _buildStyleMapping(
        layersPrev: (Layer2D, Layer2D, Layer2D, Layer2D, Layer2D),
        params: GrAI.Model.Params) -> Layer1D
    {
        let (layer1, layer2, layer3, layer4, layer5) = layersPrev
        
        var layer: Layer1D = Concat1D(
            layersPrev: [
                AvgPool2D(layerPrev: layer1, params: params),
                AvgPool2D(layerPrev: layer2, params: params),
                AvgPool2D(layerPrev: layer3, params: params),
                AvgPool2D(layerPrev: layer4, params: params),
                AvgPool2D(layerPrev: layer5, params: params)
            ],
            params: params
        )
    
        for _ in 0..<8
        {
            layer = FullyConnected(
                layerPrev: layer, nbNeurons: 8,
                activation: ReLU.str, biases: true,
                params: params
            )
        }
        return layer
    }
    
    ///
    /// Build a StyleGAN like decoder branch.
    ///
    /// - Parameters:
    ///     - style: The last layer of the style branch.
    ///     - params: Contextual parameters linking to the model.
    /// - Returns: The last layer of the decoder branch.
    ///
    func _buildStyleDecoder(style: Layer1D, params: GrAI.Model.Params)
        -> Layer2D
    {
        var layer: Layer2D
        layer = Constant2D(
            nbChannels: 8, height: 4, width: 4,
            params: params
        )
        
        layer = AdaIN(
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
        layer = AdaIN(
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
        
        for _ in 0..<5
        {
            layer = ResizeBilinearCrop(
                layerPrev: layer, scalesList: [2],
                params: params
            )
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
                activation: ReLU.str, biases: true, bn: false,
                params: params
            )
            layer = AdaIN(
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
            layer = AdaIN(
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
        return layer
    }
    
    /// Model architecture choice.
    enum ModelClass
    {
        case UNet, Style
    }
    
    ///
    /// Build the final model.
    ///
    /// - Parameter modelType: The model to build.
    /// - Returns: The model built.
    ///
    func _buildModel(modelType: ModelClass) -> Model
    {
        // Create the context to build a graph of layers where
        // there is no previous model dependency: layer id starts at 0.
        let context = ModelContext(name: "AutoEncoder", models: [])
        let params = GrAI.Model.Params(context: context)
        
        let layersPrev = _buildEncoder(params: params)
        
        var layer: Layer2D
        switch modelType
        {
        case .Style:
            layer = _buildStyleDecoder(
                style: _buildStyleMapping(
                    layersPrev: layersPrev,
                    params: params
                ),
                params: params
            )
        case .UNet:
            layer = _buildUNetDecoder(
                layersPrev: layersPrev,
                params: params
            )
        }
        
        _ = MSE2D(layerPrev: layer, params: params)
        return Model(model: context.model, modelsPrev: [])
    }
    
    ///
    /// Train the model.
    ///
    /// - Parameter model: The model to train.
    ///
    func _trainModel(model: Model)
    {
        let cifar8 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTrain8",
            size: _size
        )
        
        // Get optimizer parameters for iterating over batch size elements.
        let params = _getOptimizerParams(nbLoops: _batchSize)
        
        cifar8.initSamples(batchSize: _batchSize)
        
        // Keep a subset of the dataset to have a quicker training.
        cifar8.keep(1000)
        
        // Small trick to force full batches throughout the training:
        // this enables us to set the ground truth once and for all.
        let nbWholeBatches =
            cifar8.nbSamples / cifar8.batchSize * cifar8.batchSize
        cifar8.keep(nbWholeBatches)
        
        // Initialize for training.
        model.initialize(params: params, phase: .Training)
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        let lastLayer: MSE2D = model.layers.last as! MSE2D
        
        let nbEpochs = 5
        for epoch in 0..<nbEpochs
        {
            print("EPOCH \(epoch)/\(nbEpochs-1).")
            cifar8.shuffle()
            
            for step in 0..<cifar8.nbLoops
            {
                let samples8 = cifar8.getSamples()!
                let samples = samples8
                
                if samples.count != _batchSize
                {
                    fatalError("Unreachable.")
                }
                
                // Pre processing.
                let data = preprocess(
                    samples,
                    height: _size,
                    width: _size,
                    mean: _mean,
                    std: _std,
                    imageFormat: .Neuron
                )
                
                // Reset gradient validity for backward pass
                // and update the batch size (although here it stays the same).
                model.updateKernel(batchSize: _batchSize)
                
                // Set data.
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: _batchSize,
                    format: .Neuron
                )
                
                // Forward.
                try! model.forward()
                
                // Apply loss derivative.
                try! lastLayer.lossDerivativeGPU(
                    data,
                    batchSize: _batchSize,
                    format: .Neuron
                )
                
                // Backward.
                try! model.backward()
                
                // Update weights.
                try! model.update()
                
                // Get loss result.
                // Note that backward is explicitly
                // enabled by `applyGradient` whereas `getLoss` is
                // just an indicator.
                let loss = try! lastLayer.getLossGPU(
                    data,
                    batchSize: _batchSize,
                    format: .Neuron
                )
                print("Step \(step)/\(cifar8.nbLoops-1): \(sqrt(loss)).")
                
                // Update internal step.
                // This is not mandatory except if we used another
                // optimizer scheduler: see `_getOptimizerParams`.
                model.incStep()
            }
        }
    }
    
    /// Test1: dump CIFAR train and test datasets for labels 8 and 5.
    func test1_DumpDataset()
    {
        CIFAR.dumpTrain(
            datasetPath: _outputDir + "/datasetTrain8",
            label: 8,
            size: _size
        )
    }
    
    /// Test2: train a simple UNet like auto encoder model.
    func test2_TrainUNetModel()
    {
        // Build a model with randomly initialized weights.
        let model = _buildModel(modelType: .UNet)
        
        // Train model.
        _trainModel(model: model)
    }
    
    /// Test3: train a simple StyleGAN like auto encoder model.
    func test3_TrainStyleModel()
    {
        // Build a model with randomly initialized weights.
        let model = _buildModel(modelType: .Style)
        
        // Train model.
        _trainModel(model: model)
    }
}
