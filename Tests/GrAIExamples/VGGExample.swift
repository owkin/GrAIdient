//
// VGGExample.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 17/10/2022.
//

import XCTest
import GrAIdient

/// Train a simple VGG model on the CIFAR dataset.
final class VGGExample: XCTestCase
{
    /// Directory to dump outputs from the tests.
    let _outputDir = NSTemporaryDirectory()
    
    /// Batch size of data.
    let _batchSize = 64
    /// Size of one image (height and width are the same).
    let _size = 32
    
    /// Mean of the preprocessing to apply to data.
    let _mean: (Float, Float, Float) = (123.675, 116.28, 103.53)
    /// Deviation of the preprocessing to apply to data.
    let _std: (Float, Float, Float) = (58.395, 57.12, 57.375)
    
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
    /// Build a simple model.
    ///
    /// - Parameter bn: Whether to use batch normalization or not.
    /// - Returns: The model built.
    ///
    func _buildModel(bn: Bool) -> Model
    {
        // Create the context to build a graph of layers where
        // there is no previous model dependency: layer id starts at 0.
        let context = ModelContext(name: "VGG", models: [])
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: _size, height: _size,
            params: params
        )
        
        layer = MaxPool2D(layerPrev: layer, size: 2, stride: 2, params: params)
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(layerPrev: layer, size: 2, stride: 2, params: params)
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(layerPrev: layer, size: 2, stride: 2, params: params)
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(layerPrev: layer, size: 2, stride: 2, params: params)
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(layerPrev: layer, params: params)

        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: ReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
        
        // Retrieve base model in the context and initialize a
        // real model (with `layerPrev` links updated).
        return Model(model: context.model, modelsPrev: [])
    }
    
    ///
    /// Load a model from the disk.
    ///
    /// - Parameter modelPath: The model path on the disk.
    /// - Returns: the model loaded.
    ///
    func _loadModel(_ modelPath: String) -> Model
    {
        // Load model from the disk.
        let data = try! Data(
            contentsOf: URL(fileURLWithPath: modelPath)
        )
        
        // Decode it as a base model
        // (model where `layerPrev` links are not initialized).
        let baseModel = try! PropertyListDecoder().decode(
            BaseModel.self,
            from: data
        )
        
        // Create a model with initialized links
        // with no previous model dependencies.
        let vgg = Model(model: baseModel, modelsPrev: [])
        return vgg
    }
    
    ///
    /// Evaluate a model on the testing CIFAR dataset.
    ///
    /// - Parameter model: The model to evaluate.
    /// - Returns: The ratio (in percent) of good predictions.
    ///
    func _evaluateModel(_ model: Model) -> Int
    {
        let cifar8 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTest8",
            size: _size
        )
        let cifar5 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTest5",
            size: _size
        )
        
        cifar8.initSamples(batchSize: _batchSize)
        cifar5.initSamples(batchSize: _batchSize)
        
        // We keep a subset of the dataset to have a quicker evaluation.
        cifar8.keep(100)
        cifar5.keep(100)
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        let lastLayer: MSE1D = model.layers.last as! MSE1D
        
        var nbRight = 0
        var nbTotal = 0
        
        var sampler: CIFAR = cifar8
        var samples = sampler.getSamples()
        
        for label in 0...1
        {
            if samples == nil
            {
                sampler = cifar5
                samples = sampler.getSamples()
            }
            
            while samples != nil
            {
                // Pre processing.
                let data = preprocess(
                    samples!,
                    height: _size,
                    width: _size,
                    mean: _mean,
                    std: _std,
                    imageFormat: .Neuron
                )
                
                // Update internal batch size.
                model.updateKernel(batchSize: samples!.count)
                
                // Set data.
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: samples!.count,
                    nbChannels: 3, height: _size, width: _size,
                    format: .Neuron
                )
                
                // Forward.
                try! model.forward()
                
                for elem in 0..<samples!.count
                {
                    // Get result: 1 neuron.
                    let result: Float = lastLayer.getOutsGPU(elem: elem)[0]
                    if label == 0 && result < 0.5
                    {
                        nbRight += 1
                    }
                    else if label == 1 && result >= 0.5
                    {
                        nbRight += 1
                    }
                    nbTotal += 1
                }
                
                samples = sampler.getSamples()
            }
        }
        
        let ratio = Int(Double(nbRight) / Double(nbTotal) * 100)
        return ratio
    }
    
    /// Test1: dump CIFAR train and test datasets for labels 8 and 5.
    func test1_DumpDataset()
    {
        CIFAR.dumpTrain(
            datasetPath: _outputDir + "/datasetTrain8",
            label: 8,
            size: _size
        )
        CIFAR.dumpTrain(
            datasetPath: _outputDir + "/datasetTrain5",
            label: 5,
            size: _size
        )
        CIFAR.dumpTest(
            datasetPath: _outputDir + "/datasetTest8",
            label: 8,
            size: _size
        )
        CIFAR.dumpTest(
            datasetPath: _outputDir + "/datasetTest5",
            label: 5,
            size: _size
        )
    }
    
    /// Test2: dump CIFAR images for labels 8 and 5.
    func test2_DumpImages()
    {
        let batchSize = 16
        let cifar8 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTest8",
            size: _size
        )
        let cifar5 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTest5",
            size: _size
        )
        cifar8.initSamples(batchSize: batchSize)
        cifar5.initSamples(batchSize: batchSize)
        
        let samples8 = cifar8.getSamples()!
        let samples5 = cifar5.getSamples()!
        
        let pixels8 = Image.toRGB(samples8, width: _size, height: _size)
        let pixels5 = Image.toRGB(samples5, width: _size, height: _size)
        
        for elem in 0..<batchSize
        {
            var image = Image.buildImage(
                pixels: pixels8[elem], width: _size, height: _size
            )
            try! image.save(
                url: URL(fileURLWithPath: _outputDir + "CIFAR8_\(elem).png")
            )
            
            image = Image.buildImage(
                pixels: pixels5[elem], width: _size, height: _size
            )
            try! image.save(
                url: URL(fileURLWithPath: _outputDir + "CIFAR5_\(elem).png")
            )
        }
    }
    
    /// Test3: test that an untrained model makes bad predictions.
    func test3_UntrainedModel()
    {
        // Build a model with randomly initialized weights.
        let vgg = _buildModel(bn: true)
        
        // Initialize for inference.
        vgg.initKernel(phase: .Inference)
        
        // Evaluate model on CIFAR testing dataset.
        let ratio = _evaluateModel(vgg)
        
        print(
            "Ratio of good predictions: \(ratio)%."
        )
        
        // Encode the model.
        let encoder = PropertyListEncoder()
        let data = try! encoder.encode(vgg)
        
        // Save it to the disk.
        try! data.write(
            to: URL(fileURLWithPath: _outputDir + "/vgg1.plist")
        )
    }
    
    /// Test4: train a simple model.
    func test4_TrainVGG()
    {
        let cifar8 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTrain8",
            size: _size
        )
        let cifar5 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTrain5",
            size: _size
        )
        
        // Get optimizer parameters for iterating over batch size elements.
        let params = _getOptimizerParams(nbLoops: _batchSize)
        
        // A batch will in fact be composed of half elements coming from
        // cifar8 (ships => label: 0) and half elements coming from
        // cifar5 (dogs => label: 1).
        cifar8.initSamples(batchSize: _batchSize / 2)
        cifar5.initSamples(batchSize: _batchSize / 2)
        
        // Keep a subset of the dataset to have a quicker training.
        cifar8.keep(500)
        cifar5.keep(500)
        
        // Small trick to force full batches throughout the training:
        // this enables us to set the ground truth once and for all.
        let nbWholeBatches =
            cifar8.nbSamples / cifar8.batchSize * cifar8.batchSize
        cifar8.keep(nbWholeBatches)
        cifar5.keep(nbWholeBatches)
        
        // Load previous model from the disk.
        let vgg = _loadModel(_outputDir + "vgg1.plist")
        
        // Initialize for training.
        vgg.initialize(params: params, phase: .Training)
        
        let firstLayer: Input2D = vgg.layers.first as! Input2D
        let lastLayer: MSE1D = vgg.layers.last as! MSE1D
        
        // Initialize the ground truth once and for all.
        let groundTruth = MetalSharedBuffer<UInt16>(_batchSize, deviceID: 0)
        let buffer = groundTruth.buffer
        for elem in 0..<_batchSize / 2
        {
            buffer[elem] = 0.0
        }
        for elem in _batchSize / 2..<_batchSize
        {
            buffer[elem] = 1.0
        }
        MetalKernel.get.upload([groundTruth])
        
        let nbEpochs = 5
        for epoch in 0..<nbEpochs
        {
            print("EPOCH \(epoch + 1)/\(nbEpochs).")
            cifar8.shuffle()
            cifar5.shuffle()
            
            for step in 0..<cifar8.nbLoops
            {
                let samples8 = cifar8.getSamples()!
                let samples5 = cifar5.getSamples()!
                let samples = samples8 + samples5
                
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
                vgg.updateKernel(batchSize: _batchSize)
                
                // Set data.
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: _batchSize,
                    nbChannels: 3, height: _size, width: _size,
                    format: .Neuron
                )
                
                // Forward.
                try! vgg.forward()
                
                // Apply loss derivative.
                try! lastLayer.lossDerivativeGPU(
                    groundTruth,
                    batchSize: _batchSize,
                    nbNeurons: 1
                )
                
                // Backward.
                try! vgg.backward()
                
                // Update weights.
                try! vgg.update()
                
                // Get loss result.
                // Note that backward is explicitly
                // enabled by `applyGradient` whereas `getLoss` is
                // just an indicator.
                let loss = try! lastLayer.getLossGPU(
                    groundTruth,
                    batchSize: _batchSize,
                    nbNeurons: 1
                )
                print("Step \(step + 1)/\(cifar8.nbLoops): \(sqrt(loss)).")
                
                // Update internal step.
                // This is not mandatory except if we used another
                // optimizer scheduler: see `_getOptimizerParams`.
                vgg.incStep()
            }
        }
    
        // Encode the trained model.
        let encoder = PropertyListEncoder()
        let data = try! encoder.encode(vgg)
        
        // Save it to the disk.
        try! data.write(
            to: URL(fileURLWithPath: _outputDir + "/vgg2.plist")
        )
    }
    
    /// Test5: test that the previous trained model makes better predictions than the untrained model.
    func test5_CompareModels()
    {
        // Load previous model from the disk.
        let vgg1 = _loadModel(_outputDir + "/vgg1.plist")
        let vgg2 = _loadModel(_outputDir + "/vgg2.plist")
        
        // Initialize for inference.
        vgg1.initKernel(phase: .Inference)
        vgg2.initKernel(phase: .Inference)
        
        // Evaluate model on CIFAR testing dataset.
        let ratio1 = _evaluateModel(vgg1)
        let ratio2 = _evaluateModel(vgg2)
        
        print(
            "Ratio of good predictions before training: \(ratio1)%."
        )
        print(
            "Ratio of good predictions after training: \(ratio2)%."
        )
    }
}
