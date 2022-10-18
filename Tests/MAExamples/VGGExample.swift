//
// VGGExample.swift
// MAExamples
//
// Created by Jean-François Reboud on 17/10/2022.
//

import XCTest
import MAKit

/// Test that we can train a simple model on the CIFAR dataset.
final class VGGExample: XCTestCase
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
        MAKit.Opti.GPU = true
    }
    
    ///
    /// Get optimizer parameters for model training.
    ///
    /// - Parameter nbLoops: Number of steps per epoch.
    /// - Returns: The optimizer parameters.
    ///
    func _getOptimizerParams(nbLoops: Int) -> MAKit.Optimizer.Params
    {
        var optimizerParams = MAKit.Optimizer.Params()
        optimizerParams.nbLoops = nbLoops
        
        optimizerParams.optimizer = ConstEpochsScheduler(
            MAKit.Optimizer.Class.AdamRectified
        )
        
        optimizerParams.variables["alpha"] = ConstEpochsVar(
            value: ConstVal(1e-3)
        )
        optimizerParams.variables["lambda"] = ConstEpochsVar(
            value: ConstVal(1e-6)
        )
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
        let context = ModelContext(name: "VGG", models: [])
        let params = MAKit.Model.Params(context: context)
        
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

        head = FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: ReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
        
        return Model(model: context.model, modelsPrev: [])
    }
    
    ///
    /// Evaluate a model on the testing CIFAR dataset.
    ///
    /// - Parameter model: The model to evaluate.
    /// - Returns: (Number of correct predictions, Total number of tests)/
    ///
    func _evaluateModel(_ model: Model) -> (Int, Int)
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
                let data = preprocess(
                    samples!,
                    height: _size,
                    width: _size,
                    mean: _mean,
                    std: _std,
                    imageFormat: .Neuron
                )
                
                model.updateKernel(batchSize: samples!.count)
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: samples!.count,
                    format: .Neuron
                )
                
                try! model.forward()
                
                for elem in 0..<samples!.count
                {
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
        
        return (nbRight, nbTotal)
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
    
    /// Test2: test that an untrained model makes bad predictions.
    func test2_UntrainedModel()
    {
        let vgg = _buildModel(bn: true)
        vgg.initKernel(phase: .Inference)
        
        let (nbRight, nbTotal) = _evaluateModel(vgg)
        
        let ratio = Int(Double(nbRight) / Double(nbTotal) * 100)
        print(
            "Ratio of good predictions: \(ratio)%."
        )
        XCTAssert(ratio < 60)
    }
    
    /// Test3: train a simple model.
    func test3_TrainVGG()
    {
        let cifar8 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTrain8",
            size: _size
        )
        let cifar5 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTrain5",
            size: _size
        )
        
        let params = _getOptimizerParams(nbLoops: _batchSize)
        cifar8.initSamples(batchSize: _batchSize / 2)
        cifar5.initSamples(batchSize: _batchSize / 2)
        cifar8.keep(500)
        cifar5.keep(500)
        
        let nbWholeBatches =
            cifar8.nbSamples / cifar8.batchSize * cifar8.batchSize
        cifar8.keep(nbWholeBatches)
        cifar5.keep(nbWholeBatches)
        
        let vgg = _buildModel(bn: true)
        vgg.initialize(params: params, phase: .Training)
        
        let firstLayer: Input2D = vgg.layers.first as! Input2D
        let lastLayer: MSE1D = vgg.layers.last as! MSE1D
        
        let groundTruth = MetalSharedBuffer<Float>(_batchSize, deviceID: 0)
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
            print("EPOCH \(epoch)/\(nbEpochs-1).")
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
                
                let data = preprocess(
                    samples,
                    height: _size,
                    width: _size,
                    mean: _mean,
                    std: _std,
                    imageFormat: .Neuron
                )
                
                vgg.updateKernel(batchSize: _batchSize)
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: _batchSize,
                    format: .Neuron
                )
                
                try! vgg.forward()
                try! lastLayer.applyGradientGPU(
                    groundTruth,
                    batchSize: _batchSize
                )
                try! vgg.backward()
                try! vgg.update()
                
                let loss = try! lastLayer.getLossGPU(
                    groundTruth,
                    batchSize: _batchSize
                )
                print("Step \(step)/\(cifar8.nbLoops-1): \(sqrt(loss)).")
                
                vgg.incStep()
            }
            vgg.incEpoch()
        }
    
        let encoder = PropertyListEncoder()
        let data = try! encoder.encode(vgg)
        try! data.write(
            to: URL(fileURLWithPath: _outputDir + "/vgg.plist")
        )
    }
    
    /// Test4: test that the previous trained model makes better predictions than the untrained model.
    func test4_TrainedModel()
    {
        let data = try! Data(
            contentsOf: URL(fileURLWithPath: _outputDir + "/vgg.plist")
        )
        let baseModel = try! PropertyListDecoder().decode(
            BaseModel.self,
            from: data
        )
        
        let vgg = Model(model: baseModel, modelsPrev: [])
        vgg.initKernel(phase: .Inference)
        
        let (nbRight, nbTotal) = _evaluateModel(vgg)
        
        let ratio = Int(Double(nbRight) / Double(nbTotal) * 100)
        print(
            "Ratio of good predictions: \(ratio)%."
        )
        XCTAssert(ratio >= 60)
    }
}
