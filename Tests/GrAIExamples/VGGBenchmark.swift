//
// VGGBenchmark.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 24/02/2024.
//

import XCTest
import GrAIdient

/// Benchmark time spent for training and evaluating a VGG model with fake data.
final class VGGBenchmark: XCTestCase
{
    /// Batch size of data.
    let _batchSize = 64
    /// Size of one image (height and width are the same).
    let _size = 224
    
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
    /// Build a simple model.
    ///
    /// - Parameter bn: Whether to use batch normalization or not.
    /// - Returns: The model built.
    ///
    func _buildModel(bn: Bool) -> Model
    {
        // Create the context to build a graph of layers where
        // there is no previous model dependency: layer id starts at 0.
        let context = ModelContext(name: "VGG16", models: [])
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: _size, height: _size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 64, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 64, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(
            layerPrev: layer, size: 2, stride: 2, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 128, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 128, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(
            layerPrev: layer, size: 2, stride: 2, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 256, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 256, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 256, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(
            layerPrev: layer, size: 2, stride: 2, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 512, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 512, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 512, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(
            layerPrev: layer, size: 2, stride: 2, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 512, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 512, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer,
            size: 3, nbChannels: 512, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(
            layerPrev: layer, size: 2, stride: 2, params: params
        )
        
        layer = AdaptiveAvgPool2D(layerPrev: layer, size: 7, params: params)
        
        var head: Layer1D = try! FullyConnected(
            layerPrev: layer,
            nbNeurons: 4096,
            activation: ReLU.str,
            biases: true,
            params: params
        )
        head = try! FullyConnected(
            layerPrev: head,
            nbNeurons: 4096,
            activation: ReLU.str,
            biases: true,
            params: params
        )
        head = try! FullyConnected(
            layerPrev: head,
            nbNeurons: 1,
            activation: ReLU.str,
            biases: true,
            params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
        
        // Retrieve base model in the context and initialize a
        // real model (with `layerPrev` links updated).
        let model = Model(model: context.model, modelsPrev: [])
        return model
    }
    
    /// Test: train a VGG model.
    func _test_TrainVGG()
    {
        // Get optimizer parameters for iterating over batch size elements.
        let params = _getOptimizerParams(nbLoops: _batchSize)
        
        // Build a model with randomly initialized weights.
        let vgg = _buildModel(bn: false)
        
        // Initialize for training.
        vgg.initialize(params: params, phase: .Training)
        
        let firstLayer: Input2D = vgg.layers.first as! Input2D
        let lastLayer: MSE1D = vgg.layers.last as! MSE1D
        
        // Initialize the ground truth once and for all.
        let groundTruth = FloatBuffer(
            nbElems: _batchSize, deviceID: 0, shared: true
        )
        var gtBuffer = [Float](repeating: 0.0, count: _batchSize)
        for elem in 0..<_batchSize / 2
        {
            gtBuffer[elem] = 0.0
        }
        for elem in _batchSize / 2..<_batchSize
        {
            gtBuffer[elem] = 1.0
        }
        groundTruth.initialize(array: &gtBuffer)
        
        // Initialize data once and for all.
        let data = FloatBuffer(
            nbElems: _batchSize * 3 * _size * _size, deviceID: 0, shared: true
        )
        var dataBuffer = [Float](
            repeating: 0.0, count: _batchSize * 3 * _size * _size
        )
        for i in 0..<_batchSize * 3 * _size * _size
        {
            dataBuffer[i] = Float.random(in: -1..<1)
        }
        data.initialize(array: &dataBuffer)
        
        let nbEpochs = 1
        let nbSteps = 20
        for epoch in 0..<nbEpochs
        {
            print("EPOCH \(epoch + 1)/\(nbEpochs).")
            
            let start1 = Date()
            for step in 0..<nbSteps
            {
                let start2 = Date()
                
                // Reset gradient validity for backward pass
                // and update the batch size (although here it stays the same).
                vgg.updateKernel(batchSize: _batchSize)
                
                // Set data.
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: _batchSize,
                    nbChannels: 3, 
                    height: _size,
                    width: _size
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
                
                // Update internal step.
                // This is not mandatory except if we used another
                // optimizer scheduler: see `_getOptimizerParams`.
                vgg.incStep()
                
                let end2 = Date()
                let timeSpent = end2.timeIntervalSince(start2)
                print("Step \(step + 1)/\(nbSteps): " +
                      "\(sqrt(loss)) in \(timeSpent)s.")
            }
            
            let end1 = Date()
            let timeSpent = end1.timeIntervalSince(start1)
            print("Epoch \(epoch + 1), time spent: \(timeSpent)s.")
        }
    }
    
    /// Test: evaluate a VGG model.
    func _test_EvalTransformer()
    {
        // Build a model with randomly initialized weights.
        let vgg = _buildModel(bn: true)
        
        // Initialize for inference.
        vgg.initKernel(phase: .Inference)
        
        let firstLayer: Input2D = vgg.layers.first as! Input2D
        let lastLayer: MSE1D = vgg.layers.last as! MSE1D
        
        // Initialize the ground truth once and for all.
        let groundTruth = FloatBuffer(
            nbElems: _batchSize, deviceID: 0, shared: true
        )
        var gtBuffer = [Float](repeating: 0.0, count: _batchSize)
        for elem in 0..<_batchSize / 2
        {
            gtBuffer[elem] = 0.0
        }
        for elem in _batchSize / 2..<_batchSize
        {
            gtBuffer[elem] = 1.0
        }
        groundTruth.initialize(array: &gtBuffer)
        
        // Initialize data once and for all.
        let data = FloatBuffer(
            nbElems: _batchSize * 3 * _size * _size, deviceID: 0, shared: true
        )
        var dataBuffer = [Float](
            repeating: 0.0, count: _batchSize * 3 * _size * _size
        )
        for i in 0..<_batchSize * 3 * _size * _size
        {
            dataBuffer[i] = Float.random(in: -1..<1)
        }
        data.initialize(array: &dataBuffer)
        
        let nbEpochs = 2
        let nbSteps = 20
        for epoch in 0..<nbEpochs
        {
            print("EPOCH \(epoch + 1)/\(nbEpochs).")
            
            let start1 = Date()
            for step in 0..<nbSteps
            {
                let start2 = Date()
                
                // Reset gradient validity for backward pass
                // and update the batch size (although here it stays the same).
                vgg.updateKernel(batchSize: _batchSize)
                
                // Set data.
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: _batchSize,
                    nbChannels: 3,
                    height: _size,
                    width: _size
                )
                
                // Forward.
                try! vgg.forward()
                
                // Get predictions.
                var preds = [Float](lastLayer.outs.download()[0..<_batchSize])
                preds = preds.map { 1.0 / (1.0 + exp(-$0)) } // Sigmoid.
                
                let end2 = Date()
                let timeSpent = end2.timeIntervalSince(start2)
                print("Step \(step + 1)/\(nbSteps): in \(timeSpent)s.")
            }
            
            let end1 = Date()
            let timeSpent = end1.timeIntervalSince(start1)
            print("Epoch \(epoch + 1), time spent: \(timeSpent)s.")
        }
    }
}
