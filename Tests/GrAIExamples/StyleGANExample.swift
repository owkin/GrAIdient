//
//  StyleGANExample.swift
//  GrAIExamples
//
//  Created by Aurélien PEDEN on 23/03/2023.
//

import XCTest
import GrAIdient

final class StyleGANExample : XCTestCase
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
    /// Get optimizer parameters for model training.
    ///
    /// - Parameter nbLoops: Number of steps per epoch.
    /// - Returns: The optimizer parameters.
    ///
//    func _getOptimizerParams(nbLoops: Int) -> GrAI.Optimizer.Params
//    {
//        var optimizerParams = GrAI.Optimizer.Params()
//        optimizerParams.nbLoops = nbLoops
//
//        // Simple optimizer scheduler: always the same optimizer during
//        // the training.
//        optimizerParams.optimizer = ConstEpochsScheduler(
//            GrAI.Optimizer.Class.AdamRectified
//        )
//
//        // Simple variable scheduler: always the same variable during
//        // the training.
//        optimizerParams.variables["alpha"] = ConstEpochsVar(
//            value: ConstVal(1e-3)
//        )
//        optimizerParams.variables["lambda"] = ConstEpochsVar(
//            value: ConstVal(1e-6)
//        )
//
//        // Other schedulers can be built thanks to `GrAI.Optimizer.Params`.
//        return optimizerParams
//    }
    
    func buildCNNEncoder(params: GrAI.Model.Params) -> Layer2D
    {
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: _size, height: _size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 3, stride: 2,
            activation: ReLU.str, biases: true, bn: true,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 12, stride: 2,
            activation: ReLU.str, biases: true, bn: true,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 24, stride: 2,
            activation: ReLU.str, biases: true, bn: true,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 48, stride: 2,
            activation: ReLU.str, biases: true, bn: true,
            params: params
        )
        
        return layer
    }
    
    func buildTestCnnDecoder(params: GrAI.Model.Params, layerPrev: Layer2D) -> Layer2D
    {
        var layer: Layer2D
        
        layer = Deconvolution2D(layerPrev: layerPrev, size: 2, nbChannels: 48, stride: 2, activation: ReLU.str, biases: true, bn: true, params: params)
        
        layer = Deconvolution2D(layerPrev: layer, size: 2, nbChannels: 24, stride: 2, activation: ReLU.str, biases: true, bn: true, params: params)
        
        layer = Deconvolution2D(layerPrev: layer, size: 2, nbChannels: 12, stride: 2, activation: ReLU.str, biases: true, bn: true, params: params)
        
        layer = Deconvolution2D(layerPrev: layer, size: 2, nbChannels: 3, stride: 2, activation: Sigmoid.str, biases: true, bn: true, params: params)
        
        layer = MSE2D(layerPrev: layer, params: params)
        
        return layer
        
    }
    
    ///
    /// Build the Mapping Network, taking a latent vector as input and outputing a latent vector w of same dimension
    ///
    /// - Parameters:
    /// - Returns: The model built.
    ///
    func buildMappingNetwork(params: GrAI.Model.Params) -> Layer1D
    {
        // Latent
        // Pixel norm
        
        // Mapping Network
        var layer: Layer1D
        layer = Input1D(nbNeurons: 512, params: params)
        layer = FullyConnected(layerPrev: layer, nbNeurons: 512, activation: ReLU.str, biases: true, params: params)
        layer = FullyConnected(layerPrev: layer, nbNeurons: 512, activation: ReLU.str, biases: true, params: params)
        layer = FullyConnected(layerPrev: layer, nbNeurons: 512, activation: ReLU.str, biases: true, params: params)
        layer = FullyConnected(layerPrev: layer, nbNeurons: 512, activation: ReLU.str, biases: true, params: params)
        layer = FullyConnected(layerPrev: layer, nbNeurons: 512, activation: ReLU.str, biases: true, params: params)
        layer = FullyConnected(layerPrev: layer, nbNeurons: 512, activation: ReLU.str, biases: true, params: params)
        layer = FullyConnected(layerPrev: layer, nbNeurons: 512, activation: ReLU.str, biases: true, params: params)
        layer = FullyConnected(layerPrev: layer, nbNeurons: 512, activation: ReLU.str, biases: true, params: params)
        
        return layer
    }
    
    ///
    /// Build a multi attention branch.
    ///
    /// - Parameters:
    /// - Returns: The model built.
    ///
    func buildSynthesisNetwork(mappingNetworkOutput: Layer1D, params: GrAI.Model.Params) -> Layer2D
    {
        var layer: Layer2D
        layer = Constant2D(nbChannels: 512, height: 4, width: 4, params: params)
        // Sum ?
        // Affine transformation
        layer = AdaIN(layersPrev: [layer, mappingNetworkOutput], params: params)
        layer = Convolution2D(layerPrev: layer, size: 3, nbChannels: 3, stride: 1, activation: ReLU.str, biases: true, bn: false, params: params)
        // Sum ?
        // Affine transformation
        layer = AdaIN(layersPrev: [layer, mappingNetworkOutput], params: params)
        
        
        // Upsample
        layer = ResizeBilinearCrop(layerPrev: layer, scalesList: [2, 2], params: params)
        layer = Convolution2D(layerPrev: layer, size: 3, nbChannels: 3, stride: 1, activation: ReLU.str, biases: true, bn: false, params: params)
        
        layer = AdaIN(layersPrev: [layer, mappingNetworkOutput], params: params)
        layer = Convolution2D(layerPrev: layer, size: 3, nbChannels: 3, stride: 1, activation: ReLU.str, biases: true, bn: false, params: params)
        
        layer = AdaIN(layersPrev: [layer, mappingNetworkOutput], params: params)
        
        // Upsample
        layer = ResizeBilinearCrop(layerPrev: layer, scalesList: [2, 2], params: params)
        layer = Convolution2D(layerPrev: layer, size: 3, nbChannels: 3, stride: 1, activation: ReLU.str, biases: true, bn: false, params: params)
        
        layer = AdaIN(layersPrev: [layer, mappingNetworkOutput], params: params)
        layer = Convolution2D(layerPrev: layer, size: 3, nbChannels: 3, stride: 1, activation: ReLU.str, biases: true, bn: false, params: params)
        
        layer = AdaIN(layersPrev: [layer, mappingNetworkOutput], params: params)
        
        // Upsample
        layer = ResizeBilinearCrop(layerPrev: layer, scalesList: [2, 2], params: params)
        layer = Convolution2D(layerPrev: layer, size: 3, nbChannels: 3, stride: 1, activation: ReLU.str, biases: true, bn: false, params: params)
        
        layer = AdaIN(layersPrev: [layer, mappingNetworkOutput], params: params)
        layer = Convolution2D(layerPrev: layer, size: 3, nbChannels: 3, stride: 1, activation: ReLU.str, biases: true, bn: false, params: params)
        
        layer = AdaIN(layersPrev: [layer, mappingNetworkOutput], params: params)
        
        return layer
    }
    
    ///
    /// Build a StyleGAN model.
    ///
    ///
    /// - Returns: The model built.
    ///
    func _buildModel() -> Model
    {
        // Create the context to build a graph of layers where
        // there is no previous model dependency: layer id starts at 0.
        let context = ModelContext(name: "CNN-AE", models: [])
        let params = GrAI.Model.Params(context: context)
        
        let encoder = buildCNNEncoder(params: params)
        let decoder = buildTestCnnDecoder(params: params, layerPrev: encoder)
        // let mappingNetwork = buildMappingNetwork(params: params)
        // let synthesisNetwork = buildSynthesisNetwork(mappingNetworkOutput: mappingNetwork, params: params)
        
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
        let cnn_ae = Model(model: baseModel, modelsPrev: [])
        return cnn_ae
    }
    
    func buildMatter(size: Int, leakyReLU: Bool = false) -> Model
        {
            let context = ModelContext(name: "PrimeMatter", curID: 0)
            let params = GrAI.Model.Params(context: context)
            let activationStr = leakyReLU ? LeakyReLU.str : ReLU.str
            
            var layer, layer1, layer2, layer3, layer4: Layer2D
            layer = Input2D(
                nbChannels: 3,
                width: size,
                height: size,
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 64, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            layer1 = layer
            
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 128, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            layer2 = layer
            
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 256, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 256, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            layer3 = layer
            
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 512, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 512, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            layer4 = layer
            
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 512, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 512, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Deconvolution2D(
                layerPrev: layer,
                size: 2, nbChannels: 512, stride: 2,
                activation: nil, biases: true, bn: false,
                params: params
            )
            
            layer = Concat2D(
                layersPrev: [layer4, layer],
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 256, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 256, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Deconvolution2D(
                layerPrev: layer,
                size: 2, nbChannels: 256, stride: 2,
                activation: nil, biases: true, bn: false,
                params: params
            )
            
            layer = Concat2D(
                layersPrev: [layer3, layer],
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 128, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 128, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Deconvolution2D(
                layerPrev: layer,
                size: 2, nbChannels: 128, stride: 2,
                activation: nil, biases: true, bn: false,
                params: params
            )
            
            layer = Concat2D(
                layersPrev: [layer2, layer],
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 64, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 64, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Deconvolution2D(
                layerPrev: layer,
                size: 2, nbChannels: 64, stride: 2,
                activation: nil, biases: true, bn: false,
                params: params
            )
            
            layer = Concat2D(
                layersPrev: [layer1, layer],
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 64, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 3, nbChannels: 64, stride: 1,
                activation: activationStr, biases: true, bn: false,
                params: params
            )
            
            layer = Convolution2D(
                layerPrev: layer,
                size: 1, nbChannels: 1, stride: 1,
                activation: Sigmoid.str, biases: true, bn: false,
                params: params
            )
            
            layer = MSE2D(layerPrev: layer, params: params)
            
            let model = Model(model: context.model, modelsPrev: [])
            
            return model
        }
    
    /// Test1: dump CIFAR train and test datasets for labels 8 and 5.
    func test1_DumpDataset()
    {
        CIFAR.dumpTrain(
            datasetPath: _outputDir + "/datasetTrain8",
            label: 8,
            size: _size
        )
        
        CIFAR.dumpTest(
            datasetPath: _outputDir + "/datasetTest8",
            label: 8,
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
        
        cifar8.initSamples(batchSize: batchSize)
        
        let samples8 = cifar8.getSamples()!
        
        let pixels8 = getPixels(
            samples8, width: _size, height: _size, imageFormat: .Neuron
        )
        
        for elem in 0..<batchSize
        {
            var image = getImage(
                pixels: pixels8[elem], width: _size, height: _size
            )
            saveImage(
                image,
                url: URL(fileURLWithPath: _outputDir + "CIFAR8_\(elem).png")
            )
        }
    }
    
    func test3_SaveModel()
    {
        // Build a model with randomly initialized weights.
        //let cnn_ae = _buildModel()
        let unet = buildMatter(size: 32)
        
        // Encode the model.
        let encoder = PropertyListEncoder()
        //let data = try! encoder.encode(cnn_ae)
        let data = try! encoder.encode(unet)
        
        // Save it to the disk.
        try! data.write(
            to: URL(fileURLWithPath: _outputDir + "/cnn_ae1.plist")
        )
    }
    
    /// Test4: train a simple model.
    func test3_TrainModel()
    {
        let cifar8 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTrain8",
            size: _size
        )
        
        // Get optimizer parameters for iterating over batch size elements.
        let params = _getOptimizerParams(nbLoops: _batchSize)
        
        cifar8.initSamples(batchSize: _batchSize)
        
        // Keep a subset of the dataset to have a quicker training.
        cifar8.keep(500)
        
        // Small trick to force full batches throughout the training:
        // this enables us to set the ground truth once and for all.
        let nbWholeBatches =
            cifar8.nbSamples / cifar8.batchSize * cifar8.batchSize
        cifar8.keep(nbWholeBatches)
        
        // Load previous model from the disk.
        let cnn_ae = _loadModel(_outputDir + "cnn_ae1.plist")
        
        // Initialize for training.
        cnn_ae.initialize(params: params, phase: .Training)
        
        let firstLayer: Input2D = cnn_ae.layers.first as! Input2D
        let lastLayer: MSE2D = cnn_ae.layers.last as! MSE2D
        
        // Initialize the ground truth once and for all.
//        let groundTruth = MetalSharedBuffer<Float>(_batchSize, deviceID: 0)
//        let buffer = groundTruth.buffer
//        for elem in 0..<_batchSize / 2
//        {
//            buffer[elem] = 0.0
//        }
//        for elem in _batchSize / 2..<_batchSize
//        {
//            buffer[elem] = 1.0
//        }
//        MetalKernel.get.upload([groundTruth])
        
        let nbEpochs = 15
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
                cnn_ae.updateKernel(batchSize: _batchSize)
                
                // Set data.
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: _batchSize,
                    format: .Neuron
                )
                
                // Forward.
                try! cnn_ae.forward()
                
                // Apply loss derivative.
                try! lastLayer.lossDerivativeGPU(data, batchSize: _batchSize, format: .Neuron)
//                try! lastLayer.lossDerivativeGPU(
//                    data,
//                    //groundTruth,
//                    batchSize: _batchSize
//                )
                
                // Backward.
                try! cnn_ae.backward()
                
                // Update weights.
                try! cnn_ae.update()
                
                // Get loss result.
                // Note that backward is explicitly
                // enabled by `applyGradient` whereas `getLoss` is
                // just an indicator.
                let loss = try! lastLayer.getLossGPU(data, batchSize: _batchSize, format: .Neuron)
//                let loss = try! lastLayer.getLossGPU(
//                    groundTruth,
//                    batchSize: _batchSize
//                )
                print("Step \(step)/\(cifar8.nbLoops-1): \(sqrt(loss)).")
                
                // Update internal step.
                // This is not mandatory except if we used another
                // optimizer scheduler: see `_getOptimizerParams`.
                cnn_ae.incStep()
            }
        }
    
        // Encode the trained model.
        let encoder = PropertyListEncoder()
        let data = try! encoder.encode(cnn_ae)
        
        // Save it to the disk.
        try! data.write(
            to: URL(fileURLWithPath: _outputDir + "/cnn_ae2.plist")
        )
    }
    
}
