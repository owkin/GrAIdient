//
// TransformerExample.swift
// GrAIExamples
//
// Created by Aurélien PEDEN on 14/03/2023.
//

import XCTest
import GrAIdient

final class TransformerExample: XCTestCase
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
    
    // Initialize test.
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
    /// Build a multi attention branch.
    ///
    /// - Parameters:
    ///     - layerPrev: previous layer.
    ///     - nbHeads: Number of head in attention branches.
    ///     - hiddenDim: Dimension of neurons in the main branch.
    ///     - params: Contextual parameters linking to the model.
    /// - Returns: The model built.
    ///
    func buildMultiHeadAttention(
        layerPrev: LayerSeq,
        nbHeads: Int,
        hiddenDim: Int,
        params: GrAI.Model.Params) -> LayerSeq
    {
        var queries = [LayerSeq]()
        var keys = [LayerSeq]()
        var values = [LayerSeq]()
        
        for _ in 0..<nbHeads
        {
            queries.append(FullyConnectedSeq(
                layerPrev: layerPrev, nbNeurons: hiddenDim / nbHeads,
                activation: nil, biases: true,
                params: params
            ))
        }
        for _ in 0..<nbHeads
        {
            keys.append(FullyConnectedSeq(
                layerPrev: layerPrev, nbNeurons: hiddenDim / nbHeads,
                activation: nil, biases: true,
                params: params
            ))
        }
        for _ in 0..<nbHeads
        {
            values.append(FullyConnectedSeq(
                layerPrev: layerPrev, nbNeurons: hiddenDim / nbHeads,
                activation: nil, biases: true,
                params: params
            ))
        }
        
        var attention = [LayerSeq]()
        for head in 0..<nbHeads
        {
            var score: LayerSeq = QuerySeq(
                query: queries[head], key: keys[head], nbHeads: 1,
                params: params
            )
            score = SoftmaxSeq(layerPrev: score, params: params)
            
            attention.append(ValueSeq(
                value: values[head], score: score, params: params
            ))
        }
        
        var layerSeq: LayerSeq = Concat2Seq(
            layersPrev: attention, params: params
        )
        
        layerSeq = FullyConnectedSeq(
            layerPrev: layerSeq, nbNeurons: hiddenDim,
            activation: nil, biases: true,
            params: params
        )
        
        return layerSeq
    }
    
    ///
    /// Build a simple VisionTransformer model.
    ///
    /// - Parameters:
    ///     - size: The data input size.
    ///     - patch: Size of patch.
    ///     - nbLayers: Number of atttention branches.
    ///     - nbHeads: Number of head in attention branches.
    ///     - hiddenDim: Dimension of neurons in the main branch.
    ///     - mlpDim: Dimension of neurons in the MLP branch.
    ///     - mlpActivation: Activation function in the MLP branch.
    /// - Returns: The model built.
    ///
    func _buildModel(
        size: Int,
        patch: Int,
        nbLayers: Int,
        nbHeads: Int,
        hiddenDim: Int,
        mlpDim: Int,
        mlpActivation: String) -> Model
    {
        let context = ModelContext(name: "VisionTransformer", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: size,
            height: size,
            params: params
        )
        
        let extraClass: LayerSeq = Constant2Seq(
            sequence: 1, nbNeurons: hiddenDim, params: params
        )
        
        var layerSeq: LayerSeq = FullyConnectedPatch(
            layerPrev: layer, patch: patch, nbNeurons: hiddenDim,
            activation: nil, biases: true,
            params: params
        )
        let sequence = layerSeq.sequence + 1
        
        let posEmbedding: LayerSeq = Constant12Seq(
            sequence: sequence, nbNeurons: hiddenDim, params: params
        )
        
        layerSeq = Concat1Seq(
            layersPrev: [extraClass, layerSeq], params: params
        )
        layerSeq = SumSeq(
            layersPrev: [layerSeq, posEmbedding], params: params
        )
        
        for _ in 0..<nbLayers
        {
            var layerInput = layerSeq
            
            layerSeq = LayerNormSeq(
                layerPrev: layerSeq, activation: nil, params: params
            )
            
            layerSeq = buildMultiHeadAttention(
                layerPrev: layerSeq,
                nbHeads: nbHeads, hiddenDim: hiddenDim,
                params: params
            )
            
            layerSeq = SumSeq(
                layersPrev: [layerSeq, layerInput], params: params
            )
            layerInput = layerSeq
            
            layerSeq = LayerNormSeq(
                layerPrev: layerSeq, activation: nil, params: params
            )
            
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: mlpDim,
                activation: mlpActivation, biases: true,
                params: params
            )
            
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: hiddenDim,
                activation: nil, biases: true,
                params: params
            )
            
            layerSeq = SumSeq(
                layersPrev: [layerSeq, layerInput], params: params
            )
        }
        
        layerSeq = LayerNormSeq(
            layerPrev: layerSeq, activation: nil, params: params
        )
        
        let head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        _ = MSE1D(layerPrev: head, params: params)
        
        // Retrieve base model in the context and initialize a
        // real model (with `layerPrev` links updated).
        let model = Model(model: context.model, modelsPrev: [])
        return model
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
        let transformer = Model(model: baseModel, modelsPrev: [])
        return transformer
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
        
        let pixels8 = getPixels(
            samples8, width: _size, height: _size, imageFormat: .Neuron
        )
        let pixels5 = getPixels(
            samples5, width: _size, height: _size, imageFormat: .Neuron
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
            
            image = getImage(
                pixels: pixels5[elem], width: _size, height: _size
            )
            saveImage(
                image,
                url: URL(fileURLWithPath: _outputDir + "CIFAR5_\(elem).png")
            )
        }
    }
    
    /// Test3: test that an untrained model makes bad predictions.
    func test3_UntrainedModel()
    {
        // Build a model with randomly initialized weights.
        let transformer = _buildModel(
            size: 32,
            patch: 16,
            nbLayers: 2,
            nbHeads: 2,
            hiddenDim: 16,
            mlpDim: 32,
            mlpActivation: GELU.str
        )
        
        // Initialize for inference.
        transformer.initKernel(phase: .Inference)
        
        // Evaluate model on CIFAR testing dataset.
        let ratio = _evaluateModel(transformer)
        
        print(
            "Ratio of good predictions: \(ratio)%."
        )
        
        // Encode the model.
        let encoder = PropertyListEncoder()
        let data = try! encoder.encode(transformer)
        
        // Save it to the disk.
        try! data.write(
            to: URL(fileURLWithPath: _outputDir + "/transformer1.plist")
        )
    }
    
    /// Test4: train a simple model.
    func test4_TrainTransformer()
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
        let transformer = _loadModel(_outputDir + "transformer1.plist")
        
        // Initialize for training.
        transformer.initialize(params: params, phase: .Training)
        
        let firstLayer: Input2D = transformer.layers.first as! Input2D
        let lastLayer: MSE1D = transformer.layers.last as! MSE1D
        
        // Initialize the ground truth once and for all.
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
                transformer.updateKernel(batchSize: _batchSize)
                
                // Set data.
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: _batchSize,
                    format: .Neuron
                )
                
                // Forward.
                try! transformer.forward()
                
                // Apply loss derivative.
                try! lastLayer.lossDerivativeGPU(
                    groundTruth,
                    batchSize: _batchSize
                )
                
                // Backward.
                try! transformer.backward()
                
                // Update weights.
                try! transformer.update()
                
                // Get loss result.
                // Note that backward is explicitly
                // enabled by `applyGradient` whereas `getLoss` is
                // just an indicator.
                let loss = try! lastLayer.getLossGPU(
                    groundTruth,
                    batchSize: _batchSize
                )
                print("Step \(step)/\(cifar8.nbLoops-1): \(sqrt(loss)).")
                
                // Update internal step.
                // This is not mandatory except if we used another
                // optimizer scheduler: see `_getOptimizerParams`.
                transformer.incStep()
            }
        }
        
        // Encode the trained model.
        let encoder = PropertyListEncoder()
        let data = try! encoder.encode(transformer)
        
        // Save it to the disk.
        try! data.write(
            to: URL(fileURLWithPath: _outputDir + "/transformer2.plist")
        )
    }
    
    /// Test5: test that the previous trained model makes better predictions than the untrained model.
    func test5_CompareModels()
    {
        // Load previous model from the disk.
        let transformer1 = _loadModel(_outputDir + "/transformer1.plist")
        let transformer2 = _loadModel(_outputDir + "/transformer2.plist")
        
        // Initialize for inference.
        transformer1.initKernel(phase: .Inference)
        transformer2.initKernel(phase: .Inference)
        
        // Evaluate model on CIFAR testing dataset.
        let ratio1 = _evaluateModel(transformer1)
        let ratio2 = _evaluateModel(transformer2)
        
        print(
            "Ratio of good predictions before training: \(ratio1)%."
        )
        print(
            "Ratio of good predictions after training: \(ratio2)%."
        )
    }
}
