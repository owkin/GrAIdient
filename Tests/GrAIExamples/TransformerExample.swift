//
// TransformerExample.swift
// GrAIExamples
//
// Created by Aurélien PEDEN on 14/03/2023.
//

import XCTest
import GrAIdient

/// Train a simple Vision Transformer model on the CIFAR dataset.
final class TransformerExample: XCTestCase
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
    /// - Returns: The last layer of the branch.
    ///
    func _buildMultiHeadAttention(
        layerPrev: LayerSeq,
        nbHeads: Int,
        hiddenDim: Int,
        params: GrAI.Model.Params) -> LayerSeq
    {
        let qkv: LayerSeq = FullyConnectedSeq(
            layerPrev: layerPrev, nbNeurons: 3 * hiddenDim,
            activation: nil, biases: true,
            params: params
        )
        
        var layerSeq: LayerSeq = try! QuerySelfSeq(
            layerPrev: qkv,
            query: 0, key: 1, nbBlocksPrev: 3, nbHeads: nbHeads,
            params: params
        )
        layerSeq = try! SoftmaxSeq(
            layerPrev: layerSeq, nbHeads: nbHeads,
            params: params
        )
        layerSeq = try! ValueSelfSeq(
            value: qkv, score: layerSeq,
            offset: 2, nbBlocksPrev: 3, nbHeads: nbHeads,
            params: params
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
        
        var layerSeq: LayerSeq = try! FullyConnectedPatch(
            layerPrev: layer, patch: patch, nbNeurons: hiddenDim,
            activation: nil, biases: true,
            params: params
        )
        let sequence = layerSeq.sequence + 1
        
        let posEmbedding: LayerSeq = Constant12Seq(
            sequence: sequence, nbNeurons: hiddenDim, params: params
        )
        
        layerSeq = try! Concat1Seq(
            layersPrev: [extraClass, layerSeq], params: params
        )
        layerSeq = try! SumSeq(
            layersPrev: [layerSeq, posEmbedding], params: params
        )
        
        for _ in 0..<nbLayers
        {
            var layerInput = layerSeq
            
            layerSeq = LayerNormSeq(
                layerPrev: layerSeq, activation: nil, params: params
            )
            
            layerSeq = _buildMultiHeadAttention(
                layerPrev: layerSeq,
                nbHeads: nbHeads, hiddenDim: hiddenDim,
                params: params
            )
            
            layerSeq = try! SumSeq(
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
            
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, layerInput], params: params
            )
        }
        
        layerSeq = LayerNormSeq(
            layerPrev: layerSeq, activation: nil, params: params
        )
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: ReLU.str, biases: true,
            params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
        
        // Retrieve base model in the context and initialize a
        // real model (with `layerPrev` links updated).
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
        CIFAR.dumpTrain(
            datasetPath: _outputDir + "/datasetTrain5",
            label: 5,
            size: _size
        )
    }
    
    /// Test2: train a simple model.
    func test2_TrainTransformer()
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
        
        // Build a model with randomly initialized weights.
        let transformer = _buildModel(
            size: _size,
            patch: 16,
            nbLayers: 2,
            nbHeads: 2,
            hiddenDim: 16,
            mlpDim: 32,
            mlpActivation: ReLU.str
        )
        
        // Initialize for training.
        transformer.initialize(params: params, phase: .Training)
        
        let firstLayer: Input2D = transformer.layers.first as! Input2D
        let lastLayer: MSE1D = transformer.layers.last as! MSE1D
        
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
        groundTruth.upload()
        
        let nbEpochs = 2
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
                transformer.updateKernel(batchSize: _batchSize)
                
                // Set data.
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: _batchSize,
                    nbChannels: 3, height: _size, width: _size,
                    format: .Neuron
                )
                
                // Forward.
                try! transformer.forward()
                
                // Apply loss derivative.
                try! lastLayer.lossDerivativeGPU(
                    groundTruth,
                    batchSize: _batchSize,
                    nbNeurons: 1
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
                    batchSize: _batchSize,
                    nbNeurons: 1
                )
                print("Step \(step + 1)/\(cifar8.nbLoops): \(sqrt(loss)).")
                
                // Update internal step.
                // This is not mandatory except if we used another
                // optimizer scheduler: see `_getOptimizerParams`.
                transformer.incStep()
            }
        }
    }
}
