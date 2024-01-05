//
// TransformerBenchmark.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 21/12/2023.
//

import XCTest
import GrAIdient

/// Benchmark time spent for training or evaluating a Vision Transformer model with fake data.
final class TransformerBenchmark: XCTestCase
{
    /// Batch size of data.
    let _batchSize = 64
    /// Size of one image (height and width are the same).
    let _size = 224
    
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
    
    /// Test: train a ViT model.
    func _test_TrainTransformer()
    {
        // Get optimizer parameters for iterating over batch size elements.
        let params = _getOptimizerParams(nbLoops: _batchSize)
        
        // Build a model with randomly initialized weights.
        let transformer = _buildModel(
            size: _size,
            patch: 16,
            nbLayers: 12,
            nbHeads: 6,
            hiddenDim: 384,
            mlpDim: 4 * 384,
            mlpActivation: ReLU.str
        )
        
        // Initialize for training.
        transformer.initialize(params: params, phase: .Training)
        
        let firstLayer: Input2D = transformer.layers.first as! Input2D
        let lastLayer: MSE1D = transformer.layers.last as! MSE1D
        
        // Initialize the ground truth once and for all.
        let groundTruth = MetalSharedBuffer<Float>(_batchSize, deviceID: 0)
        let gtBuffer = groundTruth.buffer
        for elem in 0..<_batchSize / 2
        {
            gtBuffer[elem] = 0.0
        }
        for elem in _batchSize / 2..<_batchSize
        {
            gtBuffer[elem] = 1.0
        }
        groundTruth.upload()
        
        // Initialize data once and for all.
        let data = MetalPrivateBuffer<Float>(
            _batchSize * 3 * _size * _size, deviceID: 0
        )
        let dataBuffer = data.shared.buffer
        for i in 0..<_batchSize * 3 * _size * _size
        {
            dataBuffer[i] = Float.random(in: -1..<1)
        }
        data.upload()
        
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
                transformer.updateKernel(batchSize: _batchSize)
                
                // Set data.
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: _batchSize,
                    nbChannels: 3,
                    height: _size,
                    width: _size
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
                
                // Update internal step.
                // This is not mandatory except if we used another
                // optimizer scheduler: see `_getOptimizerParams`.
                transformer.incStep()
                
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
    
    /// Test: evaluate a ViT model.
    func _test_EvalTransformer()
    {
        // Build a model with randomly initialized weights.
        let transformer = _buildModel(
            size: _size,
            patch: 16,
            nbLayers: 12,
            nbHeads: 12,
            hiddenDim: 768,
            mlpDim: 4 * 768,
            mlpActivation: ReLU.str
        )
        
        // Initialize for inference.
        transformer.initKernel(phase: .Inference)
        
        let firstLayer: Input2D = transformer.layers.first as! Input2D
        let lastLayer: MSE1D = transformer.layers.last as! MSE1D
        
        // Initialize the ground truth once and for all.
        let groundTruth = MetalSharedBuffer<Float>(_batchSize, deviceID: 0)
        let gtBuffer = groundTruth.buffer
        for elem in 0..<_batchSize / 2
        {
            gtBuffer[elem] = 0.0
        }
        for elem in _batchSize / 2..<_batchSize
        {
            gtBuffer[elem] = 1.0
        }
        groundTruth.upload()
        
        // Initialize data once and for all.
        let data = MetalPrivateBuffer<Float>(
            _batchSize * 3 * _size * _size, deviceID: 0
        )
        let dataBuffer = data.shared.buffer
        for i in 0..<_batchSize * 3 * _size * _size
        {
            dataBuffer[i] = Float.random(in: -1..<1)
        }
        data.upload()
        
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
                transformer.updateKernel(batchSize: _batchSize)
                
                // Set data.
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: _batchSize,
                    nbChannels: 3,
                    height: _size,
                    width: _size
                )
                
                // Forward.
                try! transformer.forward()
                
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
