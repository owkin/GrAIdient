//
// LLMExampleTests.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 12/07/2024.
//

import XCTest
import PythonKit
import GrAIdient

/// Run generation from prompt.
final class LLMExampleTests: XCTestCase
{
    /// Model path on the disk.
    let _modelPath = "/TO/UPDATE/mistral-7B-Instruct-v0.3/"
    
    /// Prompt.
    let _prompt = "How do you do?"
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
    }
    
    ///
    /// Return the index of maximal element in array.
    ///
    /// - Parameter array: Input array.
    /// - Returns: The index of the maximal element.
    ///
    func _argmax(array: [Float]) -> Int?
    {
        if array.isEmpty
        {
            return nil
        }
        
        var maxIndex = 0
        var maxValue = array[0]
        for i in 1..<array.count
        {
            if array[i] > maxValue
            {
                maxIndex = i
                maxValue = array[i]
            }
        }
        return maxIndex
    }
    
    ///
    /// Build LLM model.
    ///
    /// - Parameters:
    ///     - sequence: Length of the sequence.
    ///     - nbBlocks: Number of transformer + MLP blocks.
    ///     - hiddenDim: Dimension of neurons in the main branch.
    ///     - headDim: Dimension of neurons in the transformer branches.
    ///     - mlpDim: Dimension of neurons in the MLP branches.
    ///     - nbHeads:  Number of heads (groups) of neurons for queries.
    ///     - nbHeadsKV: Number of heads (groups) of neurons for keys and values.
    ///     - vocabularySize: Vocabulary size.
    /// - Returns: (The model built, The list of PyTorch keys for each layer that contains weights).
    ///
    func _buildModel(
        sequence: Int,
        nbBlocks: Int,
        hiddenDim: Int,
        headDim: Int,
        mlpDim: Int,
        nbHeadsQuery: Int,
        nbHeadsKV: Int,
        vocabularySize: Int) -> (Model, [String])
    {
        let context = ModelContext(name: "LLM", curID: 0)
        let params = GrAI.Model.Params(context: context)
        var keys = [String]()
        
        var layer: LayerSeq = EmbeddingSeq(
            sequence: sequence,
            vocabularySize: vocabularySize,
            nbNeurons: hiddenDim, params: params
        )
        keys.append("tok_embeddings.weight")
        
        for i in 0..<nbBlocks
        {
            var x: LayerSeq = layer
            
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                params: params
            )
            keys.append("layers.\(i).attention_norm.weight")
            
            var query: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: nbHeadsQuery * headDim,
                activation: nil,
                biases: false,
                params: params
            )
            keys.append("layers.\(i).attention.wq.weight")
            query = try! RoPESeq(
                layerPrev: query,
                seqPositions: [Int](1...sequence),
                nbHeads: nbHeadsQuery,
                params: params
            )
            
            var key: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: nbHeadsKV * headDim,
                activation: nil,
                biases: false,
                params: params
            )
            keys.append("layers.\(i).attention.wk.weight")
            key = try! RoPESeq(
                layerPrev: key,
                seqPositions: [Int](1...sequence),
                nbHeads: nbHeadsKV,
                params: params
            )
            
            let value: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: nbHeadsKV * headDim,
                activation: nil,
                biases: false,
                params: params
            )
            keys.append("layers.\(i).attention.wv.weight")
            
            layer = try! QueryCausalSeq(
                query: query, key: key,
                nbHeadsQuery: nbHeadsQuery, nbHeadsKey: nbHeadsKV,
                params: params
            )
            layer = try! SoftmaxCausalSeq(
                layerPrev: layer,
                nbHeads: nbHeadsQuery,
                params: params
            )
            
            layer = try! ValueCausalSeq(
                value: value, score: layer,
                nbHeadsValue: nbHeadsKV, nbHeadsScore: nbHeadsQuery,
                params: params
            )
            
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: nbHeadsQuery * headDim,
                activation: nil,
                biases: false,
                params: params
            )
            keys.append("layers.\(i).attention.wo.weight")
            
            layer = try! SumSeq(layersPrev: [layer, x], params: params)
            
            x = layer
            
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                params: params
            )
            keys.append("layers.\(i).ffn_norm.weight")
            
            let mult1: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: mlpDim,
                activation: SiLU.str,
                biases: false,
                params: params
            )
            keys.append("layers.\(i).feed_forward.w1.weight")
            
            let mult2: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: mlpDim,
                activation: nil,
                biases: false,
                params: params
            )
            keys.append("layers.\(i).feed_forward.w3.weight")
            
            layer = try! MultiplySeq(layersPrev: [mult1, mult2], params: params)
            
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: hiddenDim,
                activation: nil,
                biases: false,
                params: params
            )
            keys.append("layers.\(i).feed_forward.w2.weight")
            
            layer = try! SumSeq(layersPrev: [layer, x], params: params)
        }
        
        layer = RMSNormSeq(
            layerPrev: layer,
            activation: nil,
            params: params
        )
        keys.append("norm.weight")
        
        layer = FullyConnectedSeq(
            layerPrev: layer,
            nbNeurons: vocabularySize,
            activation: nil,
            biases: false,
            params: params
        )
        keys.append("output.weight")
        
        // Retrieve base model in the context and initialize a
        // real model (with `layerPrev` links updated).
        let model = Model(model: context.model, modelsPrev: [])
        
        return (model, keys)
    }
    
    ///
    /// Load weights.
    ///
    /// - Parameters:
    ///     - model: Model.
    ///     - keys: List of PyTorch keys for each layer that contains weights.
    ///     - weights: The weights to set.
    ///     - pythonLib: Library to call Python functions.
    ///
    func _loadWeights(
        model: Model, keys: [String],
        weights: inout [String: PythonObject],
        pythonLib: PythonObject)
    {
        // Apply weights on the `GrAIdient` model's layers.
        var numKey = 0
        for layer in model.layers
        {
            // Load weights and biases.
            if let layerTmp = layer as? EmbeddingSeq
            {
                let key = keys[numKey]
                let np = pythonLib.extract_state_key(key, weights)
            
                let weightsTmp: [Float] = Array<Float>(
                    numpy: np
                )!
                layerTmp.weightsCPU = weightsTmp
                
                weights[key] = nil
                numKey += 1
            }
            if let layerTmp = layer as? RMSNormSeq
            {
                let key = keys[numKey]
                let np = pythonLib.extract_state_key(key, weights)
                
                let weightsTmp: [Float] = Array<Float>(
                    numpy: np
                )!
                layerTmp.weightsCPU = weightsTmp
                
                weights[key] = nil
                numKey += 1
            }
            if let layerTmp = layer as? FullyConnectedSeq
            {
                let key = keys[numKey]
                let np = pythonLib.extract_state_key(key, weights)
                
                let weightsTmp: [Float] = Array<Float>(
                    numpy: np
                )!
                layerTmp.weightsCPU = weightsTmp
                
                weights[key] = nil
                numKey += 1
            }
        }
    }
    
    ///
    /// Load Mistral weights.
    ///
    /// - Parameters:
    ///     - model: Model.
    ///     - keys: List of PyTorch keys for each layer that contains weights.
    ///     - weightsPath: Weights path on the disk.
    ///
    func _loadMistralWeights(
        model: Model, keys: [String], weightsPath: String)
    {
        // Get weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_mistral_state(weightsPath)
        var weights = [String: PythonObject](data)!
        
        // Load weights.
        _loadWeights(
            model: model,
            keys: keys,
            weights: &weights,
            pythonLib: pythonLib
        )
    }
    
    /// Predict text from prompt.
    func _testPredict1() throws
    {
        let nbBlocks = 1
        let hiddenDim = 4096
        let headDim = 128
        let mlpDim = 14336
        let nbHeadsQuery = 32
        let nbHeadsKV = 8
        let vocabularySize = 32768
        
        // Load tokenizer.
        let pythonLib = Python.import("python_lib")
        let tokenizer = pythonLib.load_mistral_tokenizer(_modelPath)
        
        // Encode prompt.
        let prompt = [Int](pythonLib.encode_mistral(
            _prompt,
            tokenizer
        ))!
        
        // Compute reference.
        let arrayRef = [Float](numpy: pythonLib.predict_mistral(
            _prompt,
            _modelPath,
            1
        ))!
        
        // Build LLM.
        let (model, keys) = _buildModel(
            sequence: prompt.count,
            nbBlocks: nbBlocks,
            hiddenDim: hiddenDim,
            headDim: headDim,
            mlpDim: mlpDim,
            nbHeadsQuery: nbHeadsQuery,
            nbHeadsKV: nbHeadsKV,
            vocabularySize: vocabularySize
        )
        
        // Load pre trained weights.
        _loadMistralWeights(model: model, keys: keys, weightsPath: _modelPath)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        model.updateKernel(batchSize: 1)
        
        // Forward.
        let firstLayer: EmbeddingSeq = model.layers.first as! EmbeddingSeq
        try! firstLayer.setDataGPU(
            [prompt], batchSize: 1, sequence: prompt.count
        )
        try! model.forward()
        
        // Get result.
        let arrayOut = (model.layers.last as! LayerSeq).outs.download()
        
        // Compare difference.
        for (elemOut, elemRef) in zip(arrayOut, arrayRef)
        {
            if elemRef == 0.0
            {
                XCTAssert(elemOut == 0.0)
            }
            else
            {
                let diffPercent = abs(elemOut - elemRef) / abs(elemRef) * 100.0
                if diffPercent > 1
                {
                    print(diffPercent)
                }
                XCTAssert(diffPercent < 1)
            }
        }
    }
    
    /// Predict text from prompt.
    func _testPredict32() throws
    {
        let nbBlocks = 32
        let hiddenDim = 4096
        let headDim = 128
        let mlpDim = 14336
        let nbHeadsQuery = 32
        let nbHeadsKV = 8
        let vocabularySize = 32768
        
        // Load tokenizer.
        let pythonLib = Python.import("python_lib")
        let tokenizer = pythonLib.load_mistral_tokenizer(_modelPath)
        
        // Encode prompt.
        let prompt = [Int](pythonLib.encode_mistral(
            _prompt,
            tokenizer
        ))!
        
        // Build LLM.
        let (model, keys) = _buildModel(
            sequence: prompt.count,
            nbBlocks: nbBlocks,
            hiddenDim: hiddenDim,
            headDim: headDim,
            mlpDim: mlpDim,
            nbHeadsQuery: nbHeadsQuery,
            nbHeadsKV: nbHeadsKV,
            vocabularySize: vocabularySize
        )
        
        // Load pre trained weights.
        _loadMistralWeights(model: model, keys: keys, weightsPath: _modelPath)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        model.updateKernel(batchSize: 1)
        
        // Forward.
        let firstLayer: EmbeddingSeq = model.layers.first as! EmbeddingSeq
        try! firstLayer.setDataGPU(
            [prompt], batchSize: 1, sequence: prompt.count
        )
        try! model.forward()
        
        // Get result.
        let out = (model.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        var tokens = [Int]()
        for seq in 0..<out.count / vocabularySize
        {
            let probas = [Float](
                out[vocabularySize*seq..<vocabularySize*(seq+1)]
            )
            let token = _argmax(array: probas)!
            tokens.append(token)
        }
        
        // Decode.
        let prediction = String(pythonLib.decode_mistral(
            tokens,
            tokenizer
        ))!
        
        print(prediction)
        XCTAssert(prediction == "QuestionI can I calculate a I I")
    }
}
