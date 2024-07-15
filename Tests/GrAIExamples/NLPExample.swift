//
// NLPExample.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 12/06/2024.
//

import XCTest
import PythonKit
import GrAIdient

/// Run generation from prompt.
final class NLPExample: XCTestCase
{
    /// Model path on the disk.
    let _modelPath = "/Users/jean-francoisreboud/DocumentsNonSync/Projet/Python/mistral/weights/mistral-7B-Instruct-v0.3/"
    
    /// Prompt.
    let _prompt = "What is the meaning of life?"
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float16 = true // for faster restuls
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
        let context = ModelContext(name: "NLP", curID: 0)
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
    ///
    func _loadWeights(
        model: Model, keys: [String], weights: inout [String: PythonObject])
    {
        // Apply weights on the `GrAIdient` model's layers.
        var numKey = 0
        for layer in model.layers
        {
            // Load weights and biases.
            if let layerTmp = layer as? EmbeddingSeq
            {
                let key = keys[numKey]
                let np = weights[key]!
            
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
                let np = weights[key]!
                
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
                let np = weights[key]!
                
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
        let data = pythonLib.load_mistral_weights(weightsPath)
        var weights = [String: PythonObject](data)!
        
        // Load weights.
        _loadWeights(model: model, keys: keys, weights: &weights)
    }
    
    ///
    /// Load Llama2 weights.
    ///
    /// - Parameters:
    ///     - model: Model.
    ///     - keys: List of PyTorch keys for each layer that contains weights.
    ///     - weightsPath: Weights path on the disk.
    ///
    func _loadLlama2Weights(
        model: Model, keys: [String], weightsPath: String)
    {
        // Get weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_llama2_weights(weightsPath)
        var weights = [String: PythonObject](data)!
        
        // Load weights.
        _loadWeights(model: model, keys: keys, weights: &weights)
    }
    
    ///
    /// Load Llama3 weights.
    ///
    /// - Parameters:
    ///     - model: Model.
    ///     - keys: List of PyTorch keys for each layer that contains weights.
    ///     - weightsPath: Weights path on the disk.
    ///
    func _loadLlama3Weights(
        model: Model, keys: [String], weightsPath: String)
    {
        // Get weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_llama3_weights(weightsPath)
        var weights = [String: PythonObject](data)!
        
        // Load weights.
        _loadWeights(model: model, keys: keys, weights: &weights)
    }
    
    ///
    /// Prepare model for generation.
    ///
    /// - Parameters:
    ///     - model: Model.
    ///     - nbTokens: Number of tokens which have been generated.
    ///     - seqMax: Maximal number of tokens to generate.
    /// - Returns: The cache.
    ///
    func _prepareForGeneration(
        model: Model,
        nbTokens: Int,
        seqMax: Int) -> [Int: FloatBuffer]
    {
        var cache = [Int: FloatBuffer]()
        for layer in model.layers
        {
            let id = layer.id
            if let layerTmp = layer as? QueryCausalSeq
            {
                cache[id] = (layerTmp.layersPrev[1] as! LayerSeq).outs
                layerTmp.cacheSeq = nbTokens
                layerTmp.cacheSeqMax = seqMax
            }
            else if let layerTmp = layer as? SoftmaxCausalSeq
            {
                layerTmp.cacheSeq = nbTokens
            }
            else if let layerTmp = layer as? ValueCausalSeq
            {
                cache[id] = (layerTmp.layersPrev[0] as! LayerSeq).outs
                layerTmp.cacheSeq = nbTokens
                layerTmp.cacheSeqMax = seqMax
            }
        }
        return cache
    }
    
    ///
    /// Set cache.
    ///
    /// - Parameters:
    ///     - model: Model.
    ///     - cache: The cache to set.
    ///
    /// - Returns: The cache.
    ///
    func _setCache(
        model: Model,
        cache: [Int: FloatBuffer])
    {
        for layer in model.layers
        {
            let id = layer.id
            if let layerTmp = layer as? QueryCausalSeq
            {
                layerTmp.cacheKey = cache[id]!
            }
            else if let layerTmp = layer as? ValueCausalSeq
            {
                layerTmp.cacheValue = cache[id]!
            }
        }
    }
    
    ///
    /// Update sequence positions of RoPE layers.
    ///
    /// - Parameters:
    ///     - model: Model.
    ///     - curSeq: New sequence position to set.
    ///
    func _updateRoPE(model: Model, curSeq: Int)
    {
        for layer in model.layers
        {
            if let layerTmp = layer as? RoPESeq
            {
                layerTmp.seqPositions = [curSeq]
            }
        }
    }
    
    ///
    /// Print sentence, chunk by chunk.
    ///
    /// - Parameters:
    ///     - sentence: The sentence to print.
    ///     - skip: What has already been printed.
    ///
    func _printChunk(sentence: String, skip: inout Int)
    {
        if sentence.count - skip > 1
        {
            let rangeToPrint = sentence.index(
                sentence.startIndex, offsetBy: skip
            )..<sentence.index(sentence.endIndex, offsetBy: -1)
            let strToPrint = sentence[rangeToPrint]
            
            print(strToPrint, terminator: "")
            skip = sentence.count - 1
        }
    }
    
    /// Generate text from prompt.
    func generate(
        prompt: String, 
        maxTokens: Int,
        specialLastToken: Int,
        model: Model,
        encoder: (String) -> [Int],
        decoder: ([Int]) -> String) throws
    {
        // Encode prompt.
        let promptTmp = encoder(prompt)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        model.updateKernel(batchSize: 1)
        
        // Forward.
        var firstLayer: EmbeddingSeq = model.layers.first as! EmbeddingSeq
        try! firstLayer.setDataGPU(
            [promptTmp], batchSize: 1, sequence: promptTmp.count
        )
        try! model.forward()
        
        // Get result.
        let out = (model.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        let fistLayer = model.layers.first as! EmbeddingSeq
        let vocabularySize = fistLayer.vocabularySize
        
        var tokens = [Int]()
        for seq in 0..<out.count / vocabularySize
        {
            let probas = [Float](
                out[vocabularySize * seq..<vocabularySize * (seq+1)]
            )
            let argmax = _argmax(array: probas)!
            tokens.append(argmax)
        }
        
        var lastToken = tokens.last!
        var nbTokens = tokens.count
        
        let start = Date()
        print("Start generating...")
        
        // Prepare model for generation.
        let cache = _prepareForGeneration(
            model: model,
            nbTokens: nbTokens,
            seqMax: maxTokens
        )
        
        // Update model's sequence.
        let modelTmp = Model.updateSeq(
            models: [model],
            sequence: 1,
            inPlace: true
        )[0]
        modelTmp.phase = .Inference
        modelTmp.updateKernel(batchSize: 1)
        
        // Set cache.
        firstLayer = modelTmp.layers.first as! EmbeddingSeq
        _setCache(
            model: modelTmp,
            cache: cache
        )
        
        var skip = 0
        tokens = [lastToken]
        
        // Decode.
        var sentence = decoder(tokens)
        _printChunk(sentence: sentence, skip: &skip)
        
        // Generate.
        let finalStep = maxTokens - nbTokens
        for _ in 0..<finalStep
        {
            // End generation.
            if lastToken == specialLastToken
            {
                break
            }
            
            // Forward.
            try! firstLayer.setDataGPU(
                [[lastToken]], batchSize: 1, sequence: 1
            )
            _updateRoPE(model: modelTmp, curSeq: nbTokens + 1)
            try! modelTmp.forward()
            
            // Get result.
            let out = (modelTmp.layers.last as! LayerSeq).outs.download()
            
            lastToken = _argmax(array: out)!
            tokens.append(lastToken)
            nbTokens += 1
            
            // Decode.
            sentence = decoder(tokens)
            _printChunk(sentence: sentence, skip: &skip)
        }
        
        // Decode.
        sentence = decoder(tokens)
        
        // Print.
        let rangeToPrint = sentence.index(
            sentence.startIndex, offsetBy: skip
        )..<sentence.endIndex
        let strToPrint = sentence[rangeToPrint]
        
        print(strToPrint)
        print("End generating.")
        
        let end = Date()
        let timeSpent = end.timeIntervalSince(start)
        print("Generation took \(timeSpent)s.")
    }
    
    /// Generate text from prompt with Mistral 7B Instruct.
    func testGenerateMistral() throws
    {
        let prompt = _prompt
        
        let nbBlocks = 32
        let hiddenDim = 4096
        let headDim = 128
        let mlpDim = 14336
        let nbHeadsQuery = 32
        let nbHeadsKV = 8
        let vocabularySize = 32768
        let maxTokens = 4096 // maximal number of tokens to generate
        
        // Load python objects.
        let pythonLib = Python.import("python_lib")
        let tokenizer = pythonLib.load_mistral_tokenizer(_modelPath)
        
        // Create encoder.
        let encoder = {
            (prompt: String) in
            
            return [Int](pythonLib.encode_mistral(
                prompt,
                tokenizer
            ))!
        }
        // Create decoder.
        let decoder = {
            (tokens: [Int]) in
            
            return String(pythonLib.decode_mistral(
                tokens,
                tokenizer
            ))!
        }
        
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
        
        // Generate.
        try generate(
            prompt: prompt,
            maxTokens: maxTokens,
            specialLastToken: 2,
            model: model,
            encoder: encoder,
            decoder: decoder
        )
    }
    
    /// Generate text from prompt with Metal Llama 2 7B Chat.
    func testGenerateLlama2() throws
    {
        let prompt = "How do you do?"
        
        let nbBlocks = 32
        let hiddenDim = 4096
        let headDim = 128
        let mlpDim = 11008
        let nbHeadsQuery = 32
        let nbHeadsKV = 32
        let vocabularySize = 32000
        let maxTokens = 4096 // maximal number of tokens to generate
        
        // Load python objects.
        let pythonLib = Python.import("python_lib")
        let tokenizer = pythonLib.load_llama2_tokenizer(_modelPath)
        
        // Create encoder.
        let encoder = {
            (prompt: String) in
            
            return [Int](pythonLib.encode_llama2(
                prompt,
                tokenizer
            ))!
        }
        // Create decoder.
        let decoder = {
            (tokens: [Int]) in
            
            return String(pythonLib.decode_llama2(
                tokens,
                tokenizer
            ))!
        }
        
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
        _loadLlama2Weights(model: model, keys: keys, weightsPath: _modelPath)
        
        // Generate.
        try generate(
            prompt: prompt,
            maxTokens: maxTokens,
            specialLastToken: 2,
            model: model,
            encoder: encoder,
            decoder: decoder
        )
    }
    
    /// Generate text from prompt with Metal Llama 3 8B Instruct.
    func testGenerateLlama3() throws
    {
        let prompt = _prompt
        
        let nbBlocks = 32
        let hiddenDim = 4096
        let headDim = 128
        let mlpDim = 14336
        let nbHeadsQuery = 32
        let nbHeadsKV = 8
        let vocabularySize = 128256
        let maxTokens = 4096 // maximal number of tokens to generate
        
        // Load python objects.
        let pythonLib = Python.import("python_lib")
        let tokenizer = pythonLib.load_llama2_tokenizer(_modelPath)
        
        // Create encoder.
        let encoder = {
            (prompt: String) in
            
            return [Int](pythonLib.encode_llama2(
                prompt,
                tokenizer
            ))!
        }
        // Create decoder.
        let decoder = {
            (tokens: [Int]) in
            
            return String(pythonLib.decode_llama2(
                tokens,
                tokenizer
            ))!
        }
        
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
        _loadLlama2Weights(model: model, keys: keys, weightsPath: _modelPath)
        
        // Generate.
        try generate(
            prompt: prompt,
            maxTokens: maxTokens,
            specialLastToken: 128009,
            model: model,
            encoder: encoder,
            decoder: decoder
        )
    }
}
