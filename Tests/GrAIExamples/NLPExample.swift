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
    let _modelPath = "TO/UPDATE"
    
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
    ///     - modelPath: Model path on the disk.
    ///     - sequence: Length of the sequence.
    ///     - nbBlocks: Number of transformer + MLP blocks.
    ///     - hiddenDim: Dimension of neurons in the main branch.
    ///     - headDim: Dimension of neurons in the transformer branches.
    ///     - mlpDim: Dimension of neurons in the MLP branches.
    ///     - nbHeads:  Number of heads (groups) of neurons for queries.
    ///     - nbHeadsKV: Number of heads (groups) of neurons for keys and values.
    ///     - vocabularySize: Vocabulary size.
    /// - Returns: The model built.
    ///
    func _buildModel(
        modelPath: String,
        sequence: Int,
        nbBlocks: Int,
        hiddenDim: Int,
        headDim: Int,
        mlpDim: Int,
        nbHeadsQuery: Int,
        nbHeadsKV: Int,
        vocabularySize: Int) -> Model
    {
        let context = ModelContext(name: "NLP", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var curPyTorch = 0
        var curGrAIdient = 0
        var dicoGrAIdient2PyTorch = [Int: Int]()
        
        var layer: LayerSeq = EmbeddingSeq(
            sequence: sequence,
            vocabularySize: vocabularySize,
            nbNeurons: hiddenDim, params: params
        )
        dicoGrAIdient2PyTorch[curGrAIdient] = curPyTorch
        curGrAIdient += 1
        curPyTorch += 1 + 2
        
        for _ in 0..<nbBlocks
        {
            var x: LayerSeq = layer
            
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                params: params
            )
            dicoGrAIdient2PyTorch[curGrAIdient] = curPyTorch + 4 + 3
            curGrAIdient += 1
            // curPyTorch += 1
            
            var query: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: nbHeadsQuery * headDim,
                activation: nil,
                biases: false,
                params: params
            )
            dicoGrAIdient2PyTorch[curGrAIdient] = curPyTorch
            curGrAIdient += 1
            curPyTorch += 1
            query = try! RoPESeq(
                layerPrev: query,
                seqPositions: [Int](1...sequence),
                nbHeads: nbHeadsQuery,
                params: params
            )
            curGrAIdient += 1
            
            var key: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: nbHeadsKV * headDim,
                activation: nil,
                biases: false,
                params: params
            )
            dicoGrAIdient2PyTorch[curGrAIdient] = curPyTorch
            curGrAIdient += 1
            curPyTorch += 1
            key = try! RoPESeq(
                layerPrev: key,
                seqPositions: [Int](1...sequence),
                nbHeads: nbHeadsKV,
                params: params
            )
            curGrAIdient += 1
            
            let value: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: nbHeadsKV * headDim,
                activation: nil,
                biases: false,
                params: params
            )
            dicoGrAIdient2PyTorch[curGrAIdient] = curPyTorch
            curGrAIdient += 1
            curPyTorch += 1
            
            layer = try! QueryCausalSeq(
                query: query, key: key,
                nbHeadsQuery: nbHeadsQuery, nbHeadsKey: nbHeadsKV,
                params: params
            )
            curGrAIdient += 1
            layer = try! SoftmaxCausalSeq(
                layerPrev: layer,
                nbHeads: nbHeadsQuery,
                params: params
            )
            curGrAIdient += 1
            
            layer = try! ValueCausalSeq(
                value: value, score: layer,
                nbHeadsValue: nbHeadsKV, nbHeadsScore: nbHeadsQuery,
                params: params
            )
            curGrAIdient += 1
            
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: nbHeadsQuery * headDim,
                activation: nil,
                biases: false,
                params: params
            )
            dicoGrAIdient2PyTorch[curGrAIdient] = curPyTorch
            curGrAIdient += 1
            curPyTorch += 1
            
            layer = try! SumSeq(layersPrev: [layer, x], params: params)
            curGrAIdient += 1
            
            x = layer
            
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                params: params
            )
            dicoGrAIdient2PyTorch[curGrAIdient] = curPyTorch + 3 + 1
            curGrAIdient += 1
            // curPyTorch += 1
            
            let mult1: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: mlpDim,
                activation: SiLU.str,
                biases: false,
                params: params
            )
            dicoGrAIdient2PyTorch[curGrAIdient] = curPyTorch
            curGrAIdient += 1
            curPyTorch += 1
            
            let mult2: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: mlpDim,
                activation: nil,
                biases: false,
                params: params
            )
            dicoGrAIdient2PyTorch[curGrAIdient] = curPyTorch + 1
            curGrAIdient += 1
            // curPyTorch += 1
            
            layer = try! MultiplySeq(layersPrev: [mult1, mult2], params: params)
            curGrAIdient += 1
            
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: hiddenDim,
                activation: nil,
                biases: false,
                params: params
            )
            dicoGrAIdient2PyTorch[curGrAIdient] = curPyTorch
            curGrAIdient += 1
            curPyTorch += 2
            
            layer = try! SumSeq(layersPrev: [layer, x], params: params)
            curGrAIdient += 1
            
            curPyTorch += 2
        }
        
        layer = RMSNormSeq(
            layerPrev: layer,
            activation: nil,
            params: params
        )
        dicoGrAIdient2PyTorch[curGrAIdient] = 1
        curGrAIdient += 1
        // curPyTorch += 1
        
        layer = FullyConnectedSeq(
            layerPrev: layer,
            nbNeurons: vocabularySize,
            activation: nil,
            biases: false,
            params: params
        )
        dicoGrAIdient2PyTorch[curGrAIdient] = 2
        curGrAIdient += 1
        // curPyTorch += 1
        
        // Retrieve base model in the context and initialize a
        // real model (with `layerPrev` links updated).
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_llm_weights(modelPath)
        var weightsNumpy: [PythonObject?] = [PythonObject](data.tuple2.0)!
        
        // Apply weights on the `GrAIdient` model's layers.
        for layer in model.layers
        {
            // Load weights and biases.
            if let layerTmp = layer as? EmbeddingSeq
            {
                let idGrAIdient = layerTmp.id
                let idPyTorch = dicoGrAIdient2PyTorch[idGrAIdient]!
                
                let weightsTmp: [Float] = Array<Float>(
                    numpy: weightsNumpy[idPyTorch]!
                )!
                layerTmp.weightsCPU = weightsTmp
                
                weightsNumpy[idPyTorch] = nil
            }
            if let layerTmp = layer as? RMSNormSeq
            {
                let idGrAIdient = layerTmp.id
                let idPyTorch = dicoGrAIdient2PyTorch[idGrAIdient]!
                
                let weightsTmp: [Float] = Array<Float>(
                    numpy: weightsNumpy[idPyTorch]!
                )!
                layerTmp.weightsCPU = weightsTmp
                
                weightsNumpy[idPyTorch] = nil
            }
            if let layerTmp = layer as? FullyConnectedSeq
            {
                let idGrAIdient = layerTmp.id
                let idPyTorch = dicoGrAIdient2PyTorch[idGrAIdient]!
                
                let weightsTmp: [Float] = Array<Float>(
                    numpy: weightsNumpy[idPyTorch]!
                )!
                layerTmp.weightsCPU = weightsTmp
                
                weightsNumpy[idPyTorch] = nil
            }
        }
        return model
    }
    
    /// Predict text from prompt.
    func _testPredict1() throws
    {
        // Encode prompt.
        let pythonLib = Python.import("python_lib")
        let prompt = [Int](pythonLib.encode(
            _prompt,
            _modelPath
        ))!
        
        // Compute reference.
        let arrayRef = [Float](numpy: pythonLib.predict(
            _prompt,
            _modelPath
        ))!
        
        // Load pre trained model.
        let model = _buildModel(
            modelPath: _modelPath,
            sequence: prompt.count,
            nbBlocks: 1,
            hiddenDim: 4096,
            headDim: 128,
            mlpDim: 14336,
            nbHeadsQuery: 32,
            nbHeadsKV: 8,
            vocabularySize: 32000
        )
        
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
        // Encode prompt.
        let pythonLib = Python.import("python_lib")
        let prompt = [Int](pythonLib.encode(
            _prompt,
            _modelPath
        ))!
        
        // Load pre trained model.
        let model = _buildModel(
            modelPath: _modelPath,
            sequence: prompt.count,
            nbBlocks: 32,
            hiddenDim: 4096,
            headDim: 128,
            mlpDim: 14336,
            nbHeadsQuery: 32,
            nbHeadsKV: 8,
            vocabularySize: 32000
        )
        
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
        var predictions = [Int]()
        for seq in 0..<out.count / 32000
        {
            let vector = [Float](out[32000*seq..<32000*(seq+1)])
            let argmax = _argmax(array: vector)!
            predictions.append(argmax)
        }
        
        // Decode.
        let prediction = String(pythonLib.decode(
            predictions,
            _modelPath
        ))!
        print(prediction)
    }
    
    /// Generate text from prompt.
    func _testGenerate() throws
    {
        // Use Float16 for faster results.
        GrAI.Precision.float16 = true
        
        // Encode prompt.
        let pythonLib = Python.import("python_lib")
        let prompt = [Int](pythonLib.encode(
            _prompt,
            _modelPath
        ))!
        
        // Load pre trained model.
        var model = _buildModel(
            modelPath: _modelPath,
            sequence: prompt.count,
            nbBlocks: 32,
            hiddenDim: 4096,
            headDim: 128,
            mlpDim: 14336,
            nbHeadsQuery: 32,
            nbHeadsKV: 8,
            vocabularySize: 32000
        )
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        model.updateKernel(batchSize: 1)
        
        // Forward.
        var firstLayer: EmbeddingSeq = model.layers.first as! EmbeddingSeq
        try! firstLayer.setDataGPU(
            [prompt], batchSize: 1, sequence: prompt.count
        )
        try! model.forward()
        
        // Get result.
        let out = (model.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        var predictions = [Int]()
        for seq in 0..<out.count / 32000
        {
            let vector = [Float](out[32000*seq..<32000*(seq+1)])
            let argmax = _argmax(array: vector)!
            predictions.append(argmax)
        }
        
        var lastToken = predictions.last!
        var nbTokens = predictions.count
        
        // Decode.
        let prediction = String(pythonLib.decode(
            [lastToken],
            _modelPath
        ))!
        
        let start = Date()
        print("Start generating...")
        print(prediction, terminator: "")
        
        // Save cache.
        var cache = [Int: FloatBuffer]()
        for layer in model.layers
        {
            let id = layer.id
            if let layerTmp = layer as? QueryCausalSeq
            {
                cache[id] = (layerTmp.layersPrev[1] as! LayerSeq).outs
                layerTmp.cacheSeq = nbTokens
            }
            else if let layerTmp = layer as? SoftmaxCausalSeq
            {
                layerTmp.cacheSeq = nbTokens
            }
            else if let layerTmp = layer as? ValueCausalSeq
            {
                cache[id] = (layerTmp.layersPrev[0] as! LayerSeq).outs
                layerTmp.cacheSeq = nbTokens
            }
        }
        
        // Prepare model for prediction of one token.
        model = Model.updateSeq(
            models: [model],
            sequence: 1,
            inPlace: true
        )[0]
        
        model.phase = .Inference
        model.updateKernel(batchSize: 1)
        
        // Set cache.
        firstLayer = model.layers.first as! EmbeddingSeq
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
        
        // Generate.
        let finalStep = 128 - nbTokens
        for _ in 0..<finalStep
        {
            // Forward.
            try! firstLayer.setDataGPU(
                [[lastToken]], batchSize: 1, sequence: 1
            )
            for layer in model.layers
            {
                if let layerTmp = layer as? RoPESeq
                {
                    layerTmp.seqPositions = [nbTokens + 1]
                }
            }
            try! model.forward()
            
            // Get result.
            let out = (model.layers.last as! LayerSeq).outs.download()
            let predictions = [_argmax(array: out)!]
            
            lastToken = predictions.last!
            nbTokens += 1
            
            // Decode.
            let prediction = String(pythonLib.decode(
                predictions,
                _modelPath
            ))!
            print(prediction, terminator: "")
        }
        print("")
        print("End generating.")
        
        let end = Date()
        let timeSpent = end.timeIntervalSince(start)
        print("Process took \(timeSpent)s.")
    }
}
