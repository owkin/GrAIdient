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
    let _prompt = "I"
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
    }
    
    ///
    /// Build LLM model.
    ///
    /// - Parameters:
    ///     - sequence: Length of the sequence.
    ///     - hiddenDim: Dimension of neurons in the main branch.
    ///     - nbHeads:  Number of heads (groups) of neurons.
    ///     - vocabularySize: Vocabulary size.
    /// - Returns: The model built.
    ///
    func _buildModel(
        modelPath: String,
        sequence: Int,
        hiddenDim: Int,
        nbHeads: Int,
        vocabularySize: Int) -> Model
    {
        let context = ModelContext(name: "NLP", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: LayerSeq = EmbeddingSeq(
            sequence: sequence,
            vocabularySize: vocabularySize,
            nbNeurons: hiddenDim, params: params
        )
        
        layer = FullyConnectedSeq(
            layerPrev: layer,
            nbNeurons: hiddenDim,
            activation: nil,
            biases: false,
            params: params
        )
        
        layer = try! RoPESeq(
            layerPrev: layer,
            seqPositions: [Int](1...sequence),
            nbHeads: nbHeads,
            params: params
        )
        
        /*layer = RMSNormSeq(
            layerPrev: layer,
            activation: nil,
            params: params
        )
        
        layer = FullyConnectedSeq(
            layerPrev: layer,
            nbNeurons: vocabularySize,
            activation: nil,
            biases: false,
            params: params
        )*/
        
        // Retrieve base model in the context and initialize a
        // real model (with `layerPrev` links updated).
        let model = Model(model: context.model, modelsPrev: [])
        
        // Load weights from `PyTorch`.
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.load_llm_weights(modelPath)
        var weightsNumpy = [PythonObject](data.tuple2.0)!
        
        // Apply weights on the `GrAIdient` model's layers.
        for num_layer in 0..<model.layers.count
        {
            // Load weights and biases.
            if let layer = model.layers[num_layer] as? EmbeddingSeq
            {
                let weightsTmp: [Float] = Array<Float>(
                    numpy: weightsNumpy.removeFirst()
                )!
                layer.weightsCPU = weightsTmp
                
                // TODO: remove this!
                weightsNumpy.removeFirst()
                weightsNumpy.removeFirst()
            }
            if let layer = model.layers[num_layer] as? RMSNormSeq
            {
                let weightsTmp: [Float] = Array<Float>(
                    numpy: weightsNumpy.removeFirst()
                )!
                layer.weightsCPU = weightsTmp
            }
            if let layer = model.layers[num_layer] as? FullyConnectedSeq
            {
                let weightsTmp: [Float] = Array<Float>(
                    numpy: weightsNumpy.removeFirst()
                )!
                layer.weightsCPU = weightsTmp
            }
        }
        return model
    }
    
    /// Generate text from prompt.
    func _testGenerate() throws
    {
        // Encode prompt.
        let pythonLib = Python.import("python_lib")
        let prompt = [Int](pythonLib.encode(
            _prompt,
            _modelPath
        ))!
        
        // Compute reference.
        let arrayRef = [Float](numpy: pythonLib.generate_main(
            _prompt,
            _modelPath
        ))!
        
        // Load pre trained model.
        let model = _buildModel(
            modelPath: _modelPath,
            sequence: prompt.count,
            hiddenDim: 4096,
            nbHeads: 32,
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
            let diffPercent = abs(elemOut - elemRef) / elemRef * 100.0
            XCTAssert(diffPercent < 1)
        }
    }
}
