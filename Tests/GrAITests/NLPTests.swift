//
// NLPTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 11/06/2024.
//

import XCTest
import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class NLPGradTests: EmbeddingSeqMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        
        optimizerParams.nbLoops = 2
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(_ model: String) -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(model: model, context: context)
        }
        return trainer
    }
    
    private func _buildModel(model: String, context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: LayerSeq = EmbeddingSeq(
            sequence: sequence,
            vocabularySize: vocabularySize,
            nbNeurons: 5, params: params
        )
        
        switch model
        {
        case "Embedding":
            break
            
        case "RMSNorm":
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                params: params
            )
            
        case "RoPE":
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 8,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! RoPESeq(
                layerPrev: layer,
                seqPositions: [Int](1...sequence),
                nbHeads: 3,
                params: params
            )
        
        case "QueryCausal1":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! QueryCausalSeq(
                query: layer, key: otherLayer,
                nbHeadsQuery: 3, nbHeadsKey: 3,
                params: params
            )
            layer = try! SoftmaxSeq(
                layerPrev: layer,
                nbHeads: 3,
                params: params
            )
            
        case "QueryCausal2":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 2 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 4 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! QueryCausalSeq(
                query: layer, key: otherLayer,
                nbHeadsQuery: 4, nbHeadsKey: 2,
                params: params
            )
            layer = try! SoftmaxSeq(
                layerPrev: layer,
                nbHeads: 4,
                params: params
            )
            
        case "ValueCausal1":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * sequence,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! ValueCausalSeq(
                value: layer, score: otherLayer,
                nbHeadsValue: 3, nbHeadsScore: 3,
                params: params
            )
            
        case "ValueCausal2":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 4 * sequence,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 2 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! ValueCausalSeq(
                value: layer, score: otherLayer,
                nbHeadsValue: 2, nbHeadsScore: 4,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layer, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testEmbeddingCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    func testEmbeddingGPU() throws
    {
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    func testEmbeddingSampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    func testRMSNormCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("RMSNorm")
        run(trainer)
    }
    
    func testRMSNormGPU() throws
    {
        let trainer = _buildTrainer("RMSNorm")
        run(trainer)
    }
    
    func testRoPECPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("RoPE")
        run(trainer)
    }
    
    func testRoPEGPU() throws
    {
        let trainer = _buildTrainer("RoPE")
        run(trainer)
    }
    
    func testQueryCausal1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer)
    }
    
    func testQueryCausal1GPU() throws
    {
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer)
    }
    
    func testQueryCausal2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer)
    }
    
    func testQueryCausal2GPU() throws
    {
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer)
    }
    
    func testValueCausal1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer)
    }
    
    func testValueCausal1GPU() throws
    {
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer)
    }
    
    func testValueCausal2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer)
    }
    
    func testValueCausal2GPU() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class NLPFlowTests: EmbeddingSeqMSE1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    func buildModel(model: String, context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: LayerSeq = EmbeddingSeq(
            sequence: sequence,
            vocabularySize: vocabularySize,
            nbNeurons: 5, params: params
        )
        
        switch model
        {
        case "Embedding":
            break
            
        case "RMSNorm":
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                params: params
            )
            
        case "RoPE":
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 8,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! RoPESeq(
                layerPrev: layer,
                seqPositions: [Int](1...sequence),
                nbHeads: 3,
                params: params
            )
            
        case "QueryCausal1":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! QueryCausalSeq(
                query: layer, key: otherLayer,
                nbHeadsQuery: 3, nbHeadsKey: 3,
                params: params
            )
            layer = try! SoftmaxSeq(
                layerPrev: layer,
                nbHeads: 3,
                params: params
            )
            
        case "QueryCausal2":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 2 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 4 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! QueryCausalSeq(
                query: layer, key: otherLayer,
                nbHeadsQuery: 4, nbHeadsKey: 2,
                params: params
            )
            layer = try! SoftmaxSeq(
                layerPrev: layer,
                nbHeads: 4,
                params: params
            )
            
        case "ValueCausal1":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * sequence,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! ValueCausalSeq(
                value: layer, score: otherLayer,
                nbHeadsValue: 3, nbHeadsScore: 3,
                params: params
            )
            
        case "ValueCausal2":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 4 * sequence,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 2 * 3,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! ValueCausalSeq(
                value: layer, score: otherLayer,
                nbHeadsValue: 2, nbHeadsScore: 4,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layer, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testEmbedding() throws
    {
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    func testEmbeddingSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    func testRMSNorm() throws
    {
        let trainer = _buildTrainer("RMSNorm")
        run(trainer)
    }
    
    func testRoPE() throws
    {
        let trainer = _buildTrainer("RoPE")
        run(trainer)
    }
    
    func testQueryCausal1() throws
    {
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer)
    }
    
    func testQueryCausal2() throws
    {
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer)
    }
    
    func testValueCausal1() throws
    {
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer)
    }
    
    func testValueCausal2() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class NLPFlowPrecisionTests: NLPFlowTests
{
    private func _buildTrainer(_ model: String) -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testEmbedding() throws
    {
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testEmbeddingSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testRMSNorm() throws
    {
        let trainer = _buildTrainer("RMSNorm")
        run(trainer)
    }
    
    override func testRoPE() throws
    {
        let trainer = _buildTrainer("RoPE")
        run(trainer)
    }
    
    override func testQueryCausal1() throws
    {
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testQueryCausal2() throws
    {
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testValueCausal1() throws
    {
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer)
    }
    
    override func testValueCausal2() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class NLP4FlowTests: EmbeddingSeqMSE1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    func buildModel(model: String, context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: LayerSeq = EmbeddingSeq(
            sequence: sequence,
            vocabularySize: vocabularySize,
            nbNeurons: 4, params: params
        )
        
        switch model
        {
        case "QueryCausal1":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 4,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 4,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! QueryCausalSeq(
                query: layer, key: otherLayer,
                nbHeadsQuery: 3, nbHeadsKey: 3,
                params: params
            )
            layer = try! SoftmaxSeq(
                layerPrev: layer,
                nbHeads: 3,
                params: params
            )
            
        case "QueryCausal2":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 2 * 4,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 4 * 4,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! QueryCausalSeq(
                query: layer, key: otherLayer,
                nbHeadsQuery: 4, nbHeadsKey: 2,
                params: params
            )
            layer = try! SoftmaxSeq(
                layerPrev: layer,
                nbHeads: 4,
                params: params
            )
            
        case "ValueCausal1":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * sequence,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 3 * 4,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! ValueCausalSeq(
                value: layer, score: otherLayer,
                nbHeadsValue: 3, nbHeadsScore: 3,
                params: params
            )
            
        case "ValueCausal2":
            let otherLayer: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 4 * sequence,
                activation: nil,
                biases: false,
                params: params
            )
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: 2 * 4,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! ValueCausalSeq(
                value: layer, score: otherLayer,
                nbHeadsValue: 2, nbHeadsScore: 4,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layer, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testQueryCausal1() throws
    {
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer)
    }
    
    func testQueryCausal2() throws
    {
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer)
    }
    
    func testValueCausal1() throws
    {
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer)
    }
    
    func testValueCausal2() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class NLP4FlowPrecisionTests: NLP4FlowTests
{
    private func _buildTrainer(_ model: String) -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testQueryCausal1() throws
    {
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testQueryCausal2() throws
    {
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testValueCausal1() throws
    {
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testValueCausal2() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer, diffThreshold: 0.002)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class NLPFlowResetTests: NLPFlowTests
{
    override func setUp()
    {
        super.setUp()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer(_ model: String) -> FlowResetTrainer
    {
        let trainer = FlowResetTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testEmbedding() throws
    {
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testEmbeddingSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testRMSNorm() throws
    {
        let trainer = _buildTrainer("RMSNorm")
        run(trainer)
    }
    
    override func testRoPE() throws
    {
        let trainer = _buildTrainer("RoPE")
        run(trainer)
    }
    
    override func testQueryCausal1() throws
    {
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer)
    }
    
    override func testQueryCausal2() throws
    {
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer)
    }
    
    override func testValueCausal1() throws
    {
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer)
    }
    
    override func testValueCausal2() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class NLPFlowReverseTests: NLPFlowTests
{
    override func setUp()
    {
        super.setUp()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer(_ model: String) -> FlowReverseTrainer
    {
        let trainer = FlowReverseTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testEmbedding() throws
    {
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testEmbeddingSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testRMSNorm() throws
    {
        let trainer = _buildTrainer("RMSNorm")
        run(trainer)
    }
    
    override func testRoPE() throws
    {
        let trainer = _buildTrainer("RoPE")
        run(trainer)
    }
    
    override func testQueryCausal1() throws
    {
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer)
    }
    
    override func testQueryCausal2() throws
    {
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer)
    }
    
    override func testValueCausal1() throws
    {
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer)
    }
    
    override func testValueCausal2() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class NLPFlowAccumulateTests: EmbeddingSeqMSE1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowAccumulateTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    func buildModel(model: String, context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: LayerSeq = EmbeddingSeq(
            sequence: sequence,
            vocabularySize: vocabularySize,
            nbNeurons: 5, params: params
        )
        
        switch model
        {
        case "Embedding":
            break
            
        case "RMSNorm":
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layer, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testEmbedding() throws
    {
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    func testEmbeddingSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    func testRMSNorm() throws
    {
        let trainer = _buildTrainer("RMSNorm")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class NLPInferenceTests: NLPFlowTests
{
    private func _buildTrainer(_ model: String) -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testEmbedding() throws
    {
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testEmbeddingSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testRMSNorm() throws
    {
        let trainer = _buildTrainer("RMSNorm")
        run(trainer)
    }
    
    override func testRoPE() throws
    {
        let trainer = _buildTrainer("RoPE")
        run(trainer)
    }
    
    override func testQueryCausal1() throws
    {
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer)
    }
    
    override func testQueryCausal2() throws
    {
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer)
    }
    
    override func testValueCausal1() throws
    {
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer)
    }
    
    override func testValueCausal2() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class NLPLoadTests: NLPFlowTests
{
    private func _buildTrainer(_ model: String) -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testEmbedding() throws
    {
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testEmbeddingSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testRMSNorm() throws
    {
        let trainer = _buildTrainer("RMSNorm")
        run(trainer)
    }
    
    override func testRoPE() throws
    {
        let trainer = _buildTrainer("RoPE")
        run(trainer)
    }
    
    override func testQueryCausal1() throws
    {
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer)
    }
    
    override func testQueryCausal2() throws
    {
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer)
    }
    
    override func testValueCausal1() throws
    {
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer)
    }
    
    override func testValueCausal2() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class NLPTransformTests: NLPFlowTests
{
    private func _buildTrainer(_ model: String) -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "NLP",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testEmbedding() throws
    {
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testEmbeddingSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Embedding")
        run(trainer)
    }
    
    override func testRMSNorm() throws
    {
        let trainer = _buildTrainer("RMSNorm")
        run(trainer)
    }
    
    override func testRoPE() throws
    {
        let trainer = _buildTrainer("RoPE")
        run(trainer)
    }
    
    override func testQueryCausal1() throws
    {
        throw XCTSkip("Skipping this test because of layer structure.")
        let trainer = _buildTrainer("QueryCausal1")
        run(trainer)
    }
    
    override func testQueryCausal2() throws
    {
        throw XCTSkip("Skipping this test because of layer structure.")
        let trainer = _buildTrainer("QueryCausal2")
        run(trainer)
    }
    
    override func testValueCausal1() throws
    {
        let trainer = _buildTrainer("ValueCausal1")
        run(trainer)
    }
    
    override func testValueCausal2() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer)
    }
}

// Test that generation process computes same outputs as forward pass.
class NLPGenerateTests: XCTestCase
{
    /// Length of the data sequence.
    let sequence = 5
    
    /// Initialize test.
    override func setUp()
    {
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
    func argmax(array: [Float]) -> Int?
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
    /// - Returns: The model built.
    ///
    func buildModel(
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
        
        var layer: LayerSeq = EmbeddingSeq(
            sequence: sequence,
            vocabularySize: vocabularySize,
            nbNeurons: hiddenDim, params: params
        )
        
        for _ in 0..<nbBlocks
        {
            var x: LayerSeq = layer
            
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                params: params
            )
            
            var query: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: nbHeadsQuery * headDim,
                activation: nil,
                biases: false,
                params: params
            )
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
            
            layer = try! SumSeq(layersPrev: [layer, x], params: params)
            
            x = layer
            
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                params: params
            )
            
            let mult1: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: mlpDim,
                activation: SiLU.str,
                biases: false,
                params: params
            )
            
            let mult2: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: mlpDim,
                activation: nil,
                biases: false,
                params: params
            )
            
            layer = try! MultiplySeq(layersPrev: [mult1, mult2], params: params)
            
            layer = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: hiddenDim,
                activation: nil,
                biases: false,
                params: params
            )
            
            layer = try! SumSeq(layersPrev: [layer, x], params: params)
        }
        
        layer = RMSNormSeq(
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
        )
        
        // Retrieve base model in the context and initialize a
        // real model (with `layerPrev` links updated).
        let model = Model(model: context.model, modelsPrev: [])
        return model
    }
    
    func testGenerate() throws
    {
        // Build models.
        let model1 = buildModel(
            sequence: sequence,
            nbBlocks: 1,
            hiddenDim: 8,
            headDim: 2,
            mlpDim: 8,
            nbHeadsQuery: 4,
            nbHeadsKV: 2,
            vocabularySize: 10
        )
        var model2 = buildModel(
            sequence: 2,
            nbBlocks: 1,
            hiddenDim: 8,
            headDim: 2,
            mlpDim: 8,
            nbHeadsQuery: 4,
            nbHeadsKV: 2,
            vocabularySize: 10
        )
        
        // Initialize for inference.
        model1.initKernel(phase: .Inference)
        model2.weights = model1.weights
        model2.initKernel(phase: .Inference)
        
        let firstLayer1 = model1.layers.first as! EmbeddingSeq
        var firstLayer2 = model2.layers.first as! EmbeddingSeq
        
        // Forward.
        model1.updateKernel(batchSize: 1)
        let prompt1 = [Int](0..<sequence)
        try! firstLayer1.setDataGPU(
            [prompt1], batchSize: 1, sequence: prompt1.count
        )
        try! model1.forward()
        
        // Get result.
        let out1 = (model1.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        var predictions1 = [Int]()
        for seq in 0..<out1.count / 10
        {
            let vector = [Float](out1[10*seq..<10*(seq+1)])
            let argmaxTmp = argmax(array: vector)!
            predictions1.append(argmaxTmp)
        }
        
        // Forward.
        model2.updateKernel(batchSize: 1)
        let prompt2 = [0, 1]
        
        try! firstLayer2.setDataGPU(
            [prompt2], batchSize: 1, sequence: prompt2.count
        )
        try! model2.forward()
        
        // Get result.
        let out2 = (model2.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        var predictions2 = [Int]()
        for seq in 0..<out2.count / 10
        {
            let vector = [Float](out2[10*seq..<10*(seq+1)])
            let argmaxTmp = argmax(array: vector)!
            predictions2.append(argmaxTmp)
        }
        
        var nbTokens = predictions2.count
        
        // Save cache.
        var cache = [Int: FloatBuffer]()
        for layer in model2.layers
        {
            let id = layer.id
            if let layerTmp = layer as? QueryCausalSeq
            {
                cache[id] = (layerTmp.layersPrev[1] as! LayerSeq).outs
                layerTmp.cacheSeq = nbTokens
                layerTmp.cacheSeqMax = 5
            }
            else if let layerTmp = layer as? SoftmaxCausalSeq
            {
                layerTmp.cacheSeq = nbTokens
            }
            else if let layerTmp = layer as? ValueCausalSeq
            {
                cache[id] = (layerTmp.layersPrev[0] as! LayerSeq).outs
                layerTmp.cacheSeq = nbTokens
                layerTmp.cacheSeqMax = 5
            }
        }
        
        // Prepare model for prediction of one token.
        model2 = Model.updateSeq(
            models: [model2],
            sequence: 1,
            inPlace: true
        )[0]
        
        model2.phase = .Inference
        model2.updateKernel(batchSize: 1)
        
        // Set cache.
        firstLayer2 = model2.layers.first as! EmbeddingSeq
        for layer in model2.layers
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
        for i in 0..<3
        {
            // Forward.
            try! firstLayer2.setDataGPU(
                [[prompt1[2 + i]]], batchSize: 1, sequence: 1
            )
            for layer in model2.layers
            {
                if let layerTmp = layer as? RoPESeq
                {
                    layerTmp.seqPositions = [nbTokens + 1]
                }
            }
            try! model2.forward()
            
            // Get result.
            let out2 = (model2.layers.last as! LayerSeq).outs.download()
            predictions2.append(argmax(array: out2)!)
            
            nbTokens += 1
        }
        
        print("Predictions1: \(predictions1).")
        print("Predictions2: \(predictions2).")
        XCTAssert(predictions1 == predictions2)
    }
}
