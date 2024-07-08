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
