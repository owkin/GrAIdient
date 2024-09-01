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
            
        case "RMSNorm1":
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                addUnitOffset: false,
                params: params
            )
            
        case "RMSNorm2":
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                addUnitOffset: true,
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
    
    func testRMSNorm1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("RMSNorm1")
        run(trainer)
    }
    
    func testRMSNorm1GPU() throws
    {
        let trainer = _buildTrainer("RMSNorm1")
        run(trainer)
    }
    
    func testRMSNorm2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("RMSNorm2")
        run(trainer)
    }
    
    func testRMSNorm2GPU() throws
    {
        let trainer = _buildTrainer("RMSNorm2")
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
            
        case "RMSNorm1":
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                addUnitOffset: false,
                params: params
            )
            
        case "RMSNorm2":
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                addUnitOffset: true,
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
    
    func testRMSNorm1() throws
    {
        let trainer = _buildTrainer("RMSNorm1")
        run(trainer)
    }
    
    func testRMSNorm2() throws
    {
        let trainer = _buildTrainer("RMSNorm2")
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
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testEmbeddingSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Embedding")
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testRMSNorm1() throws
    {
        let trainer = _buildTrainer("RMSNorm1")
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testRMSNorm2() throws
    {
        let trainer = _buildTrainer("RMSNorm2")
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testRoPE() throws
    {
        let trainer = _buildTrainer("RoPE")
        run(trainer, diffThreshold: 0.002)
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
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testValueCausal2() throws
    {
        let trainer = _buildTrainer("ValueCausal2")
        run(trainer, diffThreshold: 0.005)
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
    
    override func testRMSNorm1() throws
    {
        let trainer = _buildTrainer("RMSNorm1")
        run(trainer)
    }
    
    override func testRMSNorm2() throws
    {
        let trainer = _buildTrainer("RMSNorm2")
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
    
    override func testRMSNorm1() throws
    {
        let trainer = _buildTrainer("RMSNorm1")
        run(trainer)
    }
    
    override func testRMSNorm2() throws
    {
        let trainer = _buildTrainer("RMSNorm2")
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
            
        case "RMSNorm1":
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                addUnitOffset: false,
                params: params
            )
            
        case "RMSNorm2":
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                addUnitOffset: true,
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
    
    func testRMSNorm1() throws
    {
        let trainer = _buildTrainer("RMSNorm1")
        run(trainer)
    }
    
    func testRMSNorm2() throws
    {
        let trainer = _buildTrainer("RMSNorm2")
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
    
    override func testRMSNorm1() throws
    {
        let trainer = _buildTrainer("RMSNorm1")
        run(trainer)
    }
    
    override func testRMSNorm2() throws
    {
        let trainer = _buildTrainer("RMSNorm2")
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
    
    override func testRMSNorm1() throws
    {
        let trainer = _buildTrainer("RMSNorm1")
        run(trainer)
    }
    
    override func testRMSNorm2() throws
    {
        let trainer = _buildTrainer("RMSNorm2")
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
    
    override func testRMSNorm1() throws
    {
        let trainer = _buildTrainer("RMSNorm1")
        run(trainer)
    }
    
    override func testRMSNorm2() throws
    {
        let trainer = _buildTrainer("RMSNorm2")
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
    ///     - addUnitOffset: Whether to add unit offset or not in RMSNorm.
    ///     - hiddentActivation: Activation function.
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
        vocabularySize: Int,
        addUnitOffset: Bool,
        hiddenActivation: String) -> Model
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
                addUnitOffset: addUnitOffset,
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
                nbNeurons: hiddenDim,
                activation: nil,
                biases: false,
                params: params
            )
            
            layer = try! SumSeq(layersPrev: [layer, x], params: params)
            
            x = layer
            
            layer = RMSNormSeq(
                layerPrev: layer,
                activation: nil,
                addUnitOffset: addUnitOffset,
                params: params
            )
            
            let mult1: LayerSeq = FullyConnectedSeq(
                layerPrev: layer,
                nbNeurons: mlpDim,
                activation: hiddenActivation,
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
            addUnitOffset: addUnitOffset,
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
    
    ///
    /// Prepare model for generation.
    ///
    /// - Parameters:
    ///     - model: Model.
    ///     - nbTokens: Number of tokens which have been generated.
    ///     - seqMax: Maximal number of tokens to generate.
    /// - Returns: The cache.
    ///
    func prepareForGeneration(
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
                layerTmp.cacheSeqMax = seqMax
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
    func setCache(
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
    func updateRoPE(model: Model, curSeq: Int)
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
    /// Predict tokens from prompt with two ways.
    /// 1. Use end to end forward pass.
    /// 2. Use partial end to end forward pass followed by generation one token at a time.
    ///
    func runGenerate(
        addUnitOffset: Bool,
        hiddenActivation: String)
    {
        let nbBlocks = 1
        let hiddenDim = 8
        let headDim = 2
        let mlpDim = 8
        let nbHeadsQuery = 4
        let nbHeadsKV = 2
        let vocabularySize = 10
        let maxTokens = 5 // maximal number of tokens to generate
        let tmpSeq = 2 // partial forward step
        
        // Build models.
        let model1 = buildModel(
            sequence: sequence,
            nbBlocks: nbBlocks,
            hiddenDim: hiddenDim,
            headDim: headDim,
            mlpDim: mlpDim,
            nbHeadsQuery: nbHeadsQuery,
            nbHeadsKV: nbHeadsKV,
            vocabularySize: vocabularySize,
            addUnitOffset: addUnitOffset,
            hiddenActivation: hiddenActivation
        )
        var model2 = buildModel(
            sequence: tmpSeq,
            nbBlocks: nbBlocks,
            hiddenDim: hiddenDim,
            headDim: headDim,
            mlpDim: mlpDim,
            nbHeadsQuery: nbHeadsQuery,
            nbHeadsKV: nbHeadsKV,
            vocabularySize: vocabularySize,
            addUnitOffset: addUnitOffset,
            hiddenActivation: hiddenActivation
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
            [prompt1], batchSize: 1, sequence: sequence
        )
        try! model1.forward()
        
        // Get result.
        let out1 = (model1.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        var predictions1 = [Int]()
        for seq in 0..<out1.count / vocabularySize
        {
            let vector = [Float](
                out1[vocabularySize*seq..<vocabularySize*(seq+1)]
            )
            let argmaxTmp = argmax(array: vector)!
            predictions1.append(argmaxTmp)
        }
        
        // Forward.
        model2.updateKernel(batchSize: 1)
        let prompt2 = [Int](prompt1[0..<tmpSeq])
        
        try! firstLayer2.setDataGPU(
            [prompt2], batchSize: 1, sequence: tmpSeq
        )
        try! model2.forward()
        
        // Get result.
        let out2 = (model2.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        var predictions2 = [Int]()
        for seq in 0..<out2.count / vocabularySize
        {
            let vector = [Float](
                out2[vocabularySize*seq..<vocabularySize*(seq+1)]
            )
            let argmaxTmp = argmax(array: vector)!
            predictions2.append(argmaxTmp)
        }
        
        var nbTokens = predictions2.count
        
        // Prepare model for generation.
        let cache = prepareForGeneration(
            model: model2,
            nbTokens: nbTokens,
            seqMax: maxTokens
        )
        
        // Update model's sequence.
        model2 = Model.updateSeq(
            models: [model2],
            sequence: 1,
            inPlace: true
        )[0]
        model2.phase = .Inference
        model2.updateKernel(batchSize: 1)
        
        // Set cache.
        firstLayer2 = model2.layers.first as! EmbeddingSeq
        setCache(
            model: model2,
            cache: cache
        )
        
        // Generate.
        let finalStep = maxTokens - nbTokens
        for i in 0..<finalStep
        {
            // Forward.
            try! firstLayer2.setDataGPU(
                [[prompt1[tmpSeq + i]]], batchSize: 1, sequence: 1
            )
            updateRoPE(model: model2, curSeq: nbTokens + 1)
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
    
    ///
    /// Predict tokens from prompt with two ways, using batch size greater than 1.
    /// 1. Use end to end forward pass.
    /// 2. Use partial end to end forward pass followed by generation one token at a time.
    ///
    func runGenerateBatchSize(
        addUnitOffset: Bool,
        hiddenActivation: String)
    {
        let nbBlocks = 1
        let hiddenDim = 8
        let headDim = 2
        let mlpDim = 8
        let nbHeadsQuery = 4
        let nbHeadsKV = 2
        let vocabularySize = 10
        let maxTokens = 5 // maximal number of tokens to generate
        let tmpSeq = 2 // partial forward step
        
        // Build models.
        let model1 = buildModel(
            sequence: sequence,
            nbBlocks: nbBlocks,
            hiddenDim: hiddenDim,
            headDim: headDim,
            mlpDim: mlpDim,
            nbHeadsQuery: nbHeadsQuery,
            nbHeadsKV: nbHeadsKV,
            vocabularySize: vocabularySize,
            addUnitOffset: addUnitOffset,
            hiddenActivation: hiddenActivation
        )
        var model2 = buildModel(
            sequence: tmpSeq,
            nbBlocks: nbBlocks,
            hiddenDim: hiddenDim,
            headDim: headDim,
            mlpDim: mlpDim,
            nbHeadsQuery: nbHeadsQuery,
            nbHeadsKV: nbHeadsKV,
            vocabularySize: vocabularySize,
            addUnitOffset: addUnitOffset,
            hiddenActivation: hiddenActivation
        )
        
        // Initialize for inference.
        model1.initKernel(phase: .Inference)
        model2.weights = model1.weights
        model2.initKernel(phase: .Inference)
        
        let firstLayer1 = model1.layers.first as! EmbeddingSeq
        var firstLayer2 = model2.layers.first as! EmbeddingSeq
        
        // Forward.
        model1.updateKernel(batchSize: 2)
        let prompt1 = [Int](0..<sequence)
        let prompt2 = [Int](prompt1.reversed())
        
        try! firstLayer1.setDataGPU(
            [prompt1, prompt2], batchSize: 2, sequence: sequence
        )
        try! model1.forward()
        
        // Get result.
        let out1 = (model1.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        var predictions1 = [Int]()
        for seq in 0..<out1.count / vocabularySize
        {
            let vector = [Float](
                out1[vocabularySize*seq..<vocabularySize*(seq+1)]
            )
            let argmaxTmp = argmax(array: vector)!
            predictions1.append(argmaxTmp)
        }
        
        // Forward.
        model2.updateKernel(batchSize: 2)
        let prompt3 = [Int](prompt1[0..<tmpSeq])
        let prompt4 = [Int](prompt2[0..<tmpSeq])
        
        try! firstLayer2.setDataGPU(
            [prompt3, prompt4], batchSize: 2, sequence: tmpSeq
        )
        try! model2.forward()
        
        // Get result.
        let out2 = (model2.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        var predictions2 = [Int](repeating: 0, count: 2 * sequence)
        for seq in 0..<out2.count / vocabularySize
        {
            let vector = [Float](
                out2[vocabularySize*seq..<vocabularySize*(seq+1)]
            )
            let argmaxTmp = argmax(array: vector)!
            
            let offset = seq % tmpSeq + (seq / tmpSeq) * sequence
            predictions2[offset] = argmaxTmp
        }
        
        var nbTokens = tmpSeq
        
        // Prepare model for generation.
        let cache = prepareForGeneration(
            model: model2,
            nbTokens: nbTokens,
            seqMax: maxTokens
        )
        
        // Update model's sequence.
        model2 = Model.updateSeq(
            models: [model2],
            sequence: 1,
            inPlace: true
        )[0]
        model2.phase = .Inference
        model2.updateKernel(batchSize: 2)
        
        // Set cache.
        firstLayer2 = model2.layers.first as! EmbeddingSeq
        setCache(
            model: model2,
            cache: cache
        )
        
        // Generate.
        let finalStep = maxTokens - nbTokens
        for i in 0..<finalStep
        {
            // Forward.
            try! firstLayer2.setDataGPU(
                [[prompt1[tmpSeq + i]], [prompt2[tmpSeq + i]]],
                batchSize: 2, sequence: 1
            )
            updateRoPE(model: model2, curSeq: nbTokens + 1)
            try! model2.forward()
            
            // Get result.
            let out2 = (model2.layers.last as! LayerSeq).outs.download()
            
            // Compute prediction for each token.
            for seq in 0..<out2.count / vocabularySize
            {
                let vector = [Float](
                    out2[vocabularySize*seq..<vocabularySize*(seq+1)]
                )
                let argmaxTmp = argmax(array: vector)!
                
                let offset = tmpSeq + i + (seq % 2) * sequence
                predictions2[offset] = argmaxTmp
            }
            
            nbTokens += 1
        }
        
        print("Predictions1: \(predictions1).")
        print("Predictions2: \(predictions2).")
        XCTAssert(predictions1 == predictions2)
    }
    
    /// Predict tokens with sliding window.
    func runGenerateSlidingWindow(
        addUnitOffset: Bool,
        hiddenActivation: String)
    {
        let nbBlocks = 1
        let hiddenDim = 8
        let headDim = 2
        let mlpDim = 8
        let nbHeadsQuery = 4
        let nbHeadsKV = 2
        let vocabularySize = 10
        let maxTokens = 5 // maximal number of tokens to generate
        let tmpSeq = 2 // partial forward step
        
        // Build models.
        var model = buildModel(
            sequence: tmpSeq,
            nbBlocks: nbBlocks,
            hiddenDim: hiddenDim,
            headDim: headDim,
            mlpDim: mlpDim,
            nbHeadsQuery: nbHeadsQuery,
            nbHeadsKV: nbHeadsKV,
            vocabularySize: vocabularySize,
            addUnitOffset: addUnitOffset,
            hiddenActivation: hiddenActivation
        )
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        
        var firstLayer = model.layers.first as! EmbeddingSeq
        let prompt1 = [Int](0..<sequence)
        
        // Forward.
        model.updateKernel(batchSize: 1)
        let prompt2 = [Int](prompt1[0..<tmpSeq])
        
        try! firstLayer.setDataGPU(
            [prompt2], batchSize: 1, sequence: tmpSeq
        )
        try! model.forward()
        
        // Get result.
        let out = (model.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        var tokens = [Int]()
        for seq in 0..<out.count / vocabularySize
        {
            let vector = [Float](
                out[vocabularySize*seq..<vocabularySize*(seq+1)]
            )
            let argmaxTmp = argmax(array: vector)!
            tokens.append(argmaxTmp)
        }
        
        var lastToken = tokens.last!
        var nbTokens = tokens.count
        
        // Prepare model for generation.
        let cache = prepareForGeneration(
            model: model,
            nbTokens: nbTokens,
            seqMax: maxTokens
        )
        
        // Update model's sequence.
        model = Model.updateSeq(
            models: [model],
            sequence: 1,
            inPlace: true
        )[0]
        model.phase = .Inference
        model.updateKernel(batchSize: 1)
        
        // Set cache.
        firstLayer = model.layers.first as! EmbeddingSeq
        setCache(
            model: model,
            cache: cache
        )
        
        var scoreLayer: LayerSeq! = nil
        for layer in model.layers
        {
            if let layerTmp = layer as? ValueCausalSeq
            {
                scoreLayer = layerTmp.layersPrev[1] as? LayerSeq
                break
            }
        }
        
        // Generate.
        let finalStep = 2 * maxTokens - nbTokens
        for i in 0..<finalStep
        {
            // Forward.
            try! firstLayer.setDataGPU(
                [[lastToken]], batchSize: 1, sequence: 1
            )
            updateRoPE(model: model, curSeq: nbTokens + 1)
            try! model.forward()
            
            // Test that sum of scores equal to 1.
            let scores = scoreLayer.outs.download()
            var sum = 0.0
            for (j, score) in scores.enumerated()
            {
                sum += Double(score)
                
                // Every seqK is not yet used: we still have 0.0 in the
                // context cache.
                if (j + 1) == scores.count && i < maxTokens - tmpSeq - 1
                {
                    XCTAssert(sum == 0.0)
                }
                // Every seqK is used: there should not be any 0.0 as
                // the context cache is full.
                else if (j + 1) == scores.count
                {
                    let value = round(sum * 100) / 100.0
                    XCTAssert(value == 1.0)
                }
                // Nominal case, we are feeding `sum`.
                else if (j + 1) % (min(nbTokens + 1, maxTokens)) == 0
                {
                    if sum != 0.0
                    {
                        let value = round(sum * 100) / 100.0
                        XCTAssert(value == 1.0)
                    }
                    sum = 0.0
                }
            }
            
            // Get result.
            let out = (model.layers.last as! LayerSeq).outs.download()
            
            lastToken = argmax(array: out)!
            tokens.append(lastToken)
            nbTokens += 1
        }
        print("Tokens: \(tokens).")
    }
    
    /// Predict tokens with sliding window and batch size greater than 1.
    func runGenerateSlidingWindowBatchSize(
        addUnitOffset: Bool,
        hiddenActivation: String)
    {
        let nbBlocks = 1
        let hiddenDim = 8
        let headDim = 2
        let mlpDim = 8
        let nbHeadsQuery = 4
        let nbHeadsKV = 2
        let vocabularySize = 10
        let maxTokens = 5 // maximal number of tokens to generate
        let tmpSeq = 2 // partial forward step
        
        // Build models.
        var model = buildModel(
            sequence: tmpSeq,
            nbBlocks: nbBlocks,
            hiddenDim: hiddenDim,
            headDim: headDim,
            mlpDim: mlpDim,
            nbHeadsQuery: nbHeadsQuery,
            nbHeadsKV: nbHeadsKV,
            vocabularySize: vocabularySize,
            addUnitOffset: addUnitOffset,
            hiddenActivation: hiddenActivation
        )
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        
        var firstLayer = model.layers.first as! EmbeddingSeq
        let prompt1 = [Int](0..<sequence)
        let prompt2 = [Int](prompt1.reversed())
        
        // Forward.
        model.updateKernel(batchSize: 2)
        let prompt3 = [Int](prompt1[0..<tmpSeq])
        let prompt4 = [Int](prompt2[0..<tmpSeq])
        
        try! firstLayer.setDataGPU(
            [prompt3, prompt4], batchSize: 2, sequence: tmpSeq
        )
        try! model.forward()
        
        // Get result.
        let out = (model.layers.last as! LayerSeq).outs.download()
        
        // Compute prediction for each token.
        var tokens = [Int](repeating: 0, count: 4 * maxTokens)
        for seq in 0..<out.count / vocabularySize
        {
            let vector = [Float](
                out[vocabularySize*seq..<vocabularySize*(seq+1)]
            )
            let argmaxTmp = argmax(array: vector)!
            
            let offset = seq % tmpSeq + (seq / tmpSeq) * 2 * maxTokens
            tokens[offset] = argmaxTmp
        }
        
        var lastToken1 = tokens[tmpSeq-1]
        var lastToken2 = tokens[2 * maxTokens + tmpSeq - 1]
        var nbTokens = tmpSeq
        
        // Prepare model for generation.
        let cache = prepareForGeneration(
            model: model,
            nbTokens: nbTokens,
            seqMax: maxTokens
        )
        
        // Update model's sequence.
        model = Model.updateSeq(
            models: [model],
            sequence: 1,
            inPlace: true
        )[0]
        model.phase = .Inference
        model.updateKernel(batchSize: 2)
        
        // Set cache.
        firstLayer = model.layers.first as! EmbeddingSeq
        setCache(
            model: model,
            cache: cache
        )
        
        var score1Layer: LayerSeq! = nil
        for layer in model.layers
        {
            if let layerTmp = layer as? QueryCausalSeq
            {
                score1Layer = layerTmp
                break
            }
        }
        
        var score2Layer: LayerSeq! = nil
        for layer in model.layers
        {
            if let layerTmp = layer as? ValueCausalSeq
            {
                score2Layer = layerTmp.layersPrev[1] as? LayerSeq
                break
            }
        }
        
        // Generate.
        let finalStep = 2 * maxTokens - nbTokens
        for i in 0..<finalStep
        {
            // Forward.
            try! firstLayer.setDataGPU(
                [[lastToken1], [lastToken2]], batchSize: 2, sequence: 1
            )
            updateRoPE(model: model, curSeq: nbTokens + 1)
            try! model.forward()
            
            // Test that all scores are set when the context cache is full.
            var scores = score1Layer.outs.download()
            if i >= maxTokens - tmpSeq - 1
            {
                for score in scores
                {
                    XCTAssert(score != 0.0)
                }
            }
            
            // Test that sum of scores equal to 1.
            scores = score2Layer.outs.download()
            var sum = 0.0
            for (j, score) in scores.enumerated()
            {
                sum += Double(score)
                
                // Every seqK is not yet used: we still have 0.0 in the
                // context cache.
                if (j + 1) == scores.count && i < maxTokens - tmpSeq - 1
                {
                    XCTAssert(sum == 0.0)
                }
                // Every seqK is used: there should not be any 0.0 as
                // the context cache is full.
                else if (j + 1) == scores.count
                {
                    let value = round(sum * 100) / 100.0
                    XCTAssert(value == 1.0)
                }
                // Nominal case, we are feeding `sum`.
                else if (j + 1) % (min(nbTokens + 1, maxTokens)) == 0
                {
                    if sum != 0.0
                    {
                        let value = round(sum * 100) / 100.0
                        XCTAssert(value == 1.0)
                    }
                    sum = 0.0
                }
            }
            
            // Get result.
            let out = (model.layers.last as! LayerSeq).outs.download()
            
            // Compute prediction for each token.
            for seq in 0..<out.count / vocabularySize
            {
                let vector = [Float](
                    out[vocabularySize*seq..<vocabularySize*(seq+1)]
                )
                let argmaxTmp = argmax(array: vector)!
                
                let offset = tmpSeq + i + (seq % 2) * 2 * maxTokens
                tokens[offset] = argmaxTmp
            }
            
            lastToken1 = tokens[tmpSeq + i]
            lastToken2 = tokens[tmpSeq + i + 2 * maxTokens]
            
            nbTokens += 1
        }
        print("Tokens: \(tokens).")
    }
    
    func testGenerate1Float()
    {
        runGenerate(addUnitOffset: false, hiddenActivation: SiLU.str)
    }
    
    func testGenerate2Float()
    {
        runGenerate(addUnitOffset: true, hiddenActivation: GELUApprox.str)
    }
    
    func testGenerate1Float16() throws
    {
        GrAI.Precision.float16 = true
        runGenerate(addUnitOffset: false, hiddenActivation: SiLU.str)
    }
    
    func testGenerate2Float16() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        GrAI.Precision.float16 = true
        runGenerate(addUnitOffset: true, hiddenActivation: GELUApprox.str)
    }
    
    func testGenerateBatchSize1Float()
    {
        runGenerateBatchSize(addUnitOffset: false, hiddenActivation: SiLU.str)
    }
    
    func testGenerateBatchSize2Float()
    {
        runGenerateBatchSize(
            addUnitOffset: true, hiddenActivation: GELUApprox.str
        )
    }
    
    func testGenerateBatchSize1Float16() throws
    {
        GrAI.Precision.float16 = true
        runGenerateBatchSize(addUnitOffset: false, hiddenActivation: SiLU.str)
    }
    
    func testGenerateBatchSize2Float16() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        GrAI.Precision.float16 = true
        runGenerateBatchSize(
            addUnitOffset: true, hiddenActivation: GELUApprox.str
        )
    }
    
    func testGenerateSlidingWindow1Float()
    {
        runGenerateSlidingWindow(addUnitOffset: false, hiddenActivation: SiLU.str)
    }
    
    func testGenerateSlidingWindow2Float()
    {
        runGenerateSlidingWindow(
            addUnitOffset: true, hiddenActivation: GELUApprox.str
        )
    }
    
    func testGenerateSlidingWindow1Float16() throws
    {
        GrAI.Precision.float16 = true
        runGenerateSlidingWindow(addUnitOffset: false, hiddenActivation: SiLU.str)
    }
    
    func testGenerateSlidingWindow2Float16() throws
    {
        GrAI.Precision.float16 = true
        runGenerateSlidingWindow(
            addUnitOffset: true, hiddenActivation: GELUApprox.str
        )
    }
    
    func testGenerateSlidingWindowBatchSize1Float()
    {
        runGenerateSlidingWindowBatchSize(
            addUnitOffset: false, hiddenActivation: SiLU.str
        )
    }
    
    func testGenerateSlidingWindowBatchSize2Float()
    {
        runGenerateSlidingWindowBatchSize(
            addUnitOffset: true, hiddenActivation: GELUApprox.str
        )
    }
    
    func testGenerateSlidingWindowBatchSize1Float16() throws
    {
        GrAI.Precision.float16 = true
        runGenerateSlidingWindowBatchSize(
            addUnitOffset: false, hiddenActivation: SiLU.str
        )
    }
    
    func testGenerateSlidingWindowBatchSize2Float16() throws
    {
        GrAI.Precision.float16 = true
        runGenerateSlidingWindowBatchSize(
            addUnitOffset: true, hiddenActivation: GELUApprox.str
        )
    }
}
