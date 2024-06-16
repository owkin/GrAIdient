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
                nbNeurons: 8,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! RoPESeq(
                layerPrev: layer,
                seqPositions: [Int](1...sequence),
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
                nbNeurons: 8,
                activation: nil,
                biases: false,
                params: params
            )
            layer = try! RoPESeq(
                layerPrev: layer,
                seqPositions: [Int](1...sequence),
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
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class NLPTransformTests: NLPFlowTests
{
    ///
    /// Run Transform tests.
    ///
    /// The goal is to compare the losses computed in the CPU execution
    /// after transforming the model and do the same in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: TransformTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    transforms: [self.copy, self.copyInPlace],
                    setData: self.setData,
                    setLoss: self.setLoss,
                    getLoss: self.getLoss)
                {
                    (diffCPU: Double, diffGPU: Double) in
                    if diffCPU > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                    if diffGPU > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
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
}
