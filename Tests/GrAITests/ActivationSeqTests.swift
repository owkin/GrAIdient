//
// ActivationSeqTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 08/03/2023.
//

import XCTest
import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class ActivationSeqGradTests: Input2DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(model: String, activation: String?)
        -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "ActivationSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(model: model, activation: activation, context: context)
        }
        return trainer
    }
    
    private func _buildModel(
        model: String,
        activation: String?,
        context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        var layerSeq: LayerSeq = try! FullyConnectedPatch(
            layerPrev: layer, patch: width / 3, nbNeurons: 5,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 5,
                activation: activation, biases: true,
                params: params
            )
            
        case "Activation":
            layerSeq = ActivationSeq(
                layerPrev: layerSeq,
                activation: activation!,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testFLNoActivationCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: nil
        )
        run(trainer)
    }
    
    func testFLNoActivationGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: nil
        )
        run(trainer)
    }
    
    func testFLReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: ReLU.str
        )
        run(trainer)
    }
    
    func testFLReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: ReLU.str
        )
        run(trainer)
    }
    
    func testFLLeakyReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: LeakyReLU.str
        )
        run(trainer)
    }
    
    func testFLLeakyReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: LeakyReLU.str
        )
        run(trainer)
    }
    
    func testFLSoftReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: SoftReLU.str
        )
        run(trainer)
    }
    
    func testFLSoftReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: SoftReLU.str
        )
        run(trainer)
    }
    
    func testFLSigmoidCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: Sigmoid.str
        )
        run(trainer)
    }
    
    func testFLSigmoidGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: Sigmoid.str
        )
        run(trainer)
    }
    
    func testFLSiLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: SiLU.str
        )
        run(trainer)
    }
    
    func testFLSiLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: SiLU.str
        )
        run(trainer)
    }
    
    func testFLGELUApproxCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: GELUApprox.str
        )
        run(trainer)
    }
    
    func testFLGELUApproxGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: GELUApprox.str
        )
        run(trainer)
    }
    
    func testFLGELUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: GELU.str
        )
        run(trainer)
    }
    
    func testFLGELUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: GELU.str
        )
        run(trainer)
    }
    
    func testReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: ReLU.str
        )
        run(trainer)
    }
    
    func testReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: ReLU.str
        )
        run(trainer)
    }
    
    func testLeakyReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: LeakyReLU.str
        )
        run(trainer)
    }
    
    func testLeakyReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: LeakyReLU.str
        )
        run(trainer)
    }
    
    func testSoftReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: SoftReLU.str
        )
        run(trainer)
    }
    
    func testSoftReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SoftReLU.str
        )
        run(trainer)
    }
    
    func testSigmoidCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: Sigmoid.str
        )
        run(trainer)
    }
    
    func testSigmoidGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: Sigmoid.str
        )
        run(trainer)
    }
    
    func testSiLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: SiLU.str
        )
        run(trainer)
    }
    
    func testSiLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SiLU.str
        )
        run(trainer)
    }
    
    func testGELUApproxCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: GELUApprox.str
        )
        run(trainer)
    }
    
    func testGELUApproxGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: GELUApprox.str
        )
        run(trainer)
    }
    
    func testGELUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: GELU.str
        )
        run(trainer)
    }
    
    func testGELUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: GELU.str
        )
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class ActivationSeqInferenceTests: Input2DMSE1DCase
{
    private func _buildTrainer(model: String, activation: String?)
        -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "ActivationSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, activation: activation, context: context)
        }
        return trainer
    }
    
    func buildModel(
        model: String,
        activation: String?,
        context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        var layerSeq: LayerSeq = try! FullyConnectedPatch(
            layerPrev: layer, patch: width / 3, nbNeurons: 5,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 5,
                activation: activation, biases: true,
                params: params
            )
            
        case "Activation":
            layerSeq = ActivationSeq(
                layerPrev: layerSeq,
                activation: activation!,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testFLNoActivation() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: nil
        )
        run(trainer)
    }
    
    func testFLReLU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: ReLU.str
        )
        run(trainer)
    }
    
    func testFLLeakyReLU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: LeakyReLU.str
        )
        run(trainer)
    }
    
    func testFLSoftReLU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: SoftReLU.str
        )
        run(trainer)
    }
    
    func testFLSigmoid() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: Sigmoid.str
        )
        run(trainer)
    }
    
    func testFLSiLU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: SiLU.str
        )
        run(trainer)
    }
    
    func testFLGELUApprox() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: GELUApprox.str
        )
        run(trainer)
    }
    
    func testFLGELU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: GELU.str
        )
        run(trainer)
    }
    
    func testReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: ReLU.str
        )
        run(trainer)
    }
    
    func testLeakyReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: LeakyReLU.str
        )
        run(trainer)
    }
    
    func testSoftReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SoftReLU.str
        )
        run(trainer)
    }
    
    func testSigmoid() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: Sigmoid.str
        )
        run(trainer)
    }
    
    func testSiLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SiLU.str
        )
        run(trainer)
    }
    
    func testGELUApprox() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: GELUApprox.str
        )
        run(trainer)
    }
    
    func testGELU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: GELU.str
        )
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class ActivationSeqFlowPrecisionTests: ActivationSeqInferenceTests
{
    private func _buildTrainer(model: String, activation: String?)
        -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
            name: "ActivationSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, activation: activation, context: context)
        }
        return trainer
    }
    
    override func testFLNoActivation() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: nil
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testFLReLU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: ReLU.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testFLLeakyReLU() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: LeakyReLU.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testFLSoftReLU() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: SoftReLU.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testFLSigmoid() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: Sigmoid.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testFLSiLU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: SiLU.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testFLGELUApprox() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: GELUApprox.str
        )
        run(trainer)
    }
    
    override func testFLGELU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: GELU.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: ReLU.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testLeakyReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: LeakyReLU.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSoftReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SoftReLU.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSigmoid() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: Sigmoid.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSiLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SiLU.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testGELUApprox() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Activation", activation: GELUApprox.str
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testGELU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: GELU.str
        )
        run(trainer, diffThreshold: 0.005)
    }
}
