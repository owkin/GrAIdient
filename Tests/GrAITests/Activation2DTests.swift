//
// Activation2DTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 15/10/2022.
//

import XCTest
import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Activation2DGradTests: Input2DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        
        optimizerParams.nbLoops = 2
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(model: String, activation: String?, bn: Bool)
        -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "Activation2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(
                model: model, activation: activation, bn: bn, context: context
            )
        }
        return trainer
    }
    
    private func _buildModel(
        model: String,
        activation: String?,
        bn: Bool,
        context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1,
            width: width,
            height: height,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        switch model
        {
        case "Convolution":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: activation, biases: true, bn: bn, params: params
            )
            
        case "Activation":
           layer = Activation2D(
                layerPrev: layer,
                activation: activation!,
                params: params
           )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testConvNoActivationNoBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: nil, bn: false
        )
        run(trainer)
    }
    
    func testConvNoActivationBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: nil, bn: true
        )
        run(trainer)
    }
    
    func testConvNoActivationNoBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: nil, bn: false
        )
        run(trainer)
    }
    
    func testConvNoActivationBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: nil, bn: true
        )
        run(trainer)
    }
    
    func testConvReLUNoBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: ReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvReLUBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: ReLU.str, bn: true
        )
        run(trainer)
    }
    
    func testConvReLUNoBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: ReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvReLUBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: ReLU.str, bn: true
        )
        run(trainer)
    }
    
    func testConvLeakyReLUNoBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: LeakyReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvLeakyReLUBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: LeakyReLU.str, bn: true
        )
        run(trainer)
    }
    
    func testConvLeakyReLUNoBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: LeakyReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvLeakyReLUBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: LeakyReLU.str, bn: true
        )
        run(trainer)
    }
    
    func testConvSoftReLUNoBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: SoftReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvSoftReLUBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: SoftReLU.str, bn: true
        )
        run(trainer)
    }
    
    func testConvSoftReLUNoBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SoftReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvSoftReLUBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SoftReLU.str, bn: true
        )
        run(trainer)
    }
    
    func testConvSigmoidNoBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: Sigmoid.str, bn: false
        )
        run(trainer)
    }
    
    func testConvSigmoidBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: Sigmoid.str, bn: true
        )
        run(trainer)
    }
    
    func testConvSigmoidNoBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: Sigmoid.str, bn: false
        )
        run(trainer)
    }
    
    func testConvSigmoidBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: Sigmoid.str, bn: true
        )
        run(trainer)
    }
    
    func testConvSiLUNoBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: SiLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvSiLUBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: SiLU.str, bn: true
        )
        run(trainer)
    }
    
    func testConvSiLUNoBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SiLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvSiLUBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SiLU.str, bn: true
        )
        run(trainer)
    }
    
    func testConvGELUApproxNoBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELUApprox.str, bn: false
        )
        run(trainer)
    }
    
    func testConvGELUApproxBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELUApprox.str, bn: true
        )
        run(trainer)
    }
    
    func testConvGELUApproxNoBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELUApprox.str, bn: false
        )
        run(trainer)
    }
    
    func testConvGELUApproxBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELUApprox.str, bn: true
        )
        run(trainer)
    }
    
    func testConvGELUNoBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvGELUBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELU.str, bn: true
        )
        run(trainer)
    }
    
    func testConvGELUNoBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvGELUBNGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELU.str, bn: true
        )
        run(trainer)
    }
    
    func testReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: ReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: ReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testLeakyReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: LeakyReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testLeakyReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: LeakyReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testSoftReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: SoftReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testSoftReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SoftReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testSigmoidCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: Sigmoid.str, bn: false
        )
        run(trainer)
    }
    
    func testSigmoidGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: Sigmoid.str, bn: false
        )
        run(trainer)
    }
    
    func testSiLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: SiLU.str, bn: false
        )
        run(trainer)
    }
    
    func testSiLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SiLU.str, bn: false
        )
        run(trainer)
    }
    
    func testGELUApproxCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: GELUApprox.str, bn: false
        )
        run(trainer)
    }
    
    func testGELUApproxGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: GELUApprox.str, bn: false
        )
        run(trainer)
    }
    
    func testGELUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: GELU.str, bn: false
        )
        run(trainer)
    }
    
    func testGELUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: GELU.str, bn: false
        )
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Activation2DInferenceTests: Input2DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        optimizerParams.nbLoops = 2
    }
    
    private func _buildTrainer(model: String, activation: String?, bn: Bool)
        -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "Activation2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(
                model: model, activation: activation, bn: bn, context: context
            )
        }
        return trainer
    }
    
    func buildModel(
        model: String,
        activation: String?,
        bn: Bool,
        context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1,
            width: width,
            height: height,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        switch model
        {
        case "Convolution":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: activation, biases: true, bn: bn, params: params
            )
            
        case "Activation":
           layer = Activation2D(
                layerPrev: layer,
                activation: activation!,
                params: params
           )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testConvNoActivationNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: nil, bn: false
        )
        run(trainer)
    }
    
    func testConvNoActivationBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: nil, bn: true
        )
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    func testConvReLUNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: ReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvReLUBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: ReLU.str, bn: true
        )
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    func testConvLeakyReLUNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: LeakyReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvLeakyReLUBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: LeakyReLU.str, bn: true
        )
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    func testConvSoftReLUNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SoftReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvSoftReLUBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SoftReLU.str, bn: true
        )
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    func testConvSigmoidNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: Sigmoid.str, bn: false
        )
        run(trainer)
    }
    
    func testConvSigmoidBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: Sigmoid.str, bn: true
        )
        run(trainer)
    }
    
    func testConvSiLUNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SiLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvSiLUBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: SiLU.str, bn: true
        )
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    func testConvGELUApproxNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELUApprox.str, bn: false
        )
        run(trainer)
    }
    
    func testConvGELUApproxBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELUApprox.str, bn: true
        )
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    func testConvGELUNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvGELUBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELU.str, bn: true
        )
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    func testReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: ReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testLeakyReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: LeakyReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testSoftReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SoftReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testSigmoid() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: Sigmoid.str, bn: false
        )
        run(trainer)
    }
    
    func testSiLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SiLU.str, bn: false
        )
        run(trainer)
    }
    
    func testGELUApprox() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: GELUApprox.str, bn: false
        )
        run(trainer)
    }
    
    func testGELU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: GELU.str, bn: false
        )
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class Activation2DFlowPrecisionTests: Activation2DInferenceTests
{
    override func setUp()
    {
        super.setUp()
        optimizerParams.nbLoops = 3
    }
    
    private func _buildTrainer(model: String, activation: String?, bn: Bool)
        -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
            name: "Activation2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(
                model: model, activation: activation, bn: bn, context: context
            )
        }
        return trainer
    }
    
    override func testConvNoActivationNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: nil, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvNoActivationBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: nil, bn: true
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvReLUNoBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: ReLU.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvReLUBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: ReLU.str, bn: true
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvLeakyReLUNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: LeakyReLU.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvLeakyReLUBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: LeakyReLU.str, bn: true
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvSoftReLUNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SoftReLU.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvSoftReLUBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SoftReLU.str, bn: true
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvSigmoidNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: Sigmoid.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvSigmoidBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: Sigmoid.str, bn: true
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvSiLUNoBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SiLU.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvSiLUBN() throws
    {
        let trainer = _buildTrainer(
            model: "Convolution", activation: SiLU.str, bn: true
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvGELUApproxNoBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELUApprox.str, bn: false
        )
        run(trainer)
    }
    
    override func testConvGELUApproxBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELUApprox.str, bn: true
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvGELUNoBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELU.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvGELUBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Convolution", activation: GELU.str, bn: true
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: ReLU.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testLeakyReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: LeakyReLU.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSoftReLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SoftReLU.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSigmoid() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: Sigmoid.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSiLU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SiLU.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testGELUApprox() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(
            model: "Activation", activation: GELUApprox.str, bn: false
        )
        run(trainer)
    }
    
    override func testGELU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: GELU.str, bn: false
        )
        run(trainer, diffThreshold: 0.005)
    }
}
