//
// Activation2DTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 15/10/2022.
//

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
        
        var head: Layer1D = FullyConnected(
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
