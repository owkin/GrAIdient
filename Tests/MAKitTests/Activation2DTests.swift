//
// Activation2DTests.swift
// MAKitTests
//
// Created by Jean-François Reboud on 15/10/2022.
//

import MAKit
import MATestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Activation2DGradTests: Input2DMSE1DCase
{
    override func setUpWithError() throws
    {
        try! super.setUpWithError()
        
        optimizerParams.nbLoops = 1
        MAKit.Loop.gradientChecking = true
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
        let params = MAKit.Model.Params(context: context)
        
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
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: nil, bn: false
        )
        run(trainer)
    }
    
    func testConvNoActivationBNCPU() throws
    {
        MAKit.Opti.CPU = true
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
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: ReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvReLUBNCPU() throws
    {
        MAKit.Opti.CPU = true
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
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: LeakyReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvLeakyReLUBNCPU() throws
    {
        MAKit.Opti.CPU = true
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
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Convolution", activation: SoftReLU.str, bn: false
        )
        run(trainer)
    }
    
    func testConvSoftReLUBNCPU() throws
    {
        MAKit.Opti.CPU = true
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
    
    func testReLUCPU() throws
    {
        MAKit.Opti.CPU = true
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
        MAKit.Opti.CPU = true
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
        MAKit.Opti.CPU = true
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
}