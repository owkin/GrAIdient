//
// Layer2DDirtyTests.swift
// MAKitTests
//
// Created by Jean-François Reboud on 04/12/2022.
//

import MAKit
import MATestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DDirtyGradTests: Input2DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        
        optimizerParams.nbLoops = 2
        MAKit.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(model: String) -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "Layer2D",
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
        let params = MAKit.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        var firstLayer: Layer2D = layer
        var secondLayer: Layer2D
        
        switch model
        {
        case "Convolution":
            firstLayer = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = Convolution2D(
                layerPrev: firstLayer, size: 3, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            
        case "ConvolutionStride":
            firstLayer = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = Convolution2D(
                layerPrev: firstLayer, size: 3, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "BN":
            secondLayer = BN2D(
                layerPrev: layer, activation: SoftReLU.str, params: params
            )
            
        case "MaxPool":
            secondLayer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "AdaptiveAvgPool":
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: firstLayer, size: width, params: params
            )
            
        case "Activation":
            secondLayer = Activation2D(
                layerPrev: layer,
                activation: SoftReLU.str,
                params: params
            )
            
        case "IRDFT2RGB":
            secondLayer = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = IRDFT2RGB(layerPrev: secondLayer, params: params)
            
            secondLayer = Convolution2D(
                layerPrev: secondLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            
        case "DecorrelateRGB":
            secondLayer = DecorrelateRGB(
                layerPrev: layer,
                correlation: [
                    0.26, 0.26, 0.27,
                    0.09, 0.00, -0.09,
                    0.02, -0.05, 0.03
                ].map { $0 / 0.4619524 },
                params: params
            )
            
        case "LinearScale":
            secondLayer = LinearScale2D(
                layerPrev: layer,
                weight: 2.0,
                bias: 3.0,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = Sum2D(
            layersPrev: [firstLayer, secondLayer], params: params
        )
        
        var head: Layer1D = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testConvolutionCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Convolution")
        run(trainer)
    }
    
    func testConvolutionGPU() throws
    {
        let trainer = _buildTrainer(model: "Convolution")
        run(trainer)
    }
    
    func testConvolutionStrideCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "ConvolutionStride")
        run(trainer)
    }
    
    func testConvolutionStrideGPU() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride")
        run(trainer)
    }
    
    func testBN2DCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "BN")
        run(trainer)
    }
    
    func testBN2DGPU() throws
    {
        let trainer = _buildTrainer(model: "BN")
        run(trainer)
    }
    
    func testMaxPoolCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "MaxPool")
        run(trainer)
    }
    
    func testMaxPoolGPU() throws
    {
        let trainer = _buildTrainer(model: "MaxPool")
        run(trainer)
    }
    
    func testAdaptiveAvgPoolCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool")
        run(trainer)
    }
    
    func testAdaptiveAvgPoolGPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool")
        run(trainer)
    }
    
    func testActivationCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Activation")
        run(trainer)
    }
    
    func testActivationGPU() throws
    {
        let trainer = _buildTrainer(model: "Activation")
        run(trainer)
    }
    
    func testIRDFT2RGBCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "IRDFT2RGB")
        run(trainer)
    }
    
    func testIRDFT2RGBGPU() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB")
        run(trainer)
    }
    
    func testDecorrelateRGBCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "DecorrelateRGB")
        run(trainer)
    }
    
    func testDecorrelateRGBGPU() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB")
        run(trainer)
    }
    
    func testLinearScale2DCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "LinearScale")
        run(trainer)
    }
    
    func testLinearScale2DGPU() throws
    {
        let trainer = _buildTrainer(model: "LinearScale")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DDirtyFlowTests: Input2DMSE1DCase
{
    private func _buildTrainer(model: String) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "Layer2D",
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
        let params = MAKit.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        var firstLayer: Layer2D = layer
        var secondLayer: Layer2D
        
        switch model
        {
        case "Convolution":
            firstLayer = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = Convolution2D(
                layerPrev: firstLayer, size: 3, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            
        case "ConvolutionStride":
            firstLayer = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = Convolution2D(
                layerPrev: firstLayer, size: 3, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "BN":
            secondLayer = BN2D(
                layerPrev: layer, activation: SoftReLU.str, params: params
            )
            
        case "MaxPool":
            secondLayer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "AdaptiveAvgPool":
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: firstLayer, size: width, params: params
            )
            
        case "Activation":
            secondLayer = Activation2D(
                layerPrev: layer,
                activation: SoftReLU.str,
                params: params
            )
            
        case "IRDFT2RGB":
            secondLayer = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = IRDFT2RGB(layerPrev: secondLayer, params: params)
            
            secondLayer = Convolution2D(
                layerPrev: secondLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            
        case "DecorrelateRGB":
            secondLayer = DecorrelateRGB(
                layerPrev: layer,
                correlation: [
                    0.26, 0.26, 0.27,
                    0.09, 0.00, -0.09,
                    0.02, -0.05, 0.03
                ].map { $0 / 0.4619524 },
                params: params
            )
            
        case "LinearScale":
            secondLayer = LinearScale2D(
                layerPrev: layer,
                weight: 2.0,
                bias: 3.0,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = Sum2D(
            layersPrev: [firstLayer, secondLayer], params: params
        )
        
        var head: Layer1D = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testConvolution() throws
    {
        let trainer = _buildTrainer(model: "Convolution")
        run(trainer)
    }
    
    func testConvolutionStride() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride")
        run(trainer)
    }
    
    func testBN2D() throws
    {
        let trainer = _buildTrainer(model: "BN")
        run(trainer)
    }
    
    func testMaxPool() throws
    {
        let trainer = _buildTrainer(model: "MaxPool")
        run(trainer)
    }
    
    func testAdaptiveAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool")
        run(trainer)
    }
    
    func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation")
        run(trainer)
    }
    
    func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB")
        run(trainer)
    }
    
    func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB")
        run(trainer)
    }
    
    func testLinearScale2D() throws
    {
        let trainer = _buildTrainer(model: "LinearScale")
        run(trainer)
    }
}