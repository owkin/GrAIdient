//
// Layer2DTests.swift
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
class Layer2DGradTests: Input2DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        
        optimizerParams.nbLoops = 2
        MAKit.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(model: String, bn: Bool) -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    private func _buildModel(model: String, bn: Bool, context: ModelContext)
    {
        let params = MAKit.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        let firstLayer = layer
        var head: Layer1D? = nil
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        switch model
        {
        case "Convolution1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            
        case "Convolution2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            
        case "ConvolutionStride1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            
        case "ConvolutionStride2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            
        case "BN":
            layer = BN2D(
                layerPrev: layer, activation: SoftReLU.str, params: params
            )
            
        case "MaxPool1":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
        case "MaxPool2":
            layer = MaxPool2D(
                layerPrev: layer, size: 4, stride: 4, params: params
            )
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
        case "MaxPool3":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 2, params: params
            )
            
        case "AvgPooling":
            head = AvgPool2D(layerPrev: layer, params: params)
            
        case "AdaptiveAvgPool1":
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 6, params: params
            )
            
        case "AdaptiveAvgPool2":
            layer = AdaptiveAvgPool2D(layerPrev: layer, size: 2, params: params)
            
        case "AdaptiveAvgPool3":
            layer = AdaptiveAvgPool2D(layerPrev: layer, size: 3, params: params)
            
        case "AdaptiveAvgPool4":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 9, params: params
            )
            
        case "AdaptiveAvgPool5":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 6, params: params
            )
            
        case "Sum":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            
            layer = Sum2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Activation":
            layer = Activation2D(
                layerPrev: layer,
                activation: SoftReLU.str,
                params: params
            )
            
        case "SelectNeurons":
            head = SelectNeurons2D(
                layerPrev: layer,
                targetI: 1, targetJ: 3,
                params: params
            )
            
        case "IRDFT2RGB":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 6, stride: 2,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            
            layer = IRDFT2RGB(layerPrev: layer, params: params)
            
        case "DecorrelateRGB":
            layer = DecorrelateRGB(
                layerPrev: layer,
                correlation: [
                    0.26, 0.26, 0.27,
                    0.09, 0.00, -0.09,
                    0.02, -0.05, 0.03
                ].map { $0 / 0.4619524 },
                params: params
            )
            
        case "LinearScale":
            layer = LinearScale2D(
                layerPrev: layer,
                weight: 2.0,
                bias: 3.0,
                params: params
            )
            
        case "Multiply":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            
            layer = Multiply2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Pad":
            layer = Pad2D(
                layerPrev: layer,
                padDimension: 3, padValue: 0.5,
                params: params
            )
            
        case "Jitter":
            layer = Jitter2D(
                layerPrev: layer,
                jitterDimension: 3,
                params: params
            )
            
        case "ResizeBilinear":
            layer = ResizeBilinear(
                layerPrev: layer,
                scalesList: [0.8, 1.2], padValue: 0.5,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "Rotate":
            layer = Rotate2D(
                layerPrev: layer,
                anglesList: [20.0, 350.0], padValue: 0.5,
                params: params
            )
            
        case "ResizeBilinearJitter":
            layer = ResizeBilinearJitter(
                layerPrev: layer,
                scalesList: [0.8, 1.2],
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        head = FullyConnected(
            layerPrev: head != nil ? head! : layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head!, params: params)
    }
    
    func testConvolution1BNCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1BNGPU() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1BNSampleGPU() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1NoBNCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution1NoBNGPU() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution1NoBNSampleGPU() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution2CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolution2GPU() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolution2SampleGPU() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1GPU() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1SampleGPU() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2GPU() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2SampleGPU() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testBNCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    func testBNGPU() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    func testMaxPool1CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    func testMaxPool1GPU() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    func testMaxPool2CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    func testMaxPool2GPU() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    func testMaxPool3CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    func testMaxPool3GPU() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    func testAvgPoolCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    func testAvgPoolGPU() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool1CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool1GPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool2CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool2GPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool3CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool3GPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool4CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool4GPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool5CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool5GPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    func testSumCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    func testSumGPU() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    func testActivationCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    func testActivationGPU() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    func testSelectNeuronsCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    func testSelectNeuronsGPU() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    func testIRDFT2RGBCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    func testIRDFT2RGBGPU() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    func testDecorrelateRGBCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    func testDecorrelateRGBGPU() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    func testLinearScaleCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    func testLinearScaleGPU() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    func testMultiplyCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    func testMultiplyGPU() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    func testPadCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    func testPadGPU() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    func testJitterCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Jitter", bn: false)
        run(trainer)
    }
    
    func testJitterGPU() throws
    {
        let trainer = _buildTrainer(model: "Jitter", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinear", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearGPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear", bn: false)
        run(trainer)
    }
    
    func testRotateCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    func testRotateGPU() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearJitterCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearJitter", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DFlowTests: Input2DMSE1DCase
{
    private func _buildTrainer(model: String, bn: Bool) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    func buildModel(model: String, bn: Bool, context: ModelContext)
    {
        let params = MAKit.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        let firstLayer = layer
        var head: Layer1D? = nil
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        switch model
        {
        case "Convolution1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: bn, params: params
            )
            
        case "Convolution2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: bn, params: params
            )
            
        case "ConvolutionStride1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: true, bn: bn, params: params
            )
            
        case "ConvolutionStride2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: true, bn: bn, params: params
            )
            
        case "BN":
            layer = BN2D(
                layerPrev: layer, activation: LeakyReLU.str, params: params
            )
            
        case "MaxPool1":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
        case "MaxPool2":
            layer = MaxPool2D(
                layerPrev: layer, size: 4, stride: 4, params: params
            )
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
        case "MaxPool3":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 2, params: params
            )
            
        case "AvgPooling":
            head = AvgPool2D(layerPrev: layer, params: params)
            
        case "AdaptiveAvgPool1":
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 6, params: params
            )
            
        case "AdaptiveAvgPool2":
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 2, params: params
            )
            
        case "AdaptiveAvgPool3":
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 3, params: params
            )
            
        case "AdaptiveAvgPool4":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 9, params: params
            )
            
        case "AdaptiveAvgPool5":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 6, params: params
            )
            
        case "Sum":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            
            layer = Sum2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Activation":
            layer = Activation2D(
                layerPrev: layer,
                activation: LeakyReLU.str,
                params: params
            )
            
        case "SelectNeurons":
            head = SelectNeurons2D(
                layerPrev: layer,
                targetI: 1, targetJ: 3,
                params: params
            )
            
        case "IRDFT2RGB":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 6, stride: 2,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            
            layer = IRDFT2RGB(layerPrev: layer, params: params)
            
        case "DecorrelateRGB":
            layer = DecorrelateRGB(
                layerPrev: layer,
                correlation: [
                    0.26, 0.26, 0.27,
                    0.09, 0.00, -0.09,
                    0.02, -0.05, 0.03
                ].map { $0 / 0.4619524 },
                params: params
            )
            
        case "LinearScale":
            layer = LinearScale2D(
                layerPrev: layer,
                weight: 2.0,
                bias: 3.0,
                params: params
            )
            
        case "Multiply":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            
            layer = Multiply2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Pad":
            layer = Pad2D(
                layerPrev: layer,
                padDimension: 3, padValue: 0.5,
                params: params
            )
            
        case "Jitter":
            layer = Jitter2D(
                layerPrev: layer,
                jitterDimension: 3,
                offsetI: 2,
                offsetJ: 2,
                params: params
            )
            
        case "ResizeBilinear1":
            layer = ResizeBilinear(
                layerPrev: layer,
                scalesList: [0.8], padValue: 0.5,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "ResizeBilinear2":
            layer = ResizeBilinear(
                layerPrev: layer,
                scalesList: [1.2], padValue: 0.5,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "Rotate":
            layer = Rotate2D(
                layerPrev: layer,
                anglesList: [20.0], padValue: 0.5,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        head = FullyConnected(
            layerPrev: head != nil ? head! : layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head!, params: params)
    }
    
    func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    func testJitter() throws
    {
        let trainer = _buildTrainer(model: "Jitter", bn: false)
        run(trainer)
    }
    
    func testResizeBilinear1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinear2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear2", bn: false)
        run(trainer)
    }
    
    func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DFlowResetTests: Layer2DFlowTests
{
    override func setUp()
    {
        super.setUp()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer(model: String, bn: Bool) -> FlowResetTrainer
    {
        let trainer = FlowResetTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    override func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    override func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    override func testJitter() throws
    {
        let trainer = _buildTrainer(model: "Jitter", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinear1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinear2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DFlowReverseTests: Layer2DFlowTests
{
    override func setUp()
    {
        super.setUp()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer(model: String, bn: Bool) -> FlowReverseTrainer
    {
        let trainer = FlowReverseTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    override func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.00001)
    }
    
    override func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.00001)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, nbRetry: 5, diffThreshold: 0.00001)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    override func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    override func testJitter() throws
    {
        let trainer = _buildTrainer(model: "Jitter", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinear1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinear2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Layer2DInferenceTests: Layer2DFlowTests
{
    override func setUp()
    {
        super.setUp()
        optimizerParams.nbLoops = 2
    }
    
    private func _buildTrainer(model: String, bn: Bool) -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    override func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    override func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    override func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    override func testJitter() throws
    {
        let trainer = _buildTrainer(model: "Jitter", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinear1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinear2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Layer2DLoadTests: Layer2DFlowTests
{
    private func _buildTrainer(model: String, bn: Bool) -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    override func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    override func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    override func testJitter() throws
    {
        let trainer = _buildTrainer(model: "Jitter", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinear1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinear2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Layer2DTransformTests: Layer2DFlowTests
{
    private func _buildTrainer(model: String, bn: Bool) -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    override func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    override func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    override func testJitter() throws
    {
        let trainer = _buildTrainer(model: "Jitter", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinear1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinear2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DGradTests: FTFrequences2DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        
        optimizerParams.nbLoops = 2
        MAKit.Loop.gradientChecking = true
    }
    
    private func _buildTrainer() -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(context: context)
        }
        return trainer
    }
    
    private func _buildModel(context: ModelContext)
    {
        let params = MAKit.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        let frequences: Layer2D = FTFrequences2D(
            nbChannels: 6, dimension: width, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Multiply2D(
            layersPrev: [layer, frequences], params: params
        )
        
        var head: Layer1D = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testEvenCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testEvenGPU() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testOddCPU() throws
    {
        height = 7
        width = 7
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testOddGPU() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DFlowTests: FTFrequences2DMSE1DCase
{
    private func _buildTrainer() -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    func buildModel(context: ModelContext)
    {
        let params = MAKit.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        let frequences: Layer2D = FTFrequences2D(
            nbChannels: 6, dimension: width, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Multiply2D(
            layersPrev: [layer, frequences], params: params
        )
        
        var head: Layer1D = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DFlowResetTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> FlowResetTrainer
    {
        let trainer = FlowResetTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DFlowReverseTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> FlowReverseTrainer
    {
        let trainer = FlowReverseTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DFlowInferenceTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DLoadTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DTransformTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}
