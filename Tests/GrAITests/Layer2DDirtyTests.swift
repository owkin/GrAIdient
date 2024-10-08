//
// Layer2DDirtyTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 04/12/2022.
//

import GrAIdient
import GrAITestsUtils

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
        GrAI.Loop.gradientChecking = true
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
        let params = GrAI.Model.Params(context: context)
        
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
            firstLayer = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = try! IRDFT2RGB(layerPrev: firstLayer, params: params)
            
            secondLayer = Convolution2D(
                layerPrev: secondLayer, size: 1, nbChannels: 6, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            
        case "DecorrelateRGB":
            secondLayer = try! DecorrelateRGB(
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
            
        case "Pad":
            secondLayer = Pad2D(
                layerPrev: layer,
                padDimension: 3, padValue: 0.5,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "Crop":
            secondLayer = try! Crop2D(
                layerPrev: layer,
                cropDimension: 3,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "ResizeBilinearPad1":
            secondLayer = try! ResizeBilinearPad(
                layerPrev: layer,
                scalesList: [0.8, 1.2], padValue: 0.5,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "ResizeBilinearPad2":
            secondLayer = try! ResizeBilinearPad(
                layerPrev: layer,
                minScale: 0.8, maxScale: 1.2, padValue: 0.5,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "Rotate1":
            secondLayer = try! Rotate2D(
                layerPrev: layer,
                anglesList: [20.0, 350.0], padValue: 0.5,
                params: params
            )
            
        case "Rotate2":
            secondLayer = try! Rotate2D(
                layerPrev: layer,
                minAngle: 20.0, maxAngle: 350.0, padValue: 0.5,
                params: params
            )
            
        case "ResizeBilinearCrop1":
            secondLayer = try! ResizeBilinearCrop(
                layerPrev: layer,
                scalesList: [0.6, 0.8],
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "ResizeBilinearCrop2":
            secondLayer = try! ResizeBilinearCrop(
                layerPrev: layer,
                minScale: 0.6, maxScale: 0.8,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "Deconvolution":
            secondLayer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "DeconvolutionStride":
            secondLayer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 3, stride: 2,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "InstanceNorm":
            secondLayer = InstanceNorm2D(
                layerPrev: layer, activation: SoftReLU.str, params: params
            )
            
        case "SelfCorrelate":
            secondLayer = SelfCorrelate2D(layerPrev: layer, params: params)
            
            secondLayer = Convolution2D(
                layerPrev: secondLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "Normalize1":
            secondLayer = Normalize12D(layerPrev: layer, params: params)
            
        case "Normalize12":
            secondLayer = Normalize122D(layerPrev: layer, params: params)
            
        case "FlipHorizontal1":
            secondLayer = FlipHorizontal2D(
                layerPrev: layer, probability: 1.0, params: params
            )
            
        case "FlipHorizontal2":
            secondLayer = FlipHorizontal2D(
                layerPrev: layer, probability: 0.0, params: params
            )
        
        case "FlipVertical1":
            secondLayer = FlipVertical2D(
                layerPrev: layer, probability: 1.0, params: params
            )
            
        case "FlipVertical2":
            secondLayer = FlipVertical2D(
                layerPrev: layer, probability: 0.0, params: params
            )
            
        case "LayerOutput":
            secondLayer = try! MSE2D(layerPrev: layer, params: params)
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = try! Sum2D(
            layersPrev: [firstLayer, secondLayer], params: params
        )
        
        var head: Layer1D = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testConvolutionCPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ConvolutionStride")
        run(trainer)
    }
    
    func testConvolutionStrideGPU() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride")
        run(trainer)
    }
    
    func testBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "BN")
        run(trainer)
    }
    
    func testBNGPU() throws
    {
        let trainer = _buildTrainer(model: "BN")
        run(trainer)
    }
    
    func testMaxPoolCPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "DecorrelateRGB")
        run(trainer)
    }
    
    func testDecorrelateRGBGPU() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB")
        run(trainer)
    }
    
    func testLinearScaleCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "LinearScale")
        run(trainer)
    }
    
    func testLinearScaleGPU() throws
    {
        let trainer = _buildTrainer(model: "LinearScale")
        run(trainer)
    }
    
    func testPadCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Pad")
        run(trainer)
    }
    
    func testPadGPU() throws
    {
        let trainer = _buildTrainer(model: "Pad")
        run(trainer)
    }
    
    func testCropCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Crop")
        run(trainer)
    }
    
    func testCropGPU() throws
    {
        let trainer = _buildTrainer(model: "Crop")
        run(trainer)
    }
    
    func testResizeBilinearPad1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearPad1")
        run(trainer)
    }
    
    func testResizeBilinearPad1GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1")
        run(trainer)
    }
    
    func testResizeBilinearPad2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearPad2")
        run(trainer)
    }
    
    func testResizeBilinearPad2GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2")
        run(trainer)
    }
    
    func testRotate1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Rotate1")
        run(trainer)
    }
    
    func testRotate1GPU() throws
    {
        let trainer = _buildTrainer(model: "Rotate1")
        run(trainer)
    }
    
    func testRotate2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Rotate2")
        run(trainer)
    }
    
    func testRotate2GPU() throws
    {
        let trainer = _buildTrainer(model: "Rotate2")
        run(trainer)
    }
    
    func testResizeBilinearCrop1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1")
        run(trainer)
    }
    
    func testResizeBilinearCrop1GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1")
        run(trainer)
    }
    
    func testResizeBilinearCrop2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2")
        run(trainer)
    }
    
    func testResizeBilinearCrop2GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2")
        run(trainer)
    }
    
    func testDeconvolutionCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Deconvolution")
        run(trainer)
    }
    
    func testDeconvolutionGPU() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution")
        run(trainer, diffThreshold: 0.001)
    }
    
    func testDeconvolutionStrideCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "DeconvolutionStride")
        run(trainer)
    }
    
    func testDeconvolutionStrideGPU() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride")
        run(trainer, diffThreshold: 0.001)
    }
    
    func testInstanceNormCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "InstanceNorm")
        run(trainer)
    }
    
    func testInstanceNormGPU() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm")
        run(trainer)
    }
    
    func testSelfCorrelateCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "SelfCorrelate")
        run(trainer)
    }
    
    func testSelfCorrelateGPU() throws
    {
        let trainer = _buildTrainer(model: "SelfCorrelate")
        run(trainer)
    }
    
    func testNormalize1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Normalize1")
        run(trainer)
    }
    
    func testNormalize1GPU() throws
    {
        let trainer = _buildTrainer(model: "Normalize1")
        run(trainer)
    }
    
    func testNormalize12CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Normalize12")
        run(trainer)
    }
    
    func testNormalize12GPU() throws
    {
        let trainer = _buildTrainer(model: "Normalize12")
        run(trainer)
    }
    
    func testFlipHorizontal1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "FlipHorizontal1")
        run(trainer)
    }
    
    func testFlipHorizontal1GPU() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal1")
        run(trainer)
    }
    
    func testFlipHorizontal2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "FlipHorizontal2")
        run(trainer)
    }
    
    func testFlipHorizontal2GPU() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal2")
        run(trainer)
    }
    
    func testFlipVertical1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "FlipVertical1")
        run(trainer)
    }
    
    func testFlipVertical1GPU() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical1")
        run(trainer)
    }
    
    func testFlipVertical2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "FlipVertical2")
        run(trainer)
    }
    
    func testFlipVertical2GPU() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical2")
        run(trainer)
    }
    
    func testLayerOutputCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "LayerOutput")
        run(trainer)
    }
    
    func testLayerOutputGPU() throws
    {
        let trainer = _buildTrainer(model: "LayerOutput")
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
        let params = GrAI.Model.Params(context: context)
        
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
                layerPrev: layer, activation: LeakyReLU.str, params: params
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
                activation: LeakyReLU.str,
                params: params
            )
            
        case "IRDFT2RGB":
            firstLayer = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = try! IRDFT2RGB(layerPrev: firstLayer, params: params)
            
            secondLayer = Convolution2D(
                layerPrev: secondLayer, size: 1, nbChannels: 6, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            
        case "DecorrelateRGB":
            secondLayer = try! DecorrelateRGB(
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
            
        case "Pad":
            secondLayer = Pad2D(
                layerPrev: layer,
                padDimension: 3, padValue: 0.5,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "Crop":
            secondLayer = try! Crop2D(
                layerPrev: layer,
                cropDimension: 3,
                offsetI: 2,
                offsetJ: 2,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "ResizeBilinearPad":
            secondLayer = try! ResizeBilinearPad(
                layerPrev: layer,
                scalesList: [0.8], padValue: 0.5,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "Rotate":
            secondLayer = try! Rotate2D(
                layerPrev: layer,
                anglesList: [20.0], padValue: 0.5,
                params: params
            )
            
        case "ResizeBilinearCrop":
            secondLayer = try! ResizeBilinearCrop(
                layerPrev: layer,
                scale: 1.2,
                offsetI: 1,
                offsetJ: 1,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "Deconvolution":
            secondLayer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "DeconvolutionStride":
            secondLayer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 3, stride: 2,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "Concat":
            let otherLayer: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = try! Concat2D(
                layersPrev: [firstLayer, otherLayer],
                params: params
            )
            secondLayer = Convolution2D(
                layerPrev: secondLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            
        case "Sum":
            let otherLayer: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = try! Sum2D(
                layersPrev: [firstLayer, otherLayer],
                params: params
            )
            
        case "Multiply":
            let otherLayer: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = try! Multiply2D(
                layersPrev: [firstLayer, otherLayer],
                params: params
            )
            
        case "InstanceNorm":
            secondLayer = InstanceNorm2D(
                layerPrev: layer, activation: LeakyReLU.str, params: params
            )
            
        case "AdaIN":
            let otherLayer: Layer = Constant1D(
                nbNeurons: 6, params: params
            )
            (otherLayer as! Constant1D).weightsCPU = [
                0.5, -0.5, 1.5, -2.0, 3.0, 1.0
            ]
            secondLayer = try! AdaIN(
                layersPrev: [firstLayer, otherLayer],
                params: params
            )
            
        case "VQ":
            secondLayer = VQ2D(layerPrev: layer, K: 5, params: params)
            (secondLayer as! VQ2D).beta = 0.25
            
        case "SelfCorrelate":
            secondLayer = SelfCorrelate2D(layerPrev: layer, params: params)
            
            secondLayer = Convolution2D(
                layerPrev: secondLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            secondLayer = AdaptiveAvgPool2D(
                layerPrev: secondLayer, size: width, params: params
            )
            
        case "Normalize1":
            secondLayer = Normalize12D(layerPrev: layer, params: params)
            
        case "Normalize12":
            secondLayer = Normalize122D(layerPrev: layer, params: params)
            
        case "FlipHorizontal1":
            secondLayer = FlipHorizontal2D(
                layerPrev: layer, probability: 1.0, params: params
            )
            
        case "FlipHorizontal2":
            secondLayer = FlipHorizontal2D(
                layerPrev: layer, probability: 0.0, params: params
            )
        
        case "FlipVertical1":
            secondLayer = FlipVertical2D(
                layerPrev: layer, probability: 1.0, params: params
            )
            
        case "FlipVertical2":
            secondLayer = FlipVertical2D(
                layerPrev: layer, probability: 0.0, params: params
            )
            
        case "LayerOutput":
            secondLayer = try! MSE2D(layerPrev: layer, params: params)
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = try! Sum2D(
            layersPrev: [firstLayer, secondLayer], params: params
        )
        
        var head: Layer1D = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
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
    
    func testBN() throws
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
    
    func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale")
        run(trainer)
    }
    
    func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad")
        run(trainer)
    }
    
    func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop")
        run(trainer)
    }
    
    func testResizeBilinearPad() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad")
        run(trainer)
    }
    
    func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate")
        run(trainer)
    }
    
    func testResizeBilinearCrop() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop")
        run(trainer)
    }
    
    func testDeconvolution() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution")
        run(trainer)
    }
    
    func testDeconvolutionStride() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride")
        run(trainer)
    }
    
    func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat")
        run(trainer)
    }
    
    func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum")
        run(trainer)
    }
    
    func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply")
        run(trainer)
    }
    
    func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm")
        run(trainer)
    }
    
    func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN")
        run(trainer)
    }
    
    func testVQ() throws
    {
        let trainer = _buildTrainer(model: "VQ")
        run(trainer)
    }
    
    func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ")
        run(trainer)
    }
    
    func testSelfCorrelate() throws
    {
        let trainer = _buildTrainer(model: "SelfCorrelate")
        run(trainer)
    }
    
    func testNormalize1() throws
    {
        let trainer = _buildTrainer(model: "Normalize1")
        run(trainer)
    }
    
    func testNormalize12() throws
    {
        let trainer = _buildTrainer(model: "Normalize12")
        run(trainer)
    }
    
    func testFlipHorizontal1() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal1")
        run(trainer)
    }
    
    func testFlipHorizontal2() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal2")
        run(trainer)
    }
    
    func testFlipVertical1() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical1")
        run(trainer)
    }
    
    func testFlipVertical2() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical2")
        run(trainer)
    }
    
    func testLayerOutput() throws
    {
        let trainer = _buildTrainer(model: "LayerOutput")
        run(trainer)
    }
}
