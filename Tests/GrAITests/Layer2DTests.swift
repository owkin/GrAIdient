//
// Layer2DTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 15/10/2022.
//

import XCTest
import Foundation
import GrAIdient
import GrAITestsUtils

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
        GrAI.Loop.gradientChecking = true
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
        let params = GrAI.Model.Params(context: context)
        
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
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Convolution2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
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
            layer = try! Sum2D(
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
            layer = try! IRDFT2RGB(layerPrev: layer, params: params)
            
        case "DecorrelateRGB":
            layer = try! DecorrelateRGB(
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
            layer = try! Multiply2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Pad":
            layer = Pad2D(
                layerPrev: layer,
                padDimension: 3, padValue: 0.5,
                params: params
            )
            
        case "Crop":
            layer = try! Crop2D(
                layerPrev: layer,
                cropDimension: 3,
                params: params
            )
            
        case "ResizeBilinearPad1":
            layer = try! ResizeBilinearPad(
                layerPrev: layer,
                scalesList: [0.8, 1.2], padValue: 0.5,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "ResizeBilinearPad2":
            layer = try! ResizeBilinearPad(
                layerPrev: layer,
                minScale: 0.8, maxScale: 1.2, padValue: 0.5,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "Rotate1":
            layer = try! Rotate2D(
                layerPrev: layer,
                anglesList: [20.0, 350.0], padValue: 0.5,
                params: params
            )
            
        case "Rotate2":
            layer = try! Rotate2D(
                layerPrev: layer,
                minAngle: 20.0, maxAngle: 350.0, padValue: 0.5,
                params: params
            )
            
        case "ResizeBilinearCrop1":
            layer = try! ResizeBilinearCrop(
                layerPrev: layer,
                scalesList: [0.6, 0.8],
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "ResizeBilinearCrop2":
            layer = try! ResizeBilinearCrop(
                layerPrev: layer,
                scalesList: [0.8, 1.2],
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "ResizeBilinearCrop3":
            layer = try! ResizeBilinearCrop(
                layerPrev: layer,
                minScale: 0.8,
                maxScale: 1.2,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "Deconvolution1":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 4, params: params
            )
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Deconvolution2":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 4, params: params
            )
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "DeconvolutionStride1":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 4, params: params
            )
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "DeconvolutionStride2":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 4, params: params
            )
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Concat":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 9, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            layer = try! Concat2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "InstanceNorm":
            layer = InstanceNorm2D(
                layerPrev: layer, activation: SoftReLU.str, params: params
            )
            
        case "AdaIN":
            let otherLayer: Layer = Constant1D(
                nbNeurons: 6, params: params
            )
            (otherLayer as! Constant1D).weightsCPU = [
                0.5, -0.5, 1.5, -2.0, 3.0, 1.0
            ]
            layer = try! AdaIN(
                layersPrev: [layer, otherLayer], params: params
            )
            
        case "Constant":
            var otherLayer: Layer2D = Constant2D(
                nbChannels: 5, height: height, width: width, params: params
            )
            (otherLayer as! Constant2D).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            otherLayer = Convolution2D(
                layerPrev: otherLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            layer = try! Sum2D(
                layersPrev: [layer, otherLayer], params: params
            )
            
        case "SelfCorrelate":
            layer = SelfCorrelate2D(layerPrev: layer, params: params)
            
        case "Normalize1":
            layer = Normalize12D(layerPrev: layer, params: params)
            
        case "Normalize12":
            layer = Normalize122D(layerPrev: layer, params: params)
            
        case "FlipHorizontal1":
            layer = FlipHorizontal2D(
                layerPrev: layer, probability: 1.0, params: params
            )
            
        case "FlipHorizontal2":
            layer = FlipHorizontal2D(
                layerPrev: layer, probability: 0.0, params: params
            )
        
        case "FlipVertical1":
            layer = FlipVertical2D(
                layerPrev: layer, probability: 1.0, params: params
            )
            
        case "FlipVertical2":
            layer = FlipVertical2D(
                layerPrev: layer, probability: 0.0, params: params
            )
            
        case "LayerOutput":
            layer = try! MSE2D(layerPrev: layer, params: params)
            
        default:
            fatalError("Unreachable.")
        }
        
        if head == nil
        {
            head = AvgPool2D(layerPrev: layer, params: params)
        }
        
        head = try! FullyConnected(
            layerPrev: head!, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head!, params: params)
    }
    
    func testConvolution1BNCPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1NoBNCPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution2CPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1CPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2CPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    func testBNGPU() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testMaxPool1CPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    func testPadGPU() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    func testCropCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    func testCropGPU() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearPad1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearPad1GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearPad2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearPad2GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    func testRotate1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Rotate1", bn: false)
        run(trainer)
    }
    
    func testRotate1GPU() throws
    {
        let trainer = _buildTrainer(model: "Rotate1", bn: false)
        run(trainer)
    }
    
    func testRotate2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Rotate2", bn: false)
        run(trainer)
    }
    
    func testRotate2GPU() throws
    {
        let trainer = _buildTrainer(model: "Rotate2", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop1GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop2GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop3CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearCrop3", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop3GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop3", bn: false)
        run(trainer)
    }
    
    func testDeconvolution1BNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer)
    }
    
    func testDeconvolution1BNGPU() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testDeconvolution1BNSampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testDeconvolution1NoBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    func testDeconvolution1NoBNGPU() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testDeconvolution1NoBNSampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testDeconvolution2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    func testDeconvolution2GPU() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testDeconvolution2SampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testDeconvolutionStride1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testDeconvolutionStride1GPU() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testDeconvolutionStride1SampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testDeconvolutionStride2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testDeconvolutionStride2GPU() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testDeconvolutionStride2SampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.001)
    }
    
    func testConcatCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    func testConcatGPU() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    func testInstanceNormCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    func testInstanceNormGPU() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer, nbRetry: 5)
    }
    
    func testAdaINCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    func testAdaINGPU() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    func testConstantCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    func testConstantGPU() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    func testSelfCorrelateCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "SelfCorrelate", bn: false)
        run(trainer)
    }
    
    func testSelfCorrelateGPU() throws
    {
        let trainer = _buildTrainer(model: "SelfCorrelate", bn: false)
        run(trainer)
    }
    
    func testNormalize1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Normalize1", bn: false)
        run(trainer)
    }
    
    func testNormalize1GPU() throws
    {
        let trainer = _buildTrainer(model: "Normalize1", bn: false)
        run(trainer)
    }
    
    func testNormalize12CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Normalize12", bn: false)
        run(trainer)
    }
    
    func testNormalize12GPU() throws
    {
        let trainer = _buildTrainer(model: "Normalize12", bn: false)
        run(trainer)
    }
    
    func testFlipHorizontal1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "FlipHorizontal1", bn: false)
        run(trainer)
    }
    
    func testFlipHorizontal1GPU() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal1", bn: false)
        run(trainer)
    }
    
    func testFlipHorizontal2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "FlipHorizontal2", bn: false)
        run(trainer)
    }
    
    func testFlipHorizontal2GPU() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal2", bn: false)
        run(trainer)
    }
    
    func testFlipVertical1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "FlipVertical1", bn: false)
        run(trainer)
    }
    
    func testFlipVertical1GPU() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical1", bn: false)
        run(trainer)
    }
    
    func testFlipVertical2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "FlipVertical2", bn: false)
        run(trainer)
    }
    
    func testFlipVertical2GPU() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical2", bn: false)
        run(trainer)
    }
    
    func testLayerOutputCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "LayerOutput", bn: false)
        run(trainer)
    }
    
    func testLayerOutputGPU() throws
    {
        let trainer = _buildTrainer(model: "LayerOutput", bn: false)
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
        let params = GrAI.Model.Params(context: context)
        
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
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Convolution2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
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
            layer = try! Sum2D(
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
            layer = try! IRDFT2RGB(layerPrev: layer, params: params)
            
        case "DecorrelateRGB":
            layer = try! DecorrelateRGB(
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
            layer = try! Multiply2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Pad":
            layer = Pad2D(
                layerPrev: layer,
                padDimension: 3, padValue: 0.5,
                params: params
            )
            
        case "Crop":
            layer = try! Crop2D(
                layerPrev: layer,
                cropDimension: 3,
                offsetI: 2,
                offsetJ: 2,
                params: params
            )
            
        case "ResizeBilinearPad1":
            layer = try! ResizeBilinearPad(
                layerPrev: layer,
                scalesList: [0.8], padValue: 0.5,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "ResizeBilinearPad2":
            layer = try! ResizeBilinearPad(
                layerPrev: layer,
                scalesList: [1.2], padValue: 0.5,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "Rotate":
            layer = try! Rotate2D(
                layerPrev: layer,
                anglesList: [20.0], padValue: 0.5,
                params: params
            )
            
        case "ResizeBilinearCrop1":
            layer = try! ResizeBilinearCrop(
                layerPrev: layer,
                scale: 0.8,
                offsetI: 0,
                offsetJ: 0,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "ResizeBilinearCrop2":
            layer = try! ResizeBilinearCrop(
                layerPrev: layer,
                scale: 1.2,
                offsetI: 1,
                offsetJ: 1,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "Deconvolution1":
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Deconvolution2":
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "DeconvolutionStride1":
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "DeconvolutionStride2":
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Concat":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 9, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            layer = try! Concat2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "InstanceNorm":
            layer = InstanceNorm2D(
                layerPrev: layer, activation: LeakyReLU.str, params: params
            )
            
        case "AdaIN":
            let otherLayer: Layer = Constant1D(
                nbNeurons: 6, params: params
            )
            (otherLayer as! Constant1D).weightsCPU = [
                0.5, -0.5, 1.5, -2.0, 3.0, 1.0
            ]
            layer = try! AdaIN(
                layersPrev: [layer, otherLayer], params: params
            )
            
        case "Constant":
            var otherLayer: Layer2D = Constant2D(
                nbChannels: 5, height: height, width: width, params: params
            )
            (otherLayer as! Constant2D).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            otherLayer = Convolution2D(
                layerPrev: otherLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            layer = try! Sum2D(
                layersPrev: [layer, otherLayer], params: params
            )
            
        case "VQ":
            layer = VQ2D(layerPrev: layer, K: 5, params: params)
            (layer as! VQ2D).beta = 0.25
            
        case "ResizeBilinear1":
            layer = try! ResizeBilinear(
                layerPrev: layer,
                dimension: Int(round(0.8 * Double(height))),
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "ResizeBilinear2":
            layer = try! ResizeBilinear(
                layerPrev: layer,
                dimension: Int(round(1.2 * Double(height))),
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "SelfCorrelate":
            layer = SelfCorrelate2D(layerPrev: layer, params: params)
            
        case "Normalize1":
            layer = Normalize12D(layerPrev: layer, params: params)
            
        case "Normalize12":
            layer = Normalize122D(layerPrev: layer, params: params)
            
        case "FlipHorizontal1":
            layer = FlipHorizontal2D(
                layerPrev: layer, probability: 1.0, params: params
            )
            
        case "FlipHorizontal2":
            layer = FlipHorizontal2D(
                layerPrev: layer, probability: 0.0, params: params
            )
        
        case "FlipVertical1":
            layer = FlipVertical2D(
                layerPrev: layer, probability: 1.0, params: params
            )
            
        case "FlipVertical2":
            layer = FlipVertical2D(
                layerPrev: layer, probability: 0.0, params: params
            )
            
        case "LayerOutput":
            layer = try! MSE2D(layerPrev: layer, params: params)
            
        default:
            fatalError("Unreachable.")
        }
        
        if head == nil
        {
            head = AvgPool2D(layerPrev: layer, params: params)
        }
        
        head = try! FullyConnected(
            layerPrev: head!, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head!, params: params)
    }
    
    func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testConvolution1BNSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution1NoBNSample() throws
    {
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, diffThreshold: 0.0001)
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
    
    func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    func testVQ() throws
    {
        let trainer = _buildTrainer(model: "VQ", bn: false)
        run(trainer)
    }
    
    func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ", bn: false)
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
    
    func testSelfCorrelate() throws
    {
        let trainer = _buildTrainer(model: "SelfCorrelate", bn: false)
        run(trainer)
    }
    
    func testNormalize1() throws
    {
        let trainer = _buildTrainer(model: "Normalize1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testNormalize12() throws
    {
        let trainer = _buildTrainer(model: "Normalize12", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testFlipHorizontal1() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal1", bn: false)
        run(trainer)
    }
    
    func testFlipHorizontal2() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal2", bn: false)
        run(trainer)
    }
    
    func testFlipVertical1() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical1", bn: false)
        run(trainer)
    }
    
    func testFlipVertical2() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical2", bn: false)
        run(trainer)
    }
    
    func testLayerOutput() throws
    {
        let trainer = _buildTrainer(model: "LayerOutput", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class Layer2DFlowPrecisionTests: Layer2DFlowTests
{
    private func _buildTrainer(model: String, bn: Bool) -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
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
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolution1BNSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testPad() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testResizeBilinearPad1() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolutionStride1() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolutionStride2() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConstant() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer(model: "VQ", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testResizeBilinear1() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "ResizeBilinear1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testResizeBilinear2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinear2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSelfCorrelate() throws
    {
        let trainer = _buildTrainer(model: "SelfCorrelate", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testNormalize1() throws
    {
        let trainer = _buildTrainer(model: "Normalize1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testNormalize12() throws
    {
        let trainer = _buildTrainer(model: "Normalize12", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testFlipHorizontal1() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testFlipHorizontal2() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testFlipVertical1() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testFlipVertical2() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer(model: "LayerOutput", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2D16FlowTests: Input2DMSE1DCase
{
    /// Systematic call before test begins.
    override func setUp()
    {
        height = 8
        width = 8
        batchSize = 5
        
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 3
    }
    
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        var head: Layer1D? = nil
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 32, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        switch model
        {
        case "Convolution1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 32, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Convolution2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 32, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 32, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 32, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Deconvolution":
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 16, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        if head == nil
        {
            head = AvgPool2D(layerPrev: layer, params: params)
        }
        
        head = try! FullyConnected(
            layerPrev: head!, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head!, params: params)
    }
    
    func testConvolution1() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testDeconvolution() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class Layer2D16FlowPrecisionTests: Layer2D16FlowTests
{
    private func _buildTrainer(model: String, bn: Bool) -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
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
    
    override func testConvolution1() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolution2() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolutionStride1() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testConvolutionStride2() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testDeconvolution() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer(model: "Deconvolution", bn: false)
        run(trainer, diffThreshold: 0.005)
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
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConvolution1BNSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, diffThreshold: 0.0001)
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
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer(model: "VQ", bn: false)
        run(trainer)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ", bn: false)
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
    
    override func testSelfCorrelate() throws
    {
        let trainer = _buildTrainer(model: "SelfCorrelate", bn: false)
        run(trainer)
    }
    
    override func testNormalize1() throws
    {
        let trainer = _buildTrainer(model: "Normalize1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testNormalize12() throws
    {
        let trainer = _buildTrainer(model: "Normalize12", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testFlipHorizontal1() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal1", bn: false)
        run(trainer)
    }
    
    override func testFlipHorizontal2() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal2", bn: false)
        run(trainer)
    }
    
    override func testFlipVertical1() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical1", bn: false)
        run(trainer)
    }
    
    override func testFlipVertical2() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical2", bn: false)
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer(model: "LayerOutput", bn: false)
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
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConvolution1BNSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, diffThreshold: 0.0001)
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
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer(model: "VQ", bn: false)
        run(trainer)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ", bn: false)
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
    
    override func testSelfCorrelate() throws
    {
        let trainer = _buildTrainer(model: "SelfCorrelate", bn: false)
        run(trainer)
    }
    
    override func testNormalize1() throws
    {
        let trainer = _buildTrainer(model: "Normalize1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testNormalize12() throws
    {
        let trainer = _buildTrainer(model: "Normalize12", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testFlipHorizontal1() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal1", bn: false)
        run(trainer)
    }
    
    override func testFlipHorizontal2() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal2", bn: false)
        run(trainer)
    }
    
    override func testFlipVertical1() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical1", bn: false)
        run(trainer)
    }
    
    override func testFlipVertical2() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical2", bn: false)
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer(model: "LayerOutput", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DFlowAccumulateTests: Input2DMSE1DCase
{
    private func _buildTrainer(model: String, bn: Bool) -> FlowTrainer
    {
        let trainer = FlowAccumulateTrainer(
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
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
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Convolution2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "BN":
            layer = BN2D(
                layerPrev: layer, activation: LeakyReLU.str, params: params
            )
            
        case "Deconvolution1":
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Deconvolution2":
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "DeconvolutionStride1":
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "DeconvolutionStride2":
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "InstanceNorm":
            layer = InstanceNorm2D(
                layerPrev: layer, activation: LeakyReLU.str, params: params
            )
            
        case "Constant":
            var otherLayer: Layer2D = Constant2D(
                nbChannels: 5, height: height, width: width, params: params
            )
            (otherLayer as! Constant2D).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            otherLayer = Convolution2D(
                layerPrev: otherLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            layer = try! Sum2D(
                layersPrev: [layer, otherLayer], params: params
            )
            
        case "VQ":
            layer = VQ2D(layerPrev: layer, K: 5, params: params)
            (layer as! VQ2D).beta = 0.25
            
        default:
            fatalError("Unreachable.")
        }
        
        head = try! FullyConnected(
            layerPrev: head != nil ? head! : layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head!, params: params)
    }
    
    func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testConvolution1BNSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution1NoBNSample() throws
    {
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    func testVQ() throws
    {
        let trainer = _buildTrainer(model: "VQ", bn: false)
        run(trainer)
    }
    
    func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ", bn: false)
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
        /*let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)*/
    }
    
    override func testConvolution1BNSample() throws
    {
        /*GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)*/
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        /*let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)*/
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
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer(model: "VQ", bn: false)
        run(trainer)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ", bn: false)
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
    
    override func testSelfCorrelate() throws
    {
        let trainer = _buildTrainer(model: "SelfCorrelate", bn: false)
        run(trainer)
    }
    
    override func testNormalize1() throws
    {
        let trainer = _buildTrainer(model: "Normalize1", bn: false)
        run(trainer)
    }
    
    override func testNormalize12() throws
    {
        let trainer = _buildTrainer(model: "Normalize12", bn: false)
        run(trainer)
    }
    
    override func testFlipHorizontal1() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal1", bn: false)
        run(trainer)
    }
    
    override func testFlipHorizontal2() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal2", bn: false)
        run(trainer)
    }
    
    override func testFlipVertical1() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical1", bn: false)
        run(trainer)
    }
    
    override func testFlipVertical2() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical2", bn: false)
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer(model: "LayerOutput", bn: false)
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer(model: "VQ", bn: false)
        run(trainer)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ", bn: false)
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
    
    override func testSelfCorrelate() throws
    {
        let trainer = _buildTrainer(model: "SelfCorrelate", bn: false)
        run(trainer)
    }
    
    override func testNormalize1() throws
    {
        let trainer = _buildTrainer(model: "Normalize1", bn: false)
        run(trainer)
    }
    
    override func testNormalize12() throws
    {
        let trainer = _buildTrainer(model: "Normalize12", bn: false)
        run(trainer)
    }
    
    override func testFlipHorizontal1() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal1", bn: false)
        run(trainer)
    }
    
    override func testFlipHorizontal2() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal2", bn: false)
        run(trainer)
    }
    
    override func testFlipVertical1() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical1", bn: false)
        run(trainer)
    }
    
    override func testFlipVertical2() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical2", bn: false)
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer(model: "LayerOutput", bn: false)
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer(model: "VQ", bn: false)
        run(trainer)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ", bn: false)
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
    
    override func testSelfCorrelate() throws
    {
        let trainer = _buildTrainer(model: "SelfCorrelate", bn: false)
        run(trainer)
    }
    
    override func testNormalize1() throws
    {
        let trainer = _buildTrainer(model: "Normalize1", bn: false)
        run(trainer)
    }
    
    override func testNormalize12() throws
    {
        let trainer = _buildTrainer(model: "Normalize12", bn: false)
        run(trainer)
    }
    
    override func testFlipHorizontal1() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal1", bn: false)
        run(trainer)
    }
    
    override func testFlipHorizontal2() throws
    {
        let trainer = _buildTrainer(model: "FlipHorizontal2", bn: false)
        run(trainer)
    }
    
    override func testFlipVertical1() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical1", bn: false)
        run(trainer)
    }
    
    override func testFlipVertical2() throws
    {
        let trainer = _buildTrainer(model: "FlipVertical2", bn: false)
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer(model: "LayerOutput", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class MSE2DGradTests: Input2DMSE2DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 1, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        _ = try! MSE2D(layerPrev: layer, params: params)
    }
    
    func testLossCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testLossGPU() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class MSE2DFlowTests: Input2DMSE2DCase
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 1, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        _ = try! MSE2D(layerPrev: layer, params: params)
    }
    
    func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class MSE2DFlowPrecisionTests: MSE2DFlowTests
{
    private func _buildTrainer() -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class MSE2DFlowResetTests: MSE2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class MSE2DFlowReverseTests: MSE2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class MSE2DFlowInferenceTests: MSE2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class MSE2DLoadTests: MSE2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class MSE2DTransformTests: MSE2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
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
        GrAI.Loop.gradientChecking = true
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        let frequences: Layer2D = try! FTFrequences2D(
            nbChannels: 6, dimension: width, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        layer = try! Multiply2D(
            layersPrev: [layer, frequences], params: params
        )
        
        var head: Layer1D = AvgPool2D(layerPrev: layer, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testEvenCPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        let frequences: Layer2D = try! FTFrequences2D(
            nbChannels: 6, dimension: width, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        layer = try! Multiply2D(
            layersPrev: [layer, frequences], params: params
        )
        
        var head: Layer1D = AvgPool2D(layerPrev: layer, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
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
class FTFrequences2DFlowPrecisionTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
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
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
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

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-5 and less.
// -----------------------------------------------------------------------------
class SimilarityBatchError2DGradTests: Input2DSimilarityBatchError2DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 5, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        layer = SelfCorrelate2D(layerPrev: layer, params: params)
        
        layer = Normalize122D(layerPrev: layer, params: params)
        
        _ = try! SimilarityBatchError2D(layerPrev: layer, params: params)
    }
    
    func testCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testGPU() throws
    {
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.0001)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-5 and less.
// -----------------------------------------------------------------------------
class SimilarityBatchError2DFlowTests: Input2DSimilarityBatchError2DCase
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 5, stride: 1,
            activation: ReLU.str, biases: true, bn: false, params: params
        )
        
        layer = SelfCorrelate2D(layerPrev: layer, params: params)
        
        layer = Normalize122D(layerPrev: layer, params: params)
        
        _ = try! SimilarityBatchError2D(layerPrev: layer, params: params)
    }
    
    func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.0001)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class SimilarityBatchError2DFlowPrecisionTests: SimilarityBatchError2DFlowTests
{
    private func _buildTrainer() -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
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
    
    override func test() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-5 and less.
// -----------------------------------------------------------------------------
class SimilarityBatchError2DFlowResetTests: SimilarityBatchError2DFlowTests
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer, nbRetry: 5, diffThreshold: 0.0001)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-5 and less.
// -----------------------------------------------------------------------------
class SimilarityBatchError2DFlowReverseTests: SimilarityBatchError2DFlowTests
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer, nbRetry: 5, diffThreshold: 0.0001)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class SimilarityBatchError2DFlowInferenceTests: SimilarityBatchError2DFlowTests
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class SimilarityBatchError2DLoadTests: SimilarityBatchError2DFlowTests
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class SimilarityBatchError2DTransformTests: SimilarityBatchError2DFlowTests
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class SimilarityError2DGradTests: Input2DSimilarityError2DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
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
        let params = GrAI.Model.Params(context: context)
        
        let layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        var layer1, layer2: Layer2D
        layer1 = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 5, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        layer1 = SelfCorrelate2D(layerPrev: layer1, params: params)
        layer1 = Normalize122D(layerPrev: layer1, params: params)
        
        layer2 = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 5, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        layer2 = SelfCorrelate2D(layerPrev: layer2, params: params)
        layer2 = Normalize122D(layerPrev: layer2, params: params)
        
        _ = try! SimilarityError2D(layersPrev: [layer1, layer2], params: params)
    }
    
    func testCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testGPU() throws
    {
        let trainer = _buildTrainer()
        run(trainer, nbRetry: 5)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class SimilarityError2DFlowTests: Input2DSimilarityError2DCase
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
        let params = GrAI.Model.Params(context: context)
        
        let layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        var layer1, layer2: Layer2D
        layer1 = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 5, stride: 1,
            activation: ReLU.str, biases: true, bn: false, params: params
        )
        layer1 = SelfCorrelate2D(layerPrev: layer1, params: params)
        layer1 = Normalize122D(layerPrev: layer1, params: params)
        
        layer2 = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 5, stride: 1,
            activation: ReLU.str, biases: true, bn: false, params: params
        )
        layer2 = SelfCorrelate2D(layerPrev: layer2, params: params)
        layer2 = Normalize122D(layerPrev: layer2, params: params)
        
        _ = try! SimilarityError2D(layersPrev: [layer1, layer2], params: params)
    }
    
    func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class SimilarityError2DFlowPrecisionTests: SimilarityError2DFlowTests
{
    private func _buildTrainer() -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class SimilarityError2DFlowResetTests: SimilarityError2DFlowTests
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class SimilarityError2DFlowReverseTests: SimilarityError2DFlowTests
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class SimilarityError2DFlowInferenceTests: SimilarityError2DFlowTests
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class SimilarityError2DLoadTests: SimilarityError2DFlowTests
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class SimilarityError2DTransformTests: SimilarityError2DFlowTests
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
    
    override func test() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class BCE2DGradTests: Input2DBCE2DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 1, stride: 1,
            activation: Sigmoid.str, biases: true, bn: false, params: params
        )
        
        _ = try! BCE2D(layerPrev: layer, params: params)
    }
    
    func testLossCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testLossGPU() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class BCE2DFlowTests: Input2DBCE2DCase
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 1, stride: 1,
            activation: Sigmoid.str, biases: true, bn: false, params: params
        )
        
        _ = try! BCE2D(layerPrev: layer, params: params)
    }
    
    func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class BCE2DFlowPrecisionTests: BCE2DFlowTests
{
    private func _buildTrainer() -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class BCE2DFlowResetTests: BCE2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class BCE2DFlowReverseTests: BCE2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class BCE2DFlowInferenceTests: BCE2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class BCE2DLoadTests: BCE2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class BCE2DTransformTests: BCE2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class BCESigmoid2DGradTests: Input2DBCESigmoid2DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 1, stride: 1,
            activation: nil, biases: true, bn: false, params: params
        )
        
        _ = try! BCESigmoid2D(layerPrev: layer, params: params)
    }
    
    func testLossCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testLossGPU() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class BCESigmoid2DFlowTests: Input2DBCESigmoid2DCase
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 1, stride: 1,
            activation: nil, biases: true, bn: false, params: params
        )
        
        _ = try! BCESigmoid2D(layerPrev: layer, params: params)
    }
    
    func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class BCESigmoid2DFlowPrecisionTests: BCESigmoid2DFlowTests
{
    private func _buildTrainer() -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class BCESigmoid2DFlowResetTests: BCESigmoid2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class BCESigmoid2DFlowReverseTests: BCESigmoid2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class BCESigmoid2DFlowInferenceTests: BCESigmoid2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class BCESigmoid2DLoadTests: BCESigmoid2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class BCESigmoid2DTransformTests: BCESigmoid2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class VQ2DFlowTests: Input2DVQ2DCase
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
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 1, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        _ = VQ2D(layerPrev: layer, K: 5, params: params)
    }
    
    func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class VQ2DFlowPrecisionTests: VQ2DFlowTests
{
    private func _buildTrainer() -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class VQ2DFlowResetTests: VQ2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class VQ2DFlowReverseTests: VQ2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class VQ2DFlowInferenceTests: VQ2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class VQ2DLoadTests: VQ2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class VQ2DTransformTests: VQ2DFlowTests
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
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// Tests for the LayerCAM2D layer.
class LayerCAM2DTests: XCTestCase
{
    var height = 6
    var width = 6
    
    /// Batch size of data.
    var batchSize: Int = -1
    /// Optimizer parameters.
    var optimizerParams = GrAI.Optimizer.Params()
    
    /// Systematic call before test begins.
    override func setUp()
    {
        batchSize = 5
        _ = MetalKernel.get
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 3
    }
    
    ///
    /// Build the two branches of the model.
    ///
    /// - Returns:
    ///     (frist branch, last branch of the model).
    ///
    func buildModel() -> (Model, Model)
    {
        var context = ModelContext(name: "MainBranch", curID: 0)
        var params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        var head: Layer1D = AvgPool2D(layerPrev: layer, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
        
        let mainBranch = Model(model: context.model, modelsPrev: [])
        
        context = ModelContext(name: "SecondBranch", models: [mainBranch])
        params = GrAI.Model.Params(context: context)
        
        _ = try! LayerCAM2D(layerPrev: layer, params: params)
        
        let secondBranch = Model(model: context.model, modelsPrev: [mainBranch])
        
        return (mainBranch, secondBranch)
    }
    
    ///
    /// Get the current batch size of data.
    ///
    /// This function allows to simulate the fact that the batch size of data may be smalling during the
    /// last iteration of the training.
    ///
    /// - Parameter model: The model.
    /// - Returns: The batch size of data.
    ///
    func getBatchSize(_ model: Model) -> Int
    {
        if model.optimizerParams.step == model.optimizerParams.nbLoops-1
        {
            return batchSize / 2
        }
        else
        {
            return batchSize
        }
    }
    
    ///
    /// Create synthetic data.
    ///
    /// - Parameters:
    ///     - dim1: The first dimension of the data.
    ///     - dim2: The second dimension of the data.
    /// - Returns: The created data.
    ///
    func buildData<T: BinaryFloatingPoint>(dim1: Int, dim2: Int) -> [[T]]
    {
        var data = [[T]]()
        for _ in 0..<dim1
        {
            var data1 = [T]()
            for _ in 0..<dim2
            {
                data1.append(T(Double.random(in: -1.0..<1.0)))
            }
            data.append(data1)
        }
        return data
    }
    
    ///
    /// A function to create/set data to the model.
    ///
    /// - Parameters:
    ///     - inputs: The data to set.
    ///     - model: The model.
    /// - Returns: (The data, the batch size).
    ///
    func setData(_ inputs: [[Double]]?, _ model: Model) -> ([[Double]], Int)
    {
        let firstLayer = model.layers.first as! Input2D
        let ins: [[Double]]
        if let insTmp = inputs
        {
            ins = insTmp
        }
        else
        {
            ins = buildData(dim1: getBatchSize(model), dim2: height * width)
        }
        
        if GrAI.Opti.GPU
        {
            try! firstLayer.setDataGPU(
                ins.reduce([], +),
                batchSize: ins.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
        else
        {
            try! firstLayer.setDataCPU(
                ins.reduce([], +),
                batchSize: ins.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
        return (ins, ins.count)
    }
    
    func testPrecision() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        
        let (mainFloat, secondFloat) = buildModel()
        let (mainFloat16, secondFloat16) = buildModel()
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
        randomSelectWeightsInitializationScheme(model: mainFloat)
        
        mainFloat.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondFloat.initKernel(
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        mainFloat16.weights = mainFloat.weights
        
        GrAI.Precision.float16 = true
        mainFloat16.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondFloat16.initKernel(
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let lastLayerFloat = mainFloat.layers.last as! MSE1D
        let gradLayerFloat = secondFloat.layers.last as! LayerCAM2D
        let lastLayerFloat16 = mainFloat16.layers.last as! MSE1D
        let gradLayerFloat16 = secondFloat16.layers.last as! LayerCAM2D
        
        lastLayerFloat.coeff = -1.0
        lastLayerFloat16.coeff = -1.0
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            if numLoop % 2 == 0
            {
                gradLayerFloat.keepPositive = true
                gradLayerFloat16.keepPositive = true
            }
            else
            {
                gradLayerFloat.keepPositive = false
                gradLayerFloat16.keepPositive = false
            }
            GrAI.Precision.float = true
            
            let (inputs, batchSize) = setData(nil, mainFloat)
            mainFloat.updateKernel(batchSize: batchSize)
            secondFloat.updateKernel(batchSize: batchSize)
            
            try! mainFloat.forward()
            try! lastLayerFloat.lossDerivativeGPU(
                [[Double]](repeating: [1.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainFloat.backward()
            try! mainFloat.update()
            
            try! secondFloat.forward()
            var valuesFloat = [Float]()
            for elem in 0..<batchSize
            {
                valuesFloat += gradLayerFloat.getOutsGPU(elem: elem)
            }
            
            GrAI.Precision.float16 = true
            
            _ = setData(inputs, mainFloat16)
            mainFloat16.updateKernel(batchSize: batchSize)
            secondFloat16.updateKernel(batchSize: batchSize)
            
            try! mainFloat16.forward()
            try! lastLayerFloat16.lossDerivativeGPU(
                [[Double]](repeating: [1.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainFloat16.backward()
            try! mainFloat16.update()
            
            try! secondFloat16.forward()
            var valuesFloat16 = [Float]()
            for elem in 0..<batchSize
            {
                valuesFloat16 += gradLayerFloat16.getOutsGPU(elem: elem)
            }
            
            for (elem1, elem2) in zip(valuesFloat, valuesFloat16)
            {
                if elem1 == 0
                {
                    XCTAssert(elem2 == 0)
                }
                else
                {
                    let diff = (elem1 - elem2) * (elem1 - elem2) /
                               (elem1 * elem1 + elem2 * elem2)
                    XCTAssert(diff < 0.005)
                }
            }
            
            mainFloat.incStep()
            mainFloat16.incStep()
            numLoop += 1
        }
    }
    
    func testInference()
    {
        let (mainCPU, secondCPU) = buildModel()
        let (mainGPU, secondGPU) = buildModel()
        
        GrAI.Opti.CPU = true
        randomSelectWeightsInitializationScheme(model: mainCPU)
        
        mainCPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondCPU.initKernel(
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        mainGPU.weights = mainCPU.weights
        
        GrAI.Opti.GPU = true
        mainGPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondGPU.initKernel(
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let lastLayerCPU = mainCPU.layers.last as! MSE1D
        let gradLayerCPU = secondCPU.layers.last as! LayerCAM2D
        let lastLayerGPU = mainGPU.layers.last as! MSE1D
        let gradLayerGPU = secondGPU.layers.last as! LayerCAM2D
        
        lastLayerCPU.coeff = -1.0
        lastLayerGPU.coeff = -1.0
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            if numLoop % 2 == 0
            {
                gradLayerCPU.keepPositive = true
                gradLayerGPU.keepPositive = true
            }
            else
            {
                gradLayerCPU.keepPositive = false
                gradLayerGPU.keepPositive = false
            }
            GrAI.Opti.CPU = true
            
            let (inputs, batchSize) = setData(nil, mainCPU)
            mainCPU.updateKernel(batchSize: batchSize)
            secondCPU.updateKernel(batchSize: batchSize)
            
            try! mainCPU.forward()
            try! lastLayerCPU.lossDerivativeCPU(
                [[Double]](repeating: [1.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainCPU.backward()
            try! mainCPU.update()
            
            try! secondCPU.forward()
            var valuesCPU = [Float]()
            for elem in 0..<batchSize
            {
                valuesCPU += gradLayerCPU.getOutsCPU(elem: elem)
            }
            
            GrAI.Opti.GPU = true
            
            _ = setData(inputs, mainGPU)
            mainGPU.updateKernel(batchSize: batchSize)
            secondGPU.updateKernel(batchSize: batchSize)
            
            try! mainGPU.forward()
            try! lastLayerGPU.lossDerivativeGPU(
                [[Double]](repeating: [1.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainGPU.backward()
            try! mainGPU.update()
            
            try! secondGPU.forward()
            var valuesGPU = [Float]()
            for elem in 0..<batchSize
            {
                valuesGPU += gradLayerGPU.getOutsGPU(elem: elem)
            }
            
            for (elem1, elem2) in zip(valuesCPU, valuesGPU)
            {
                let diff = (elem1 - elem2) * (elem1 - elem2) /
                           (elem1 * elem1 + elem2 * elem2)
                XCTAssert(diff < 0.00001)
            }
            
            mainCPU.incStep()
            mainGPU.incStep()
            numLoop += 1
        }
    }
    
    func testLoad()
    {
        GrAI.Opti.GPU = true
        var (mainBranch, secondBranch) = buildModel()
        
        randomSelectWeightsInitializationScheme(model: mainBranch)
        
        mainBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondBranch.initKernel(
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let folderURL = FileManager.default.temporaryDirectory
        let mainPath =
            folderURL.appendingPathComponent("testMain.plist").path
        let secondPath =
            folderURL.appendingPathComponent("testSecond.plist").path
        
        let encoder = PropertyListEncoder()
    
        var data = try! encoder.encode(mainBranch)
        try! data.write(to: URL(fileURLWithPath: mainPath))
        
        data = try! encoder.encode(secondBranch)
        try! data.write(to: URL(fileURLWithPath: secondPath))
        
        data = try! Data(contentsOf: URL(fileURLWithPath: mainPath))
        let mainBase = try! PropertyListDecoder().decode(
            BaseModel.self, from: data
        )
        data = try! Data(contentsOf: URL(fileURLWithPath: secondPath))
        let secondBase = try! PropertyListDecoder().decode(
            BaseModel.self, from: data
        )
        
        mainBranch = Model(model: mainBase, modelsPrev: [])
        secondBranch = Model(model: secondBase, modelsPrev: [mainBranch])
        
        mainBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondBranch.initKernel(
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let lastLayer = mainBranch.layers.last as! MSE1D
        let gradLayer = secondBranch.layers.last as! LayerCAM2D
        
        lastLayer.coeff = -1.0
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            let (_, batchSize) = setData(nil, mainBranch)
            mainBranch.updateKernel(batchSize: batchSize)
            secondBranch.updateKernel(batchSize: batchSize)
            
            try! mainBranch.forward()
            try! lastLayer.lossDerivativeGPU(
                [[Double]](repeating: [0.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainBranch.backward()
            try! mainBranch.update()
            
            try! secondBranch.forward()
            var values = [Float]()
            for elem in 0..<batchSize
            {
                values += gradLayer.getOutsGPU(elem: elem)
            }
            
            mainBranch.incStep()
            numLoop += 1
        }
    }
    
    func testTransform()
    {
        GrAI.Opti.GPU = true
        var (mainBranch, secondBranch) = buildModel()
        
        randomSelectWeightsInitializationScheme(model: mainBranch)
        
        mainBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondBranch.initKernel(
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let branches = Model.copy(
            models: [mainBranch, secondBranch],
            inPlace: true
        )
        mainBranch = branches[0]
        secondBranch = branches[1]
        
        mainBranch.setupOptimizers(params: optimizerParams)
        mainBranch.phase = .Inference
        secondBranch.phase = .Inference
        
        let lastLayer = mainBranch.layers.last as! MSE1D
        let gradLayer = secondBranch.layers.last as! LayerCAM2D
        
        lastLayer.coeff = -1.0
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            let (_, batchSize) = setData(nil, mainBranch)
            mainBranch.updateKernel(batchSize: batchSize)
            secondBranch.updateKernel(batchSize: batchSize)
            
            try! mainBranch.forward()
            try! lastLayer.lossDerivativeGPU(
                [[Double]](repeating: [0.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainBranch.backward()
            try! mainBranch.update()
            
            try! secondBranch.forward()
            var values = [Float]()
            for elem in 0..<batchSize
            {
                values += gradLayer.getOutsGPU(elem: elem)
            }
            
            mainBranch.incStep()
            numLoop += 1
        }
    }
}

// Tests for the VQGrad2D layer.
class VQGrad2DTests: XCTestCase
{
    var height = 6
    var width = 6
    
    /// Batch size of data.
    var batchSize: Int = -1
    /// Optimizer parameters.
    var optimizerParams = GrAI.Optimizer.Params()
    
    /// Systematic call before test begins.
    override func setUp()
    {
        batchSize = 5
        _ = MetalKernel.get
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 3
    }
    
    ///
    /// Build the two branches of the model.
    ///
    /// - Returns:
    ///     (frist branch, last branch of the model).
    ///
    func buildModel() -> (Model, Model)
    {
        var context = ModelContext(name: "MainBranch", curID: 0)
        var params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        var head: Layer1D = AvgPool2D(layerPrev: layer, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
        
        let mainBranch = Model(model: context.model, modelsPrev: [])
        
        context = ModelContext(name: "SecondBranch", models: [mainBranch])
        params = GrAI.Model.Params(context: context)
        
        _ = VQGrad2D(layerPrev: layer, K: 5, params: params)
        
        let secondBranch = Model(model: context.model, modelsPrev: [mainBranch])
        
        return (mainBranch, secondBranch)
    }
    
    ///
    /// Get the current batch size of data.
    ///
    /// This function allows to simulate the fact that the batch size of data may be smalling during the
    /// last iteration of the training.
    ///
    /// - Parameter model: The model.
    /// - Returns: The batch size of data.
    ///
    func getBatchSize(_ model: Model) -> Int
    {
        if model.optimizerParams.step == model.optimizerParams.nbLoops-1
        {
            return batchSize / 2
        }
        else
        {
            return batchSize
        }
    }
    
    ///
    /// Create synthetic data.
    ///
    /// - Parameters:
    ///     - dim1: The first dimension of the data.
    ///     - dim2: The second dimension of the data.
    /// - Returns: The created data.
    ///
    func buildData<T: BinaryFloatingPoint>(dim1: Int, dim2: Int) -> [[T]]
    {
        var data = [[T]]()
        for _ in 0..<dim1
        {
            var data1 = [T]()
            for _ in 0..<dim2
            {
                data1.append(T(Double.random(in: -1.0..<1.0)))
            }
            data.append(data1)
        }
        return data
    }
    
    ///
    /// A function to create/set data to the model.
    ///
    /// - Parameters:
    ///     - inputs: The data to set.
    ///     - model: The model.
    /// - Returns: (The data, the batch size).
    ///
    func setData(_ inputs: [[Double]]?, _ model: Model) -> ([[Double]], Int)
    {
        let firstLayer = model.layers.first as! Input2D
        let ins: [[Double]]
        if let insTmp = inputs
        {
            ins = insTmp
        }
        else
        {
            ins = buildData(dim1: getBatchSize(model), dim2: height * width)
        }
        
        if GrAI.Opti.GPU
        {
            try! firstLayer.setDataGPU(
                ins.reduce([], +),
                batchSize: ins.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
        else
        {
            try! firstLayer.setDataCPU(
                ins.reduce([], +),
                batchSize: ins.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
        return (ins, ins.count)
    }
    
    func testPrecision() throws
    {
        throw XCTSkip("Skipping this test because of precision issue.")
        
        let (mainFloat, secondFloat) = buildModel()
        let (mainFloat16, secondFloat16) = buildModel()
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
        randomSelectWeightsInitializationScheme(model: mainFloat)
        randomSelectWeightsInitializationScheme(model: secondFloat)
        
        mainFloat.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondFloat.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        mainFloat16.weights = mainFloat.weights
        secondFloat16.weights = secondFloat.weights
        
        GrAI.Precision.float16 = true
        mainFloat16.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondFloat16.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let lastLayerFloat = mainFloat.layers.last as! MSE1D
        let gradLayerFloat = secondFloat.layers.last as! VQGrad2D
        let lastLayerFloat16 = mainFloat16.layers.last as! MSE1D
        let gradLayerFloat16 = secondFloat16.layers.last as! VQGrad2D
        
        lastLayerFloat.coeff = -1.0
        lastLayerFloat16.coeff = -1.0
        gradLayerFloat.magnitudeCoeff = 0.6
        gradLayerFloat16.magnitudeCoeff = 0.6
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            if numLoop % 2 == 0
            {
                gradLayerFloat.keepPositive = true
                gradLayerFloat16.keepPositive = true
            }
            else
            {
                gradLayerFloat.keepPositive = false
                gradLayerFloat16.keepPositive = false
            }
            GrAI.Precision.float = true
            
            let (inputs, batchSize) = setData(nil, mainFloat)
            mainFloat.updateKernel(batchSize: batchSize)
            secondFloat.updateKernel(batchSize: batchSize)
            
            try! mainFloat.forward()
            try! lastLayerFloat.lossDerivativeGPU(
                [[Double]](repeating: [1.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainFloat.backward()
            try! mainFloat.update()
            
            try! secondFloat.forward()
            try! gradLayerFloat.lossDerivativeGPU()
            let lossFloat: Double = try! gradLayerFloat.getLossGPU()
            try! secondFloat.update()
            
            GrAI.Precision.float16 = true
            
            _ = setData(inputs, mainFloat16)
            mainFloat16.updateKernel(batchSize: batchSize)
            secondFloat16.updateKernel(batchSize: batchSize)
            
            try! mainFloat16.forward()
            try! lastLayerFloat16.lossDerivativeGPU(
                [[Double]](repeating: [1.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainFloat16.backward()
            try! mainFloat16.update()
            
            try! secondFloat16.forward()
            try! gradLayerFloat16.lossDerivativeGPU()
            let lossFloat16: Double = try! gradLayerFloat16.getLossGPU()
            try! secondFloat16.update()
            
            let diff = (lossFloat16 - lossFloat) * (lossFloat16 - lossFloat) /
                       (lossFloat * lossFloat + lossFloat16 * lossFloat16)
            print(diff)
            XCTAssert(diff < 0.005)
            
            mainFloat.incStep()
            secondFloat.incStep()
            mainFloat16.incStep()
            secondFloat16.incStep()
            numLoop += 1
        }
    }
    
    func testInference()
    {
        let (mainCPU, secondCPU) = buildModel()
        let (mainGPU, secondGPU) = buildModel()
        
        GrAI.Opti.CPU = true
        randomSelectWeightsInitializationScheme(model: mainCPU)
        randomSelectWeightsInitializationScheme(model: secondCPU)
        
        mainCPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondCPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        mainGPU.weights = mainCPU.weights
        secondGPU.weights = secondCPU.weights
        
        GrAI.Opti.GPU = true
        mainGPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondGPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let lastLayerCPU = mainCPU.layers.last as! MSE1D
        let gradLayerCPU = secondCPU.layers.last as! VQGrad2D
        let lastLayerGPU = mainGPU.layers.last as! MSE1D
        let gradLayerGPU = secondGPU.layers.last as! VQGrad2D
        
        lastLayerCPU.coeff = -1.0
        lastLayerGPU.coeff = -1.0
        gradLayerCPU.magnitudeCoeff = 0.6
        gradLayerGPU.magnitudeCoeff = 0.6
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            if numLoop % 2 == 0
            {
                gradLayerCPU.keepPositive = true
                gradLayerGPU.keepPositive = true
            }
            else
            {
                gradLayerCPU.keepPositive = false
                gradLayerGPU.keepPositive = false
            }
            GrAI.Opti.CPU = true
            
            let (inputs, batchSize) = setData(nil, mainCPU)
            mainCPU.updateKernel(batchSize: batchSize)
            secondCPU.updateKernel(batchSize: batchSize)
            
            try! mainCPU.forward()
            try! lastLayerCPU.lossDerivativeCPU(
                [[Double]](repeating: [0.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainCPU.backward()
            try! mainCPU.update()
            
            try! secondCPU.forward()
            try! gradLayerCPU.lossDerivativeCPU()
            let lossCPU: Double = gradLayerCPU.getLossCPU()
            try! secondCPU.update()
            
            GrAI.Opti.GPU = true
            
            _ = setData(inputs, mainGPU)
            mainGPU.updateKernel(batchSize: batchSize)
            secondGPU.updateKernel(batchSize: batchSize)
            
            try! mainGPU.forward()
            try! lastLayerGPU.lossDerivativeGPU(
                [[Double]](repeating: [0.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainGPU.backward()
            try! mainGPU.update()
            
            try! secondGPU.forward()
            try! gradLayerGPU.lossDerivativeGPU()
            let lossGPU: Double = try! gradLayerGPU.getLossGPU()
            try! secondGPU.update()
            
            let diff = (lossGPU - lossCPU) * (lossGPU - lossCPU) /
                       (lossCPU * lossCPU + lossGPU * lossGPU)
            print(diff)
            XCTAssert(diff < 0.001)
            
            mainCPU.incStep()
            secondCPU.incStep()
            mainGPU.incStep()
            secondGPU.incStep()
            numLoop += 1
        }
    }
    
    func testLoad()
    {
        GrAI.Opti.GPU = true
        var (mainBranch, secondBranch) = buildModel()
        
        randomSelectWeightsInitializationScheme(model: mainBranch)
        randomSelectWeightsInitializationScheme(model: secondBranch)
        
        mainBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let folderURL = FileManager.default.temporaryDirectory
        let mainPath =
            folderURL.appendingPathComponent("testMain.plist").path
        let secondPath =
            folderURL.appendingPathComponent("testSecond.plist").path
        
        let encoder = PropertyListEncoder()
    
        var data = try! encoder.encode(mainBranch)
        try! data.write(to: URL(fileURLWithPath: mainPath))
        
        data = try! encoder.encode(secondBranch)
        try! data.write(to: URL(fileURLWithPath: secondPath))
        
        data = try! Data(contentsOf: URL(fileURLWithPath: mainPath))
        let mainBase = try! PropertyListDecoder().decode(
            BaseModel.self, from: data
        )
        data = try! Data(contentsOf: URL(fileURLWithPath: secondPath))
        let secondBase = try! PropertyListDecoder().decode(
            BaseModel.self, from: data
        )
        
        mainBranch = Model(model: mainBase, modelsPrev: [])
        secondBranch = Model(model: secondBase, modelsPrev: [mainBranch])
        
        mainBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let lastLayer = mainBranch.layers.last as! MSE1D
        let gradLayer = secondBranch.layers.last as! VQGrad2D
        
        lastLayer.coeff = -1.0
        gradLayer.magnitudeCoeff = 0.6
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            let (_, batchSize) = setData(nil, mainBranch)
            mainBranch.updateKernel(batchSize: batchSize)
            secondBranch.updateKernel(batchSize: batchSize)
            
            try! mainBranch.forward()
            try! lastLayer.lossDerivativeGPU(
                [[Double]](repeating: [0.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainBranch.backward()
            try! mainBranch.update()
            
            try! secondBranch.forward()
            try! gradLayer.lossDerivativeGPU()
            let lossVal: Double = try! gradLayer.getLossGPU()
            try! secondBranch.update()
            
            print(lossVal)
            
            mainBranch.incStep()
            secondBranch.incStep()
            numLoop += 1
        }
    }
    
    func testTransform()
    {
        GrAI.Opti.GPU = true
        var (mainBranch, secondBranch) = buildModel()
        
        randomSelectWeightsInitializationScheme(model: mainBranch)
        randomSelectWeightsInitializationScheme(model: secondBranch)
        
        mainBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        secondBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let branches = Model.copy(
            models: [mainBranch, secondBranch],
            inPlace: true
        )
        mainBranch = branches[0]
        secondBranch = branches[1]
        
        mainBranch.setupOptimizers(params: optimizerParams)
        secondBranch.setupOptimizers(params: optimizerParams)
        mainBranch.phase = .Inference
        secondBranch.phase = .Inference
        
        let lastLayer = mainBranch.layers.last as! MSE1D
        let gradLayer = secondBranch.layers.last as! VQGrad2D
        
        lastLayer.coeff = -1.0
        gradLayer.magnitudeCoeff = 0.6
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            let (_, batchSize) = setData(nil, mainBranch)
            mainBranch.updateKernel(batchSize: batchSize)
            secondBranch.updateKernel(batchSize: batchSize)
            
            try! mainBranch.forward()
            try! lastLayer.lossDerivativeGPU(
                [[Double]](repeating: [0.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainBranch.backward()
            try! mainBranch.update()
            
            try! secondBranch.forward()
            try! gradLayer.lossDerivativeGPU()
            let lossVal: Double = try! gradLayer.getLossGPU()
            try! secondBranch.update()
            
            print(lossVal)
            
            mainBranch.incStep()
            secondBranch.incStep()
            numLoop += 1
        }
    }
}
