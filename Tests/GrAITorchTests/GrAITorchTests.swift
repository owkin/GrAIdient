//
// GrAITorchTests.swift
// GrAITorchTests
//
// Created by Jean-François Reboud on 19/10/2022.
//

import XCTest
import GrAIdient

/// Compare models created by GrAIdient and PyTorch.
final class GrAITorchTests: XCTestCase
{
    /// Size of one image (height and width are the same).
    let _size = 32
    /// Kernel split size of one image (height and width are the same).
    let _patch = 8
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
    }
    
    ///
    /// Compute the gradient norm on the first layer of the model.
    ///
    /// - Parameters:
    ///     - model: The model we want to evalulate the gradient norm on.
    ///     - size: The size of the input data.
    /// - Returns: The gradient norm on the first layer.
    ///
    func _getGradientNormMSE1D(model: Model, size: Int) -> Double
    {
        // Create the context to build a graph of layers
        // that come after the layers inside `model`.
        let context = ModelContext(name: "ModelTest", models: [model])
        let params = GrAI.Model.Params(context: context)
        
        // Append a loss layer.
        let lastLayer = MSE1D(
            layerPrev: model.layers.last! as! Layer1D,
            params: params
        )
        lastLayer.coeff = 1.0 / 2.0
        
        // Initialize the finalModel with the links (`layerPrev` updated).
        context.model.layers = model.layers + context.model.layers
        let finalModel = Model(model: context.model, modelsPrev: [])
        
        // Initialize for inference.
        finalModel.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let optimizerParams = getOptimizerParams(nbLoops: 1)
        finalModel.setupOptimizers(params: optimizerParams)
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Update internal batch size.
        finalModel.updateKernel(batchSize: 1)
        
        // Forward.
        try! finalModel.forward()
        
        // Apply loss derivative.
        let groundTruth: [[Double]] = [[0.0]]
        try! lastLayer.lossDerivativeGPU(groundTruth)
        
        // Backward.
        try! finalModel.backward()
        
        // Get the gradient norm on the first layer.
        let gradNormOutput: Double =
            try! finalModel.getGradientNorm(layers: [firstLayer])
        return gradNormOutput
    }
    
    ///
    /// Compute the gradient norm on the first layer of the model.
    ///
    /// - Parameters:
    ///     - model: The model we want to evalulate the gradient norm on.
    ///     - size: The size of the input data.
    /// - Returns: The gradient norm on the first layer.
    ///
    func _getGradientNormMSE2D(model: Model, size: Int) -> Double
    {
        // Create the context to build a graph of layers
        // that come after the layers inside `model`.
        let context = ModelContext(name: "ModelTest", models: [model])
        let params = GrAI.Model.Params(context: context)
        
        // Append a loss layer.
        let lastLayer = MSE2D(
            layerPrev: model.layers.last! as! Layer2D,
            params: params
        )
        
        // Initialize the finalModel with the links (`layerPrev` updated).
        context.model.layers = model.layers + context.model.layers
        let finalModel = Model(model: context.model, modelsPrev: [model])
        
        // Initialize for inference.
        finalModel.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let optimizerParams = getOptimizerParams(nbLoops: 1)
        finalModel.setupOptimizers(params: optimizerParams)
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Update internal batch size.
        finalModel.updateKernel(batchSize: 1)
        
        // Forward.
        try! finalModel.forward()
        
        // Apply loss derivative.
        try! lastLayer.lossDerivativeGPU(data, batchSize: 1, format: .RGB)
        
        // Backward.
        try! finalModel.backward()
        
        // Get the gradient norm on the first layer.
        let gradNormOutput: Double =
            try! finalModel.getGradientNorm(layers: [firstLayer])
        return gradNormOutput
    }
    
    ///
    /// Compute the gradient norm on the first layer of the model.
    ///
    /// - Parameters:
    ///     - model: The model we want to evalulate the gradient norm on.
    ///     - size: The size of the input data.
    ///     - batchSize: The number of images in the batch.
    /// - Returns: The gradient norm on the first layer.
    ///
    func _getGradientNormGram2D(
        model: Model,
        size: Int,
        batchSize: Int) -> Double
    {
        // Create the context to build a graph of layers
        // that come after the layers inside `model`.
        let context = ModelContext(name: "ModelTest", models: [model])
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = SelfCorrelate2D(
            layerPrev: model.layers[1] as! Layer2D,
            params: params
        )
        
        layer = Normalize122D(layerPrev: layer, params: params)
        
        let similarityLayer = SimilarityBatchError2D(
            layerPrev: layer,
            params: params
        )
        
        // Append a loss layer.
        let lastLayer = MSE1D(
            layerPrev: model.layers.last! as! Layer1D,
            params: params
        )
        lastLayer.coeff = 1.0 / 2.0
        
        // Initialize the finalModel with the links (`layerPrev` updated).
        context.model.layers = model.layers + context.model.layers
        let finalModel = Model(model: context.model, modelsPrev: [])
        
        // Initialize for inference.
        finalModel.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let optimizerParams = getOptimizerParams(nbLoops: 1)
        finalModel.setupOptimizers(params: optimizerParams)
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getBatchData(size: size, batchSize: batchSize)
        try! firstLayer.setDataGPU(data, batchSize: batchSize, format: .RGB)
        
        // Update internal batch size.
        finalModel.updateKernel(batchSize: batchSize)
        
        // Forward.
        try! finalModel.forward()
        
        // Apply loss derivative.
        var groundTruth = [[Double]]()
        for _ in 0..<batchSize
        {
            groundTruth.append([0.0])
        }
        try! similarityLayer.lossDerivativeGPU()
        try! lastLayer.lossDerivativeGPU(groundTruth)
        
        // Backward.
        try! finalModel.backward()
        
        // Get the gradient norm on the first layer.
        let gradNormOutput: Double =
            try! finalModel.getGradientNorm(layers: [firstLayer])
        return gradNormOutput
    }
    
    /// Test that modelConv1 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelConv1()
    {
        // Build model.
        let model = ModelTestConv1.build(_size)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConv1GradNorm(_size))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelConv2 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelConv2()
    {
        // Build model.
        let model = ModelTestConv2.build(_size)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConv2GradNorm(_size))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelFFT backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelFFT()
    {
        // Build model.
        let model = ModelTestFFT.build(_size)
        
        // Create the context to build a graph of layers
        // that come after the layers inside `model`.
        let context = ModelContext(name: "ModelTest", models: [model])
        let params = GrAI.Model.Params(context: context)
        
        // Append a loss layer.
        let lastLayer = MSE1D(
            layerPrev: model.layers.last! as! Layer1D,
            params: params
        )
        lastLayer.coeff = 1.0 / 2.0
        
        // Initialize the finalModel with the links (`layerPrev` updated).
        context.model.layers = model.layers + context.model.layers
        let finalModel = Model(model: context.model, modelsPrev: [])
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let optimizerParams = getOptimizerParams(nbLoops: 1)
        finalModel.setupOptimizers(params: optimizerParams)
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        let secondLayer: FTFrequences2D = model.layers[1] as! FTFrequences2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getComplexData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        try! secondLayer.setDataGPU(batchSize: 1)
        
        // Update internal batch size.
        finalModel.updateKernel(batchSize: 1)
        
        // Forward.
        try! finalModel.forward()
        
        // Apply loss derivative.
        let groundTruth: [[Double]] = [[0.0]]
        try! lastLayer.lossDerivativeGPU(groundTruth)
        
        // Backward.
        try! finalModel.backward()
        
        // Get the gradient norm on the first layer.
        let gradNormOutput: Double =
            try! finalModel.getGradientNorm(layers: [firstLayer])
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeFFTGradNorm(_size))
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS1K1 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS1K1()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 1, kernel: 1)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 1, kernel: 1
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS1K2 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS1K2()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 1, kernel: 2)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 1, kernel: 2
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS1K3 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS1K3()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 1, kernel: 3)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 1, kernel: 3
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS1K4 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS1K4()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 1, kernel: 4)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 1, kernel: 4
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS1K5 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS1K5()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 1, kernel: 5)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 1, kernel: 5
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS2K1 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS2K1()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 2, kernel: 1)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 2, kernel: 1
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS2K2 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS2K2()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 2, kernel: 2)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 2, kernel: 2
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS2K3 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS2K3()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 2, kernel: 3)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 2, kernel: 3
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS2K4 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS2K4()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 2, kernel: 4)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 2, kernel: 4
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS2K5 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS2K5()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 2, kernel: 5)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 2, kernel: 5
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS4K1 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS4K1()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 4, kernel: 1)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 4, kernel: 1
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS4K2 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS4K2()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 4, kernel: 2)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 4, kernel: 2
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS4K3 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS4K3()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 4, kernel: 3)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 4, kernel: 3
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS4K4 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS4K4()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 4, kernel: 4)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 4, kernel: 4
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelConvS4K5 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelConvS4K5()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 4, kernel: 5)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 4, kernel: 5
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS1K1 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS1K1()
    {
        // Build model.
        let model = ModelTestDeConvSK.build(size: _size, stride: 1, kernel: 1)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConvSKGradNorm(
            size: _size, stride: 1, kernel: 1
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS1K2 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS1K2()
    {
        // Build model.
        let model = ModelTestDeConvSK.build(size: _size, stride: 1, kernel: 2)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConvSKGradNorm(
            size: _size, stride: 1, kernel: 2
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS1K3 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS1K3()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 1, kernel: 3)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 1, kernel: 3
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS1K4 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS1K4()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 1, kernel: 4)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 1, kernel: 4
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS1K5 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS1K5()
    {
        // Build model.
        let model = ModelTestDeConvSK.build(size: _size, stride: 1, kernel: 5)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConvSKGradNorm(
            size: _size, stride: 1, kernel: 5
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS2K1 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS2K1()
    {
        // Build model.
        let model = ModelTestDeConvSK.build(size: _size, stride: 2, kernel: 1)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConvSKGradNorm(
            size: _size, stride: 2, kernel: 1
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS2K2 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS2K2()
    {
        // Build model.
        let model = ModelTestDeConvSK.build(size: _size, stride: 2, kernel: 2)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConvSKGradNorm(
            size: _size, stride: 2, kernel: 2
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS2K3 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS2K3()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 2, kernel: 3)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 2, kernel: 3
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS2K4 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS2K4()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 2, kernel: 4)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 2, kernel: 4
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS2K5 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS2K5()
    {
        // Build model.
        let model = ModelTestDeConvSK.build(size: _size, stride: 2, kernel: 5)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConvSKGradNorm(
            size: _size, stride: 2, kernel: 5
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS4K1 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS4K1()
    {
        // Build model.
        let model = ModelTestDeConvSK.build(size: _size, stride: 4, kernel: 1)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConvSKGradNorm(
            size: _size, stride: 4, kernel: 1
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS4K2 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS4K2()
    {
        // Build model.
        let model = ModelTestDeConvSK.build(size: _size, stride: 4, kernel: 2)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConvSKGradNorm(
            size: _size, stride: 4, kernel: 2
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS4K3 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS4K3()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 4, kernel: 3)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 4, kernel: 3
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS4K4 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS4K4()
    {
        // Build model.
        let model = ModelTestConvSK.build(size: _size, stride: 4, kernel: 4)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConvSKGradNorm(
            size: _size, stride: 4, kernel: 4
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelDeConvS4K5 backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelDeConvS4K5()
    {
        // Build model.
        let model = ModelTestDeConvSK.build(size: _size, stride: 4, kernel: 5)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConvSKGradNorm(
            size: _size, stride: 4, kernel: 5
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelCat backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelCat()
    {
        // Build model.
        let model = ModelTestCat.build(_size)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeCatGradNorm(_size))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelResize backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelResizeBilinear1()
    {
        let sizeOutput = Int(round(0.8 * Double(_size)))
        
        // Build model.
        let model = ModelTestResizeBilinear.build(
            sizeInput: _size, sizeOutput: sizeOutput
        )
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeResizeGradNorm(
            sizeInput: _size, sizeOutput: sizeOutput
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelResize backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelResizeBilinear2()
    {
        let sizeOutput = Int(round(1.2 * Double(_size)))
        
        // Build model.
        let model = ModelTestResizeBilinear.build(
            sizeInput: _size, sizeOutput: sizeOutput
        )
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeResizeGradNorm(
            sizeInput: _size, sizeOutput: sizeOutput
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelResize backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelResize3()
    {
        let sizeOutput = 2 * _size
        
        // Build model.
        let model = ModelTestResizeBilinear.build(
            sizeInput: _size, sizeOutput: sizeOutput
        )
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeResizeGradNorm(
            sizeInput: _size, sizeOutput: sizeOutput
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelResize backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelResizeBilinearPad1()
    {
        let sizeOutput = Int(round(0.8 * Double(_size)))
        
        // Build model.
        let model = ModelTestResizeBilinearPad.build(
            sizeInput: _size, sizeOutput: sizeOutput
        )
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeResizeGradNorm(
            sizeInput: _size, sizeOutput: sizeOutput
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelResize backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelResizeBilinearPad2()
    {
        let sizeOutput = Int(round(1.2 * Double(_size)))
        
        // Build model.
        let model = ModelTestResizeBilinearPad.build(
            sizeInput: _size, sizeOutput: sizeOutput
        )
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeResizeGradNorm(
            sizeInput: _size, sizeOutput: sizeOutput
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelResize backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelResizePad3()
    {
        let sizeOutput = 2 * _size
        
        // Build model.
        let model = ModelTestResizeBilinearPad.build(
            sizeInput: _size, sizeOutput: sizeOutput
        )
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeResizeGradNorm(
            sizeInput: _size, sizeOutput: sizeOutput
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelResize backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelResizeBilinearCrop1()
    {
        let sizeOutput = Int(round(0.8 * Double(_size)))
        
        // Build model.
        let model = ModelTestResizeBilinearCrop.build(
            sizeInput: _size, sizeOutput: sizeOutput
        )
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeResizeGradNorm(
            sizeInput: _size, sizeOutput: sizeOutput
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelPatchConv backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelPatchConv()
    {
        // Build model.
        let model = ModelTestPatchConv.build(size: _size, patch: _patch)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computePatchConvGradNorm(
            size: _size, patch: _patch
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelAttention1 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelAttention1()
    {
        // Build model.
        let model = ModelTestAttention1.build(size: _size, patch: _patch)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeAttention1GradNorm(
            size: _size, patch: _patch
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelAttention2 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelAttention2()
    {
        // Build model.
        let model = ModelTestAttention2.build(size: _size, patch: _patch)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeAttention2GradNorm(
            size: _size, patch: _patch
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelLayerNorm backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelLayerNorm()
    {
        // Build model.
        let model = ModelTestLayerNorm.build(size: _size, patch: _patch)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeLayerNormGradNorm(
            size: _size, patch: _patch
        ))
        let gradNormOutput: Double = _getGradientNormMSE1D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    func testModelAutoEncoder1()
    {
        let model = ModelTestAutoEncoder1.build(_size)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeAutoEncoder1GradNorm(_size))
        let gradNormOutput: Double = _getGradientNormMSE2D(
            model: model, size: _size
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelGram backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelGram()
    {
        let batchSize = 4
        
        // Build model.
        let model = ModelTestGram.build(_size)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeGramGradNorm(
            size: _size, batchSize: batchSize
        ))
        let gradNormOutput: Double = _getGradientNormGram2D(
            model: model, size: _size, batchSize: batchSize
        )
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
}
