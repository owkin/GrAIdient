//
// MATorchTests.swift
// MATorchTests
//
// Created by Jean-François Reboud on 19/10/2022.
//

import XCTest
import MAKit

/// Compare models created by MAKit and PyTorch.
final class MATorchTests: XCTestCase
{
    /// Size of one image (height and width are the same).
    let _size = 32
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        MAKit.Opti.GPU = true
    }
    
    ///
    /// Compute gradient norm and compare with PyTorch.
    ///
    /// - Parameters:
    ///     - model: The model we want to evaludate the gradient norm.
    ///     - expectedNorm: The gradient norm computed in the Python environment.
    ///
    func _compareGradientNorm(_ model: Model, expectedNorm: Double)
    {
        // Create the context to build a graph of layers
        // that come after the layers inside `model`.
        let context = ModelContext(name: "ModelTest", models: [model])
        let params = MAKit.Model.Params(context: context)
        
        // Append a loss layer.
        let lastLayer = MSE1D(
            layerPrev: model.layers.last! as! Layer1D,
            params: params
        )
        lastLayer.coeff = 1.0 / 2.0
        
        // The final model contains the layers of `model` and the loss layer.
        // Initialize the model with the links (`layerPrev` updated).
        var finalModel = Model(name: "ModelTest")
        finalModel.layers = model.layers + context.model.layers
        finalModel = Model(model: finalModel, modelsPrev: [])
        
        // Initialize for inference.
        finalModel.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        finalModel.computeDeltaWeights = false
        
        let optimizerParams = getOptimizerParams(nbLoops: 1)
        finalModel.setupOptimizers(params: optimizerParams)
        
        let data: [Float] = getInputData(_size)
        let groundTruth: [[Double]] = [[0.0]]
        
        let firstLayer: Input2D = finalModel.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Update internal batch size.
        finalModel.updateKernel(batchSize: 1)
        
        // Forward.
        try! finalModel.forward()
        
        // Apply loss derivative.
        try! lastLayer.lossDerivativeGPU(groundTruth)
        
        // Backward.
        try! finalModel.backward()
        
        // Get the gradient norm on the first layer.
        let gradNormOutput: Double =
            try! finalModel.getGradientNorm(layers: [firstLayer])
        
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that model1 backward pass returns the same gradient norm in MAKit and PyTorch.
    func testModel1()
    {
        let model = ModelTest1.build(_size)
        let expectedNorm: Double = Double(computeTest1GradNorm(_size))
        _compareGradientNorm(model, expectedNorm: expectedNorm)
    }
    
    /// Test that model2 backward pass returns the same gradient norm in MAKit and PyTorch.
    func testModel2()
    {
        let model = ModelTest2.build(_size)
        let expectedNorm: Double = Double(computeTest2GradNorm(_size))
        _compareGradientNorm(model, expectedNorm: expectedNorm)
    }
}
