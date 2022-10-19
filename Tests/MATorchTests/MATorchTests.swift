//
// MATorchTests.swift
// MATorchTests
//
// Created by Jean-François Reboud on 19/10/2022.
//

import XCTest
import MAKit

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
    
    // Test loading weights from PyTorch, transforming target layer and
    // back propagating gradients to the first layer.
    func testModel1()
    {
        var model = ModelTest1.build(_size)
        
        let context = ModelContext(name: "ModelTest1", models: [model])
        let params = MAKit.Model.Params(context: context)
        
        let lastLayer = MSE1D(
            layerPrev: model.layers.last! as! Layer1D,
            params: params
        )
        lastLayer.coeff = -1.0 / 2.0
        
        model.layers += context.model.layers
        model = Model(model: model, modelsPrev: [])
        
        model.initKernel(phase: .Inference)
        model.computeDeltaWeights = false
        
        let optimizerParams = getOptimizerParams(nbLoops: 1)
        model.setupOptimizers(params: optimizerParams)
        
        let data: [Float] = getInputData(_size)
        let groundTruth: [[Double]] = [[0.0]]
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        firstLayer.computeDelta = true
        firstLayer.computeDeltaWeights = true
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        model.updateKernel(batchSize: 1)
        try! model.forward()
        try! lastLayer.applyGradientGPU(groundTruth)
        try! model.backward()
        
        let gradNormOutput: Double =
            try! model.getGradientNorm(layers: [firstLayer])
        
        let gradNormExpected: Double = Double(computeTest1GradNorm(_size))
        
        let diffPercent =
            abs(gradNormOutput - gradNormExpected) / gradNormExpected * 100.0
        XCTAssert(diffPercent < 1.0)
    }
}
