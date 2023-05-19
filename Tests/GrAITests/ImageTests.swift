//
// ImageTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 19/05/2023.
//

import Foundation
import XCTest
import GrAIdient

/// Test operations on images.
@available(macOS 13.0, *)
class ImageTests: XCTestCase
{
    /// Directory containing slides.
    let _inputURL = URL(string: #file)!
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appending(path: "data")
        .appending(path: "in")
        .appending(path: "224x224")
    /// Directory to dump outputs from the tests.
    let _outputDir = NSTemporaryDirectory()
    
    /// Size of one image (height and width are the same).
    let _size = 224
    
    override func setUp()
    {
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
    }
    
    private func _buildModel(model: String) -> (Model, Input2D, Layer2D)
    {
        let context = ModelContext(name: "Image", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 3, width: _size, height: _size, params: params
        )
        
        switch model
        {
        case "ColorJitterHSV":
            layer = ColorJitterHSV(
                layerPrev: layer,
                rangeH: Range(min: 0.0, max: 10.0),
                rangeS: Range(min: 0.0, max: 0.1),
                rangeV: Range(min: 0.0, max: 0.1),
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        let model = Model(model: context.model, modelsPrev: [])
        model.initKernel()
        
        let firstLayer = model.layers.first as! Input2D
        let lastLayer = model.layers.last as! Layer2D
        
        return (model, firstLayer, lastLayer)
    }
    
    func testColorJitterHSVGPU()
    {
        let (model, firstLayer, lastLayer) = _buildModel(
            model: "ColorJitterHSV"
        )
        
        let buffer = MetalPrivateBuffer<Float>(
            3 * 3 * _size * _size, deviceID: 0
        )
        
        let imagesURL = [
            _inputURL.appending(path: "harp.png"),
            _inputURL.appending(path: "monastery.png"),
            _inputURL.appending(path: "snail.png")
        ]
        try! Image.loadImages(
            imagesURL: imagesURL,
            imagesBuffer: buffer,
            width: _size, height: _size
        )
        
        try! firstLayer.setDataGPU(buffer, batchSize: 3)
        model.updateKernel(batchSize: 3)
        try! model.forward()
        
        let pixels = Image.extractPixels(
            lastLayer.outs,
            width: _size, height: _size
        )
        for pixelsI in pixels
        {
            let image = Image.buildImage(
                pixels: pixelsI,
                width: _size, height: _size
            )
            image.save(
                url: URL(fileURLWithPath: _outputDir)
                    .appending(path: "test.png")
            )
        }
    }
}
