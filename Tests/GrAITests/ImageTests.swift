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
    /// Directory containing input images.
    let _inputURL = URL(string: #file)!
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appending(path: "data")
        .appending(path: "in")
        .appending(path: "224x224")
    /// Directory containing rerence images.
    let _outputURL = URL(string: #file)!
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appending(path: "data")
        .appending(path: "out")
        .appending(path: "augmentation")
    
    /// Input images.
    let _imageNames = [
        "harp",
        "monastery",
        "snail"
    ]
    
    /// Directory to dump outputs from the tests.
    let _outputDir = NSTemporaryDirectory()
    
    /// Size of one image (height and width are the same).
    let _size = 224
    
    var imagesURL: [URL]
    {
        get {
            return [URL](repeating: _inputURL, count: 3).enumerated().map
            {
                (index, element) in
                return element.appending(path: "\(_imageNames[index]).png")
            }
        }
    }
    
    override func setUp()
    {
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
    }
    
    private func _buildModel(
        model: String,
        parameters: Any) -> (Model, Input2D, Layer2D)
    {
        let context = ModelContext(name: "Image", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 3, width: _size, height: _size, params: params
        )
        
        switch model
        {
        case "ColorJitterHSV":
            let p = parameters as! (Range<Double>, Range<Double>, Range<Double>)
            layer = ColorJitterHSV(
                layerPrev: layer,
                rangeH: p.0,
                rangeS: p.1,
                rangeV: p.2,
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
    
    private func _colorJitterHSVCPU(
        rangeH: Range<Double>,
        rangeS: Range<Double>,
        rangeV: Range<Double>,
        suffix: String)
    {
        GrAI.Opti.CPU = true
        let (model, firstLayer, lastLayer) = _buildModel(
            model: "ColorJitterHSV", parameters: (rangeH, rangeS, rangeV)
        )
        
        let batchSize = imagesURL.count
        
        var data = [Double]()
        for imageURL in imagesURL
        {
            let image = NSImage(byReferencingFile: imageURL.path)!
            let pixels = try! image.extractPaddedPixels(
                width: CGFloat(_size), height: CGFloat(_size)
            )
            data += Image.toFloat([pixels])[0]
        }
        
        try! firstLayer.setDataCPU(data, batchSize: batchSize, format: .RGB)
        model.updateKernel(batchSize: batchSize)
        try! model.forward()
        
        for (elem, name) in _imageNames.enumerated()
        {
            let pixelsOut: [Float] = lastLayer.getOutsCPU(elem: elem)
            var pixels = Image.toPixel([pixelsOut])
            pixels = Image.toRGB(pixels, width: _size, height: _size)
            
            let image = Image.buildImage(
                pixels: pixels[0],
                width: _size, height: _size
            )
            image.save(
                url: URL(fileURLWithPath: _outputDir)
                    .appending(path: "\(name)_\(suffix).png")
            )
        }
    }
    
    private func _colorJitterHSVGPU(
        rangeH: Range<Double>,
        rangeS: Range<Double>,
        rangeV: Range<Double>,
        suffix: String)
    {
        let (model, firstLayer, lastLayer) = _buildModel(
            model: "ColorJitterHSV", parameters: (rangeH, rangeS, rangeV)
        )
        
        let batchSize = imagesURL.count
        let buffer = MetalPrivateBuffer<Float>(
            batchSize * 3 * _size * _size, deviceID: 0
        )
        
        try! Image.loadImages(
            imagesURL: imagesURL,
            imagesBuffer: buffer,
            width: _size, height: _size
        )
        
        try! firstLayer.setDataGPU(buffer, batchSize: batchSize)
        model.updateKernel(batchSize: batchSize)
        try! model.forward()
        
        let pixelsBatch = Image.extractPixels(
            lastLayer.outs,
            width: _size, height: _size
        )
        for (elem, pixels) in pixelsBatch.enumerated()
        {
            let image = Image.buildImage(
                pixels: pixels,
                width: _size, height: _size
            )
            image.save(
                url: URL(fileURLWithPath: _outputDir)
                    .appending(path: "\(_imageNames[elem])_\(suffix).png")
            )
        }
    }
    
    func testColorJitterHSV1CPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        _colorJitterHSVCPU(
            rangeH: rangeH,
            rangeS: rangeS,
            rangeV: rangeV,
            suffix: "cpu_hsv1"
        )
    }
    
    func testColorJitterHSV1GPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        _colorJitterHSVGPU(
            rangeH: rangeH,
            rangeS: rangeS,
            rangeV: rangeV,
            suffix: "gpu_hsv1"
        )
    }
    
    func testColorJitterHSV2CPU()
    {
        let rangeH = try! Range<Double>(min: 50.0, max: 50.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        _colorJitterHSVCPU(
            rangeH: rangeH,
            rangeS: rangeS,
            rangeV: rangeV,
            suffix: "cpu_hsv2"
        )
    }
    
    func testColorJitterHSV2GPU()
    {
        let rangeH = try! Range<Double>(min: 50.0, max: 50.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        _colorJitterHSVGPU(
            rangeH: rangeH,
            rangeS: rangeS,
            rangeV: rangeV,
            suffix: "gpu_hsv2"
        )
    }
    
    func testColorJitterHSV3CPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.5, max: 0.5)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        _colorJitterHSVCPU(
            rangeH: rangeH,
            rangeS: rangeS,
            rangeV: rangeV,
            suffix: "cpu_hsv3"
        )
    }
    
    func testColorJitterHSV3GPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.5, max: 0.5)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        _colorJitterHSVGPU(
            rangeH: rangeH,
            rangeS: rangeS,
            rangeV: rangeV,
            suffix: "gpu_hsv3"
        )
    }
    
    func testColorJitterHSV4CPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.5, max: 0.5)
        _colorJitterHSVCPU(
            rangeH: rangeH,
            rangeS: rangeS,
            rangeV: rangeV,
            suffix: "cpu_hsv4"
        )
    }
    
    func testColorJitterHSV4GPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.5, max: 0.5)
        _colorJitterHSVGPU(
            rangeH: rangeH,
            rangeS: rangeS,
            rangeV: rangeV,
            suffix: "gpu_hsv4"
        )
    }
}
