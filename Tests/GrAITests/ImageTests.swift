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
    let _referenceURL = URL(string: #file)!
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
        GrAI.Precision.float = true
    }
    
    private func _buildModel(
        modelName: String,
        parameters: Any) -> (Model, Input2D, Layer2D)
    {
        let context = ModelContext(name: "Image", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 3, width: _size, height: _size, params: params
        )
        
        switch modelName
        {
        case "Rotate":
            let p = parameters as! Double
            layer = try! Rotate2D(
                layerPrev: layer,
                anglesList: [p],
                padValue: 0.0,
                params: params
            )
            
        case "FlipHorizontal":
            let p = parameters as! Double
            layer = FlipHorizontal2D(
                layerPrev: layer,
                probability: p,
                params: params
            )
            
        case "FlipVertical":
            let p = parameters as! Double
            layer = FlipVertical2D(
                layerPrev: layer,
                probability: p,
                params: params
            )
            
        case "ColorJitterHSV":
            let p = parameters as! (Range<Double>, Range<Double>, Range<Double>)
            layer = try! ColorJitterHSV(
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
    
    private func _compareCPU(lastLayer: Layer2D, suffix: String)
    {
        for (elem, name) in _imageNames.enumerated()
        {
            let pixelsOut: [Float] = lastLayer.getOutsCPU(elem: elem)
            var pixels = Image.toPixel([pixelsOut])
            pixels = Image.toRGB(pixels, width: _size, height: _size)
            
            let image = Image.buildImage(
                pixels: pixels[0],
                width: _size, height: _size
            )
            try! image.save(
                url: URL(fileURLWithPath: _outputDir)
                    .appending(path: "\(name)_\(suffix).png")
            )
            
            let imageRef = NSImage(
                byReferencingFile: _referenceURL
                    .appending(path: "\(name)_\(suffix).png").path
            )!
            let pixelsRef = try! imageRef.extractPaddedPixels(
                width: CGFloat(_size), height: CGFloat(_size)
            )
            XCTAssert(pixels[0] == pixelsRef)
        }
    }
    
    private func _compareGPU(lastLayer: Layer2D, suffix: String)
    {
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
            try! image.save(
                url: URL(fileURLWithPath: _outputDir)
                    .appending(path: "\(_imageNames[elem])_\(suffix).png")
            )
            
            let imageRef = NSImage(
                byReferencingFile: _referenceURL
                    .appending(path: "\(_imageNames[elem])_\(suffix).png").path
            )!
            let pixelsRef = try! imageRef.extractPaddedPixels(
                width: CGFloat(_size), height: CGFloat(_size)
            )
            
            let test1 = pixelsBatch[elem] == pixelsRef
            if !test1
            {
                var nbFail = 0
                for (val1, val2) in zip(pixelsBatch[elem], pixelsRef)
                {
                    if val1 != val2
                    {
                        nbFail += 1
                    }
                }
                let ratioFail = Double(nbFail) / Double(pixelsRef.count) * 100.0
                let test2 = ratioFail < 0.1
                
                if !test2
                {
                    for (val1, val2) in zip(pixelsBatch[elem], pixelsRef)
                    {
                        if val1 != val2
                        {
                            let diff = abs(Double(val1) - Double(val2))
                            let test3 = diff <= 1
                            XCTAssert(test3)
                        }
                    }
                }
                else
                {
                    XCTAssert(test2)
                }
            }
            else
            {
                XCTAssert(test1)
            }
        }
    }
    
    private func _runRGBCPU(
        modelName: String,
        parameters: Any,
        suffix: String)
    {
        GrAI.Opti.CPU = true
        let (model, firstLayer, lastLayer) = _buildModel(
            modelName: modelName, parameters: parameters
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
        
        try! firstLayer.setDataCPU(
            data,
            batchSize: batchSize,
            nbChannels: 3, height: _size, width: _size,
            format: .RGB
        )
        model.updateKernel(batchSize: batchSize)
        try! model.forward()
        
        _compareCPU(lastLayer: lastLayer, suffix: suffix)
    }
    
    private func _runNeuronCPU(
        modelName: String,
        parameters: Any,
        suffix: String)
    {
        GrAI.Opti.CPU = true
        let (model, firstLayer, lastLayer) = _buildModel(
            modelName: modelName, parameters: parameters
        )
        
        let batchSize = imagesURL.count
        
        var data = [Double]()
        for imageURL in imagesURL
        {
            let image = NSImage(byReferencingFile: imageURL.path)!
            let pixels1 = try! image.extractPaddedPixels(
                width: CGFloat(_size), height: CGFloat(_size)
            )
            let pixels2 = Image.toNeuron(
                [pixels1], width: _size, height: _size
            )
            data += Image.toFloat(pixels2)[0]
        }
        
        try! firstLayer.setDataCPU(
            data,
            batchSize: batchSize,
            nbChannels: 3, height: _size, width: _size,
            format: .Neuron
        )
        model.updateKernel(batchSize: batchSize)
        try! model.forward()
        
        _compareCPU(lastLayer: lastLayer, suffix: suffix)
    }
    
    private func _runRGBGPU(
        modelName: String,
        parameters: Any,
        suffix: String)
    {
        let (model, firstLayer, lastLayer) = _buildModel(
            modelName: modelName, parameters: parameters
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
        
        try! firstLayer.setDataGPU(
            data,
            batchSize: batchSize,
            nbChannels: 3, height: _size, width: _size,
            format: .RGB
        )
        model.updateKernel(batchSize: batchSize)
        try! model.forward()
        
        _compareGPU(lastLayer: lastLayer, suffix: suffix)
    }
    
    private func _runNeuronGPU(
        modelName: String,
        parameters: Any,
        suffix: String)
    {
        let (model, firstLayer, lastLayer) = _buildModel(
            modelName: modelName, parameters: parameters
        )
        
        let batchSize = imagesURL.count
        
        var data = [Double]()
        for imageURL in imagesURL
        {
            let image = NSImage(byReferencingFile: imageURL.path)!
            let pixels1 = try! image.extractPaddedPixels(
                width: CGFloat(_size), height: CGFloat(_size)
            )
            let pixels2 = Image.toNeuron(
                [pixels1], width: _size, height: _size
            )
            data += Image.toFloat(pixels2)[0]
        }
        
        try! firstLayer.setDataGPU(
            data,
            batchSize: batchSize,
            nbChannels: 3, height: _size, width: _size,
            format: .Neuron
        )
        model.updateKernel(batchSize: batchSize)
        try! model.forward()
        
        _compareGPU(lastLayer: lastLayer, suffix: suffix)
    }
    
    private func _runBufferGPU(
        modelName: String,
        parameters: Any,
        suffix: String)
    {
        let (model, firstLayer, lastLayer) = _buildModel(
            modelName: modelName, parameters: parameters
        )
        
        let batchSize = imagesURL.count
        let buffer = FloatBuffer(nbElems: 
            batchSize * 3 * _size * _size, deviceID: 0
        )
        
        try! Image.loadImages(
            imagesURL: imagesURL,
            imagesBuffer: buffer,
            width: _size, height: _size
        )
        
        try! firstLayer.setDataGPU(
            buffer,
            batchSize: batchSize,
            nbChannels: 3, height: _size, width: _size
        )
        model.updateKernel(batchSize: batchSize)
        try! model.forward()
        
        _compareGPU(lastLayer: lastLayer, suffix: suffix)
    }
    
    func testRotate1CPU()
    {
        _runRGBCPU(
            modelName: "Rotate",
            parameters: 0.0,
            suffix: "cpu_rotate1"
        )
    }
    
    func testRotate1GPU()
    {
        _runNeuronGPU(
            modelName: "Rotate",
            parameters: 0.0,
            suffix: "gpu_rotate1"
        )
    }
    
    func testRotate2CPU()
    {
        _runNeuronCPU(
            modelName: "Rotate",
            parameters: 45.0,
            suffix: "cpu_rotate2"
        )
    }
    
    func testRotate2GPU()
    {
        _runRGBGPU(
            modelName: "Rotate",
            parameters: 45.0,
            suffix: "gpu_rotate2"
        )
    }
    
    func testRotate3CPU()
    {
        _runRGBCPU(
            modelName: "Rotate",
            parameters: -90.0,
            suffix: "cpu_rotate3"
        )
    }
    
    func testRotate3GPU()
    {
        _runBufferGPU(
            modelName: "Rotate",
            parameters: -90.0,
            suffix: "gpu_rotate3"
        )
    }
    
    func testFlipHorizontal1CPU()
    {
        _runRGBCPU(
            modelName: "FlipHorizontal",
            parameters: 0.0,
            suffix: "cpu_fliph1"
        )
    }
    
    func testFlipHorizontal1GPU()
    {
        _runNeuronGPU(
            modelName: "FlipHorizontal",
            parameters: 0.0,
            suffix: "gpu_fliph1"
        )
    }
    
    func testFlipHorizontal2CPU()
    {
        _runNeuronCPU(
            modelName: "FlipHorizontal",
            parameters: 1.0,
            suffix: "cpu_fliph2"
        )
    }
    
    func testFlipHorizontal2GPU()
    {
        _runRGBGPU(
            modelName: "FlipHorizontal",
            parameters: 1.0,
            suffix: "gpu_fliph2"
        )
    }
    
    func testFlipVertical1CPU()
    {
        _runRGBCPU(
            modelName: "FlipVertical",
            parameters: 0.0,
            suffix: "cpu_flipv1"
        )
    }
    
    func testFlipVertical1GPU()
    {
        _runNeuronGPU(
            modelName: "FlipVertical",
            parameters: 0.0,
            suffix: "gpu_flipv1"
        )
    }
    
    func testFlipVertical2CPU()
    {
        _runNeuronCPU(
            modelName: "FlipVertical",
            parameters: 1.0,
            suffix: "cpu_flipv2"
        )
    }
    
    func testFlipVertical2GPU()
    {
        _runBufferGPU(
            modelName: "FlipVertical",
            parameters: 1.0,
            suffix: "gpu_flipv2"
        )
    }
    
    func testColorJitterHSV1CPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        let parameters = (rangeH, rangeS, rangeV)
        _runRGBCPU(
            modelName: "ColorJitterHSV",
            parameters: parameters,
            suffix: "cpu_hsv1"
        )
    }
    
    func testColorJitterHSV1GPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        let parameters = (rangeH, rangeS, rangeV)
        _runNeuronGPU(
            modelName: "ColorJitterHSV",
            parameters: parameters,
            suffix: "gpu_hsv1"
        )
    }
    
    func testColorJitterHSV2CPU()
    {
        let rangeH = try! Range<Double>(min: 50.0, max: 50.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        let parameters = (rangeH, rangeS, rangeV)
        _runNeuronCPU(
            modelName: "ColorJitterHSV",
            parameters: parameters,
            suffix: "cpu_hsv2"
        )
    }
    
    func testColorJitterHSV2GPU()
    {
        let rangeH = try! Range<Double>(min: 50.0, max: 50.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        let parameters = (rangeH, rangeS, rangeV)
        _runRGBGPU(
            modelName: "ColorJitterHSV",
            parameters: parameters,
            suffix: "gpu_hsv2"
        )
    }
    
    func testColorJitterHSV3CPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.5, max: 0.5)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        let parameters = (rangeH, rangeS, rangeV)
        _runRGBCPU(
            modelName: "ColorJitterHSV",
            parameters: parameters,
            suffix: "cpu_hsv3"
        )
    }
    
    func testColorJitterHSV3GPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.5, max: 0.5)
        let rangeV = try! Range<Double>(min: 0.0, max: 0.0)
        let parameters = (rangeH, rangeS, rangeV)
        _runBufferGPU(
            modelName: "ColorJitterHSV",
            parameters: parameters,
            suffix: "gpu_hsv3"
        )
    }
    
    func testColorJitterHSV4CPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.5, max: 0.5)
        let parameters = (rangeH, rangeS, rangeV)
        _runRGBCPU(
            modelName: "ColorJitterHSV",
            parameters: parameters,
            suffix: "cpu_hsv4"
        )
    }
    
    func testColorJitterHSV4GPU()
    {
        let rangeH = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeS = try! Range<Double>(min: 0.0, max: 0.0)
        let rangeV = try! Range<Double>(min: 0.5, max: 0.5)
        let parameters = (rangeH, rangeS, rangeV)
        _runBufferGPU(
            modelName: "ColorJitterHSV",
            parameters: parameters,
            suffix: "gpu_hsv4"
        )
    }
}
