//
// Utils.swift
// MAExamples
//
// Created by Jean-François Reboud on 19/10/2022.
//

import Foundation
import Cocoa
import MAKit

/// Python library default path.
let PYTHON_LIB =
    FileManager.default.homeDirectoryForCurrentUser.path +
    "/opt/anaconda3/envs/maexamples/lib/libpython3.9.dylib"

/// Set the Python library path.
func setPythonLib()
{
    if ProcessInfo.processInfo.environment["PYTHON_LIBRARY"] == nil
    {
        let task = Process()
        task.launchPath = "/usr/bin/which"
        task.arguments = ["python"]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        task.launch()
        task.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: String.Encoding.utf8)!
        
        if output.count > 0
        {
            var url = URL(fileURLWithPath: output)
            url = url.deletingLastPathComponent().deletingLastPathComponent()
            url = url.appendingPathComponent("lib")
            url = url.appendingPathComponent("libpython3.7m.dylib")
            setenv("PYTHON_LIBRARY", url.path, 1)
        }
        else
        {
            setenv("PYTHON_LIBRARY", PYTHON_LIB, 1)
        }
    }
}

///
/// Get pixels out of buffer of images.
///
/// Consider the buffer of images is in the .Neuron `ImageFormat`.
///
/// - Parameters:
///     - metalBuffer: Buffer of images.
///     - width: Width of the images.
///     - height: Height of the images.
/// - Returns: The list of images as list of pixels.
///
func getPixels(
    _ metalBuffer: MetalBuffer<Float>,
    width: Int,
    height: Int) -> [[UInt8]]
{
    let bufferPtr: UnsafeMutableBufferPointer<Float>
    if let sBuffer = metalBuffer as? MetalSharedBuffer<Float>
    {
        MetalKernel.get.download([sBuffer])
        bufferPtr = sBuffer.buffer
    }
    else if let pBuffer = metalBuffer as? MetalPrivateBuffer<Float>
    {
        MetalKernel.get.download([pBuffer])
        bufferPtr = pBuffer.shared.buffer
    }
    else
    {
        fatalError()
    }
    
    let nbImages = metalBuffer.nbElems / (width * height * 3)
    
    var output = [[UInt8]]()
    for elem in 0..<nbImages
    {
        var grid: [UInt8] = [UInt8](repeating: 0, count: width * height * 3)
        grid.withUnsafeMutableBufferPointer { gridPtr in
        Concurrency.slice(gridPtr.count)
        {
            (index: Int) in
            
            let depth = index / (width * height)
            let i = (index - depth * width * height) / width
            let j = (index - depth * width * height) % width
            
            let offsetGet = elem * 3 * height * width
            let offsetSet = j + i * width
            
            let valTmp = bufferPtr[index + offsetGet] * 255.0
            let val: UInt8
            if valTmp < 0
            {
                val = 0
            }
            else if valTmp > 255.0
            {
                val = 255
            }
            else
            {
                val = UInt8(valTmp)
            }
                
            gridPtr[3 * offsetSet + depth] = val
        }}
        output.append(grid)
    }
    return output
}

///
/// Get pixels out of images.
///
/// - Parameters:
///     - images: List of images.
///     - width: Width of the images.
///     - height: Height of the images.
///     - imageFormat: The image format.
/// - Returns: The list of images as list of pixels.
///
func getPixels(
    _ images: [[UInt8]],
    width: Int,
    height: Int,
    imageFormat: ImageFormat) -> [[UInt8]]
{
    switch imageFormat
    {
    case .RGB:
        return images
        
    case .Neuron:
        var output = [[UInt8]]()
        for elem in 0..<images.count
        {
            var grid: [UInt8] = [UInt8](repeating: 0, count: width * height * 3)
            grid.withUnsafeMutableBufferPointer { gridPtr in
            Concurrency.slice(gridPtr.count)
            {
                (index: Int) in
                
                let depth = index / (width * height)
                let i = (index - depth * width * height) / width
                let j = (index - depth * width * height) % width
                let offset = j + i * width
                
                let val = images[elem][index]
                gridPtr[3 * offset + depth] = val
            }}
            output.append(grid)
        }
        return output
    }
}

///
/// Get an image out of pixels.
///
/// This function takes as input the result of the `getPixels` function.
/// 
/// - Parameters:
///     - pixels: List of pixels.
///     - width: Width of the image.
///     - height: Height of the image.
/// - Returns: The image built.
///
func getImage(
    pixels: [UInt8],
    width: Int,
    height: Int) -> NSImage
{
    let bitsPerComponent = 8
    let bitsPerPixel = 24
    
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
    
    if (pixels.count != Int(3 * width * height))
    {
        fatalError()
    }
    
    var data = pixels // Copy to mutable []
    let providerRef = CGDataProvider(
        data: Data(
            bytes: &data,
            count: data.count * MemoryLayout<UInt8>.size
        ) as CFData
    )
    
    let cgImage = CGImage(
        width: width,
        height: height,
        bitsPerComponent: bitsPerComponent,
        bitsPerPixel: bitsPerPixel,
        bytesPerRow: 3 * width * MemoryLayout<UInt8>.size,
        space: rgbColorSpace,
        bitmapInfo: bitmapInfo,
        provider: providerRef!,
        decode: nil,
        shouldInterpolate: true,
        intent: CGColorRenderingIntent.defaultIntent
    )!
    return NSImage(cgImage: cgImage, size: NSZeroSize)
}

///
/// Save an image to the disk.
///
/// - Parameters:
///     - image: The image to save to the disk.
///     - url: The path where to dump the image.
///
func saveImage(_ image: NSImage, url: URL)
{
    if image.representations.count > 0 {
    if let imageData = image.tiffRepresentation
    {
        let rep = NSBitmapImageRep(data: imageData)!
        let pngData = rep.representation(
            using: NSBitmapImageRep.FileType.png,
            properties: [:]
        )!
        try! pngData.write(to: url, options: [])
    }}
}
