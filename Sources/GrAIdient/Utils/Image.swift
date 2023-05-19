//
// Image.swift
// GrAIdient
//
// Created by Jean-François Reboud on 19/05/2023.
//

import Foundation
import Cocoa

public enum ImageError: Error
{
    case UnexpectedSize
    case UnexpectedDataStructure
}

extension ImageError: CustomStringConvertible
{
    public var description: String
    {
        switch self
        {
        case .UnexpectedSize:
            return "Size is too big."
        case .UnexpectedDataStructure:
            return "Cannot extract pixels."
        }
    }
}

public class Image
{
    public static func loadImages(
        imagesURL: [URL],
        imagesBuffer: MetalBuffer<Float>,
        width: Int,
        height: Int) throws
    {
        let batchSize = imagesURL.count
        if imagesBuffer.nbElems < batchSize * 3 * height * width
        {
            fatalError("`imagesBuffer` has not enough space.")
        }
        
        let bufferPtr = imagesBuffer.download()
        for (elem, imageURL) in imagesURL.enumerated()
        {
            let image = NSImage(byReferencing: imageURL)
            let pixelsData = try image.getPaddedPixels(
                width: CGFloat(width), height: CGFloat(height)
            )
            
            for i in 0..<height {
            for j in 0..<width
            {
                let offsetGet: Int = i * width + j
                
                for depth in 0..<3
                {
                    let offsetStart = (depth + 3 * elem) * height
                    let offsetSet = j + (offsetStart + i) * width
                    
                    bufferPtr[offsetSet] =
                        Float(pixelsData[3 * offsetGet + depth]) / 255.0
                }
            }}
        }
        imagesBuffer.upload()
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
    public static func getPixels(
        _ metalBuffer: MetalBuffer<Float>,
        width: Int,
        height: Int) -> [[UInt8]]
    {
        let bufferPtr = metalBuffer.download()
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
    public static func getPixels(
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
                var grid: [UInt8] = [UInt8](
                    repeating: 0,
                    count: width * height * 3
                )
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
    public static func getImage(
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
    public static func saveImage(_ image: NSImage, url: URL)
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
}

public extension NSImage
{
    func getPaddedPixels(
        width: CGFloat,
        height: CGFloat) throws -> [UInt8]
    {
        let pixelsIn = try getPixels()
        
        if self.size.height == height &&
           self.size.width == width
        {
            return pixelsIn
        }
        else if self.size.height < height ||
                self.size.width < width
        {
            var pixelsOut = [UInt8](
                repeating: 0, count: Int(3 * width * height)
            )
            let offsetIn = 3 * Int(size.width)
            let offsetOut = 3 * Int(width)
            for i in 0..<Int(size.height)
            {
                pixelsOut[i*offsetOut..<i*offsetOut+offsetIn] =
                    pixelsIn[i*offsetIn..<(i+1)*offsetIn]
            }
            return pixelsOut
        }
        else
        {
            throw ImageError.UnexpectedSize
        }
    }
    
    func getPixels() throws -> [UInt8]
    {
        if let imageData = tiffRepresentation,
           let imageRep = NSBitmapImageRep(data: imageData),
           let dataPtr = imageRep.bitmapData
        {
            let bufferPtr = UnsafeBufferPointer(
                start: dataPtr,
                count: Int(3 * size.height * size.width)
            )
            return [UInt8](bufferPtr)
        }
        else
        {
            throw ImageError.UnexpectedDataStructure
        }
    }
}
