//
// Image.swift
// GrAIdient
//
// Created by Jean-François Reboud on 19/05/2023.
//

import Foundation
import AppKit

/// Error occuring when processing images.
public enum ImageError: Error
{
    /// Problem in the dimensions of the image.
    case UnexpectedSize
    /// Impossible to get access to the image's pixels.
    case UnexpectedDataStructure
    /// Not enough space in a buffer.
    case MissingSpace
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
        case .MissingSpace:
            return "Not enough space."
        }
    }
}

public class Image
{
    ///
    /// Load images into a buffer of pixels.
    ///
    /// Consider the input images are in the .RGB `ImageFormat` and
    /// the output buffer in the .Neuron format.
    ///
    /// - Parameters:
    ///     - imagesURL: Images on the disk.
    ///     - imagesBuffer: Buffer of images.
    ///     - width: Width of the images.
    ///     - height: Height of the images.
    ///
    public static func loadImages(
        imagesURL: [URL],
        imagesBuffer: FloatBuffer,
        width: Int,
        height: Int) throws
    {
        let batchSize = imagesURL.count
        if imagesBuffer.nbElems < batchSize * 3 * height * width
        {
            throw ImageError.MissingSpace
        }
        
        let shared = true
        _ = imagesBuffer.download(shared)
        
        var buffer = [Float](
            repeating: 0.0,
            count: batchSize * 3 * height * width
        )
        
        for (elem, imageURL) in imagesURL.enumerated()
        {
            let image = NSImage(contentsOfFile: imageURL.path)!
            let pixels = try image.extractPaddedPixels(
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
                    
                    buffer[offsetSet] =
                        Float(pixels[3 * offsetGet + depth]) / 255.0
                }
            }}
        }
        imagesBuffer.initialize(array: &buffer, shared: shared)
    }
    
    ///
    /// Get pixels out of buffer of images.
    ///
    /// Consider the input buffer is in the .Neuron `ImageFormat` and
    /// the output images are in the .RGB format.
    ///
    /// - Parameters:
    ///     - metalBuffer: Buffer of images.
    ///     - width: Width of the images.
    ///     - height: Height of the images.
    /// - Returns: The list of images as list of pixels.
    ///
    public static func extractPixels(
        _ metalBuffer: FloatBuffer,
        width: Int,
        height: Int) -> [[UInt8]]
    {
        let buffer = metalBuffer.download()
        let nbImages = metalBuffer.nbElems / (width * height * 3)
        
        var images = [[Float]]()
        for i in 0..<nbImages
        {
            images.append([Float](
                buffer[i * 3 * height * width..<(i+1) * 3 * height * width]
            ))
        }
        return toRGB(toPixel(images), width: width, height: height)
    }

    ///
    /// Convert float images (values in [0; 1]) to pixel images (values in [0; 255]).
    ///
    /// - Parameter images: List of images with values in [0; 1].
    /// - Returns: The list of images with values in [0; 255].
    ///
    public static func toPixel<T: BinaryFloatingPoint>(
        _ images: [[T]]) -> [[UInt8]]
    {
        var output = [[UInt8]]()
        for elem in 0..<images.count
        {
            output.append(images[elem].map 
            {
                let valTmp = $0 * T(255.0)
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
                return val
            })
        }
        return output
    }
    
    ///
    /// Convert pixel images (values in [0; 255]) to float images (values in [0; 1]).
    ///
    /// - Parameter images: List of images with values in [0; 255].
    /// - Returns: The list of images with values in [0; 1].
    ///
    public static func toFloat<T: BinaryFloatingPoint>(
        _ images: [[UInt8]]) -> [[T]]
    {
        var output = [[T]]()
        for elem in 0..<images.count
        {
            output.append(images[elem].map { T($0) / 255.0 })
        }
        return output
    }
    
    ///
    /// Organize pixels into the RGB format.
    ///
    /// Consider the input images are in the .Neuron `ImageFormat` format.
    ///
    /// - Parameters:
    ///     - images: List of images.
    ///     - width: Width of the images.
    ///     - height: Height of the images.
    /// - Returns: The list of images as list of pixels.
    ///
    public static func toRGB(
        _ images: [[UInt8]],
        width: Int,
        height: Int) -> [[UInt8]]
    {
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
    
    ///
    /// Organize pixels into the Neuron format.
    ///
    /// Consider the input images are in the .RGB `ImageFormat` format.
    ///
    /// - Parameters:
    ///     - images: List of images.
    ///     - width: Width of the images.
    ///     - height: Height of the images.
    /// - Returns: The list of images as list of pixels.
    ///
    public static func toNeuron(
        _ images: [[UInt8]],
        width: Int,
        height: Int) -> [[UInt8]]
    {
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
                
                let i = index / (3 * width)
                let j = (index % (3 * width)) / 3
                let depth = index % 3
                
                let offset = j + (depth * height + i) * width
                
                let val = images[elem][index]
                gridPtr[offset] = val
            }}
            output.append(grid)
        }
        return output
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
    public static func buildImage(
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
}

public extension NSImage
{
    ///
    /// Extract the underlying pixels in the image.
    /// Pad with zeros when needed.
    ///
    /// Throw an error when image is too big or it is
    /// impossible to retrieve the pixels.
    ///
    /// - Parameters:
    ///     - width: The expected output image width.
    ///     - height: The expected output image height.
    /// - Returns: An array of pixels.
    ///
    func extractPaddedPixels(
        width: CGFloat,
        height: CGFloat) throws -> [UInt8]
    {
        let pixelsIn = try extractPixels()
        
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
    
    ///
    /// Extract the underlying pixels in the image.
    ///
    /// Throw an error when it is impossible to retrieve the pixels.
    ///
    /// - Returns: An array of pixels.
    ///
    func extractPixels() throws -> [UInt8]
    {
        if let pixelData = (cgImage(
            forProposedRect: nil, context: nil, hints: nil)!).dataProvider?.data
        {
            let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
            
            var pixels = [UInt8]()
            for i in 0..<Int(size.height) {
            for j in 0..<Int(size.width)
            {
                let pos = CGPoint(x: j, y: i)
                
                let pixelInfo: Int = (Int(size.width) * Int(pos.y) * 4) +
                    Int(pos.x) * 4
                
                let r = data[pixelInfo]
                let g = data[pixelInfo + 1]
                let b = data[pixelInfo + 2]
                pixels += [r, g, b]
            }}
            return pixels
        }
        else
        {
            throw ImageError.UnexpectedDataStructure
        }
    }
    
    ///
    /// Save an image to the disk.
    ///
    /// Throw an error in the Cocoa domain, if there is an error writing to the `URL`.
    ///
    /// - Parameter url: The path where to dump the image.
    ///
    func save(url: URL) throws
    {
        if representations.count > 0 {
        if let imageData = tiffRepresentation
        {
            let rep = NSBitmapImageRep(data: imageData)!
            let pngData = rep.representation(
                using: NSBitmapImageRep.FileType.png,
                properties: [:]
            )!
            try pngData.write(to: url, options: [])
        }}
    }
}
