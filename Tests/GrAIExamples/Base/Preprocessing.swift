//
// Preprocessing.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 18/10/2022.
//

import GrAIdient

///
/// Transform data images.
///
/// Transformation is: (data - mean) / std
///
/// - Parameters:
///     - data: The data to transform.
///     - height: The height of the images.
///     - width: The width of the images.
///     - mean: The mean of the transformation.
///     - std: The deviation of the transformation.
///     - imageFormat: The image format.
/// - Returns: The transformed data.
///
func preprocess<T: BinaryFloatingPoint>(
    _ data: [[UInt8]],
    height: Int,
    width: Int,
    mean: (T, T, T),
    std: (T, T, T),
    imageFormat: ImageFormat) -> [T]
{
    let dataTmp1 = data.reduce([], +)
    let dataTmp2 = dataTmp1.map { T($0) }
    return preprocess(
        dataTmp2,
        batchSize: data.count,
        height: height,
        width: width,
        mean: mean,
        std: std,
        imageFormat: imageFormat
    )
}

///
/// Transform data images.
///
/// Transformation is: (data - mean) / std
///
/// - Parameters:
///     - data: The data to transform.
///     - height: The height of the images.
///     - width: The width of the images.
///     - mean: The mean of the transformation.
///     - std: The deviation of the transformation.
///     - imageFormat: The image format.
/// - Returns: The transformed data.
///
func preprocess<T: BinaryFloatingPoint>(
    _ data: [[T]],
    height: Int,
    width: Int,
    mean: (T, T, T),
    std: (T, T, T),
    imageFormat: ImageFormat) -> [T]
{
    let dataTmp = data.reduce([], +)
    return preprocess(
        dataTmp,
        batchSize: data.count,
        height: height,
        width: width,
        mean: mean,
        std: std,
        imageFormat: imageFormat
    )
}

///
/// Transform data images.
///
/// Transformation is: (data - mean) / std
///
/// - Parameters:
///     - data: The data to transform.
///     - batchSize: The batch size of data.
///     - height: The height of the images.
///     - width: The width of the images.
///     - mean: The mean of the transformation.
///     - std: The deviation of the transformation.
///     - imageFormat: The image format.
/// - Returns: The transformed data.
///
func preprocess<T: BinaryFloatingPoint>(
    _ data: [T],
    batchSize: Int,
    height: Int,
    width: Int,
    mean: (T, T, T),
    std: (T, T, T),
    imageFormat: ImageFormat) -> [T]
{
    let nbChannels = data.count / (batchSize * height * width)
    if nbChannels > 3
    {
        fatalError("`nbChannels` should be 1, 2 or 3.")
    }
    var newData = [T](repeating: T(0.0), count: data.count)
    
    switch imageFormat
    {
    case .RGB:
        for elem in 0..<batchSize {
        for i in 0..<height {
        for j in 0..<width
        {
            let offset = j + (elem * height + i) * width
            
            for channel in 0..<nbChannels
            {
                let value = data[nbChannels * offset + channel]
                switch channel
                {
                case 0:
                    newData[nbChannels * offset + channel] =
                        (value - mean.0) / std.0
                case 1:
                    newData[nbChannels * offset + channel] =
                        (value - mean.1) / std.1
                case 2:
                    newData[nbChannels * offset + channel] =
                        (value - mean.2) / std.2
                default:
                    fatalError("Unreachable.")
                }
            }
        }}}
    case .Neuron:
        for elem in 0..<batchSize {
        for i in 0..<height {
        for j in 0..<width
        {
            for channel in 0..<nbChannels
            {
                let offsetStart = (channel + elem * nbChannels) * height
                let offset = j + (offsetStart + i) * width
        
                let value = data[offset]
                switch channel
                {
                case 0:
                    newData[offset] = (value - mean.0) / std.0
                case 1:
                    newData[offset] = (value - mean.1) / std.1
                case 2:
                    newData[offset] = (value - mean.2) / std.2
                default:
                    fatalError("Unreachable.")
                }
            }
        }}}
    }
    return newData
}
