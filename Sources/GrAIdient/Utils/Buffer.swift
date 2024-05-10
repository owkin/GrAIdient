//
// Buffer.swift
// GrAIdient
//
// Created by Jean-François Reboud on 08/05/2024.
//

import Foundation
import Accelerate

///
/// Copy, convert and upload Float array to Half buffer.
///
/// - Parameters:
///     - array: Input array.
///     - out: Output buffer.
///     - start: Start index in `array`.
///     - nbElems: Number of elements to copy.
///     - deviceID: GPU device.
///
public func setupHalfBuffer(
    array: inout [Float],
    out: MetalBuffer<UInt16>,
    start: Int,
    nbElems: Int,
    deviceID: Int)
{
    let temp = MetalSharedBuffer<Float>(nbElems, deviceID: deviceID)
    copyArrayToBuffer(
        array: &array,
        buffer: temp.buffer,
        start: start,
        nbElems: nbElems
    )
    
    temp.upload()
    convertFloat2Half(
        inBuffer: temp,
        outBuffer: out,
        nbElems: nbElems,
        deviceID: deviceID
    )
    
    // Make sure operation has ended because returning.
    _ = out.download()
}

///
/// Convert Half buffer to Float buffer and download content.
///
/// - Parameter buffer: Input buffer.
///
/// - Returns: Float buffer.
///
public func getHalfBuffer(
    _ buffer: MetalBuffer<UInt16>
) -> MetalSharedBuffer<Float>
{
    let temp = MetalSharedBuffer<Float>(
        buffer.nbElems,
        deviceID: buffer.deviceID
    )
    convertHalf2Float(
        inBuffer: buffer,
        outBuffer: temp,
        nbElems: buffer.nbElems,
        deviceID: buffer.deviceID
    )
    
    _ = temp.download()
    return temp
}

///
/// Convert a Float32 buffer into a Float16 buffer.
///
/// - Parameters:
///     - inBuffer: Input buffer.
///     - outBuffer: Output buffer.
///     - nbElems: Number of elements.
///     - deviceID: GPU device.
///
public func convertFloat2Half(
    inBuffer: MetalBuffer<Float>,
    outBuffer: MetalBuffer<UInt16>,
    nbElems: Int,
    deviceID: Int)
{
    let pNbElems: [UInt32] = [UInt32(nbElems)]
    
    let command = MetalKernel.get.createCommand(
        "convertFloat2Half", deviceID: deviceID
    )
    command.setBuffer(inBuffer.metal, atIndex: 0)
    command.setBytes(pNbElems, atIndex: 1)
    command.setBuffer(outBuffer.metal, atIndex: 2)
    
    command.dispatchThreads(nbElems)
    command.enqueue()
}

///
/// Convert a Float16 into a Float32 buffer.
///
/// - Parameters:
///     - inBuffer: Input buffer.
///     - outBuffer: Output buffer.
///     - nbElems: Number of elements.
///     - deviceID: GPU device.
///
public func convertHalf2Float(
    inBuffer: MetalBuffer<UInt16>,
    outBuffer: MetalBuffer<Float>,
    nbElems: Int,
    deviceID: Int)
{
    let pNbElems: [UInt32] = [UInt32(nbElems)]
    
    let command = MetalKernel.get.createCommand(
        "convertHalf2Float", deviceID: deviceID
    )
    command.setBuffer(inBuffer.metal, atIndex: 0)
    command.setBytes(pNbElems, atIndex: 1)
    command.setBuffer(outBuffer.metal, atIndex: 2)
    
    command.dispatchThreads(nbElems)
    command.enqueue()
}

///
/// Copy array to buffer.
///
/// - Parameters:
///     - array: Input array.
///     - buffer: Output buffer.
///     - start: Start index in `array`.
///     - nbElems: Number of elements to copy.
///
public func copyArrayToBuffer<T: BNNSScalar>(
    array: inout [T],
    buffer: UnsafeMutableBufferPointer<T>,
    start: Int,
    nbElems: Int)
{
    var dest = BNNSNDArrayDescriptor(
        data: buffer,
        shape: .vector(nbElems)
    )!
    
    array.withUnsafeMutableBufferPointer
    {
        ptr in
        
        let base = ptr.baseAddress
        let offset = base?.advanced(by: start)
        let bufferPtr = UnsafeMutableBufferPointer<T>(
            start: offset, count: nbElems
        )
        
        var src = BNNSNDArrayDescriptor(
            data: bufferPtr,
            shape: .vector(nbElems)
        )!
        
        BNNSCopy(&dest, &src, nil)
    }
}
