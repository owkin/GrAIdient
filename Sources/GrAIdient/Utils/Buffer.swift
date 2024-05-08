//
// Buffer.swift
// GrAIdient
//
// Created by Jean-François Reboud on 08/05/2024.
//

import Foundation
import Accelerate

///
/// Convert Float32 buffer into a Float16 buffer.
///
/// - Parameters:
///     - inBuffer: input buffer.
///     - outBuffer: output buffer.
///     - nbElems: number of elements.
///     - deviceID: GPU device.
///
public func convertFloat2Half(
    inBuffer: MetalBuffer<Float>,
    outBuffer: MetalBuffer<Float16>,
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
/// Copy array to buffer.
///
/// - Parameters:
///     - array: input array.
///     - buffer: output buffer.
///     - start: start index in `array`.
///     - nbElems: Number of elements to copy.
///
func copyFloat16ArrayToBuffer(
    array: inout [Float16],
    buffer: UnsafeMutableBufferPointer<Float16>,
    start: Int,
    nbElems: Int)
{
    let elemSize = MemoryLayout<Float16>.stride

    array.withUnsafeBytes 
    {
        (srcBuffer: UnsafeRawBufferPointer) in

        let destPtr = UnsafeMutableRawPointer(buffer.baseAddress)

        let srcBase = srcBuffer.baseAddress
        let srcPtr = srcBase?.advanced(by: start)
        
        memmove(destPtr, srcPtr, elemSize * nbElems)
    }
}

///
/// Copy array to buffer.
///
/// - Parameters:
///     - array: input array.
///     - buffer: output buffer.
///     - start: start index in `array`.
///     - nbElems: Number of elements to copy.
///
func copyArrayToBuffer<T: BNNSScalar>(
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
