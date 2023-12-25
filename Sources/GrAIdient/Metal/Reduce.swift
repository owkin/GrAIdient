//
// Reduce.swift
// GrAIdient
//
// Created by Jean-François Reboud on 17/05/2023.
//

import MetalKit

/// Default number of threads per threadgroup on the GPU.
private let THREADS_PER_THREADGROUP = 64

///
/// Compute the number of threadgroups resulting in the reduction of `nbElems` on the GPU.
///
/// - Parameters:
///     - nbElems: The number of elements to reduce.
///     - threadsPerThreadroup: The number of threads per thread group.
/// - Returns: The number of resulting threadgroups.
///
private func getNbThreadgroups(
    nbElems: Int,
    threadsPerThreadgroup: Int) -> Int
{
    let value = Double(nbElems) / Double(threadsPerThreadgroup)
    return Int(ceil(value))
}

///
/// Reduce (sum) the elements in the `dim1` dimension.
///
/// - Parameters:
///     - inBuffer: The input buffer.
///     - outBuffer: The final buffer (also used for writing intermediate results).
///     - dim1: The dimension of the elements to reduce.
///     - dim2: A dimension for elements we do not want to reduce.
///
public func reduceSum(
    inBuffer: MTLBuffer,
    outBuffer: MTLBuffer,
    dim1: Int,
    dim2: Int,
    deviceID: Int)
{
    var nbElems = dim1
    var command: MetalEncoder
    
    while nbElems > 1
    {
        let pNbDimensions: [UInt32] = [UInt32(nbElems), UInt32(dim2)]
        
        // Reduce thanks to thread group shared memory.
        if nbElems > 100
        {
            let nbThreadgroups = getNbThreadgroups(
                nbElems: nbElems,
                threadsPerThreadgroup: THREADS_PER_THREADGROUP
            )
            let pNbThreadgroups: [UInt32] = [UInt32(nbThreadgroups)]
            
            command = MetalKernel.get.createEncoder(
                "reduceSum64", deviceID: deviceID
            )
            command.setBuffer(inBuffer, atIndex: 0)
            command.setBytes(pNbDimensions, atIndex: 1)
            command.setBytes(pNbThreadgroups, atIndex: 2)
            command.setBuffer(outBuffer, atIndex: 3)
            
            let threadsPerThreadgroup = MTLSizeMake(
                THREADS_PER_THREADGROUP, 1, 1
            )
            let threadsPerGrid = MTLSizeMake(
                nbElems, dim2, 1
            )
            command.dispatchThreads(
                threadsPerGrid: threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            
            nbElems = nbThreadgroups
        }
        
        // Simple reduce.
        else
        {
            command = MetalKernel.get.createEncoder(
                "reduceSum", deviceID: deviceID
            )
            command.setBuffer(inBuffer, atIndex: 0)
            command.setBytes(pNbDimensions, atIndex: 1)
            command.setBuffer(outBuffer, atIndex: 2)
            
            command.dispatchThreads(dim2)
            
            nbElems = 1
        }
        command.endEncoding()
    }
}

///
/// Reduce (max) the elements in the `dim1` dimension.
///
/// - Parameters:
///     - inBuffer: The input buffer.
///     - outBuffer: The final buffer (also used for writing intermediate results).
///     - dim1: The dimension of the elements to reduce.
///     - dim2: A dimension for elements we do not want to reduce.
///
public func reduceMax(
    inBuffer: MTLBuffer,
    outBuffer: MTLBuffer,
    dim1: Int,
    dim2: Int,
    deviceID: Int)
{
    var nbElems = dim1
    var command: MetalEncoder
    
    while nbElems > 1
    {
        let pNbDimensions: [UInt32] = [UInt32(nbElems), UInt32(dim2)]
        
        // Reduce thanks to thread group shared memory.
        if nbElems > 100
        {
            let nbThreadgroups = getNbThreadgroups(
                nbElems: nbElems,
                threadsPerThreadgroup: THREADS_PER_THREADGROUP
            )
            let pNbThreadgroups: [UInt32] = [UInt32(nbThreadgroups)]
            
            command = MetalKernel.get.createEncoder(
                "reduceMax64", deviceID: deviceID
            )
            command.setBuffer(inBuffer, atIndex: 0)
            command.setBytes(pNbDimensions, atIndex: 1)
            command.setBytes(pNbThreadgroups, atIndex: 2)
            command.setBuffer(outBuffer, atIndex: 3)
            
            let threadsPerThreadgroup = MTLSizeMake(
                THREADS_PER_THREADGROUP, 1, 1
            )
            let threadsPerGrid = MTLSizeMake(
                nbElems, dim2, 1
            )
            command.dispatchThreads(
                threadsPerGrid: threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
            
            nbElems = nbThreadgroups
        }
        
        // Simple reduce.
        else
        {
            command = MetalKernel.get.createEncoder(
                "reduceMax", deviceID: deviceID
            )
            command.setBuffer(inBuffer, atIndex: 0)
            command.setBytes(pNbDimensions, atIndex: 1)
            command.setBuffer(outBuffer, atIndex: 2)
            
            command.dispatchThreads(dim2)
            
            nbElems = 1
        }
        command.endEncoding()
    }
}
