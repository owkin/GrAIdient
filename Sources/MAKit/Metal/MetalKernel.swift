//
// MetalKernel.swift
// MAKit
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Foundation

import Metal

public class MetalKernel
{
    static private var _metalKernel: MetalKernel! = nil
    fileprivate let _context: MetalListDevices
    
    public static var get: MetalKernel
    {
        get {
            if _metalKernel == nil
            {
                var listDevices = [MTLDevice]()
                
                let gpuNames = MAKit.Opti.gpuNamedPriority
                if gpuNames.count > 0
                {
                    for gpuName in gpuNames
                    {
                        var foundDevice = false
                        for device in MTLCopyAllDevices()
                        {
                            if device.name == gpuName
                            {
                                listDevices.append(device)
                                foundDevice = true
                                break
                            }
                        }
                        if !foundDevice
                        {
                            fatalError("Did not find device: \(gpuName).")
                        }
                    }
                }
                else
                {
                    // Get access to an external GPU
                    for device in MTLCopyAllDevices()
                    {
                        if device.isRemovable
                        {
                            listDevices.append(device)
                        }
                    }
                }
                    
                if let device = MTLCreateSystemDefaultDevice()
                {
                    var foundDevice = false
                    for deviceTmp in listDevices
                    {
                        if deviceTmp === device
                        {
                            foundDevice = true
                        }
                    }
                    if !foundDevice
                    {
                        listDevices.append(device)
                    }
                }
                
                _metalKernel = MetalKernel(listDevices)
            }
            return _metalKernel
        }
    }
    
    private init(_ listDevices: [MTLDevice])
    {
        _context = MetalListDevices(listDevices)
    }
    
    public func buildKernels(
        libraryContent: String,
        kernelNames: [String],
        optimalThreadGroupSize: Bool = false)
    {
        _context.buildKernels(
            libraryContent: libraryContent,
            kernelNames: kernelNames,
            optimalThreadGroupSize: optimalThreadGroupSize
        )
    }
    
    public func buildKernels(
        libraryURL: URL,
        kernelNames: [String],
        optimalThreadGroupSize: Bool = false)
    {
        _context.buildKernels(
            libraryURL: libraryURL,
            kernelNames: kernelNames,
            optimalThreadGroupSize: optimalThreadGroupSize
        )
    }
    
    public func getDevice(deviceID: Int) -> MTLDevice
    {
        return _context.getDevice(deviceID: deviceID)
    }
    
    public func getQueue(deviceID: Int) -> MTLCommandQueue
    {
        return _context.getQueue(deviceID: deviceID)
    }
    
    public func createCommand(_ pipeline: String, deviceID: Int) -> MetalCommand
    {
        return _context.createCommand(pipeline, deviceID: deviceID)
    }
    
    public func createBuffer<T>(metalBuffer: MetalBuffer<T>,
                                options: MTLResourceOptions,
                                deviceID: Int)
        -> MTLBuffer
    {
        return _context.createBuffer(
            metalBuffer: metalBuffer as! MetalSharedBuffer,
            options: options,
            deviceID: deviceID)
    }
    
    public func createBuffer(_ byteLength: Int,
                             options: MTLResourceOptions,
                             deviceID: Int)
        -> MTLBuffer
    {
        return _context.createBuffer(byteLength,
                                     options: options,
                                     deviceID: deviceID)
    }

    public func download<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        _checkDevice(metalBuffers)
        _context.download(metalBuffers)
    }
    
    public func download<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        _checkDevice(metalBuffers)
        _context.download(metalBuffers)
    }
    
    public func upload<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        _checkDevice(metalBuffers)
        _context.upload(metalBuffers)
    }
    
    public func upload<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        _checkDevice(metalBuffers)
        _context.upload(metalBuffers)
    }
    
    private func _checkDevice<T>(_ metalBuffers: [MetalBuffer<T>])
    {
        if metalBuffers.count == 0
        {
            fatalError("'metalBuffers' must contain at least one MetalBuffer.")
        }
        for metalBuffer in metalBuffers
        {
            if metalBuffer.deviceID != metalBuffers[0].deviceID
            {
                fatalError("'metalBuffers' must be on same device.")
            }
        }
    }
}

private class MetalListDevices
{
    var _listDevices = [MetalDevice]()
    
    init(_ listDevices: [MTLDevice])
    {
        for device in listDevices
        {
            _listDevices.append(MetalDevice(device: device))
        }
    }
    
    func buildKernels(
        libraryContent: String,
        kernelNames: [String],
        optimalThreadGroupSize: Bool)
    {
        for device in _listDevices
        {
            device.buildKernels(
                libraryContent: libraryContent,
                kernelNames: kernelNames,
                optimalThreadGroupSize: optimalThreadGroupSize
            )
        }
    }
    
    func buildKernels(
        libraryURL: URL,
        kernelNames: [String],
        optimalThreadGroupSize: Bool)
    {
        for device in _listDevices
        {
            device.buildKernels(
                libraryURL: libraryURL,
                kernelNames: kernelNames,
                optimalThreadGroupSize: optimalThreadGroupSize
            )
        }
    }
    
    func getDevice(deviceID: Int) -> MTLDevice
    {
        return _listDevices[deviceID].device
    }
    
    func getQueue(deviceID: Int) -> MTLCommandQueue
    {
        return _listDevices[deviceID].queue
    }
    
    func createCommand(_ pipeline: String, deviceID: Int) -> MetalCommand
    {
        return _listDevices[deviceID].createCommand(pipeline)
    }
    
    func createBuffer<T>(metalBuffer: MetalSharedBuffer<T>,
                         options: MTLResourceOptions,
                         deviceID: Int) -> MTLBuffer
    {
        return _listDevices[deviceID].createBuffer(
            metalBuffer: metalBuffer,
            options: options)
    }
    
    func createBuffer(_ byteLength: Int,
                      options: MTLResourceOptions,
                      deviceID: Int) -> MTLBuffer
    {
        return _listDevices[deviceID].createBuffer(byteLength,
                                                   options: options)
    }
    
    func download<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        _listDevices[metalBuffers[0].deviceID].download(metalBuffers)
    }
    
    func download<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        _listDevices[metalBuffers[0].deviceID].download(metalBuffers)
    }
    
    func upload<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        _listDevices[metalBuffers[0].deviceID].upload(metalBuffers)
    }
    
    func upload<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        _listDevices[metalBuffers[0].deviceID].upload(metalBuffers)
    }
}

private class MetalDevice
{
    let _device: MTLDevice
    let _queue: MTLCommandQueue
    
    var _pipelines: [String : MTLComputePipelineState] = [:]
    
    var device: MTLDevice
    {
        get {
            return _device
        }
    }
    
    var queue: MTLCommandQueue
    {
        get {
            return _queue
        }
    }
    
    init(device: MTLDevice)
    {
        _device = device
        
        // Queue to handle an ordered list of command buffers
        _queue = _device.makeCommandQueue()!
        
        _initKernels()
    }
    
    private func _initKernels()
    {
        // TODO: add kernels.
    }
    
    func buildKernels(
        libraryContent: String,
        kernelNames: [String],
        optimalThreadGroupSize: Bool)
    {
        let library = _buildLibrary(content: libraryContent)
        for name in kernelNames
        {
            _buildKernel(
                name: name,
                library: library,
                optimalThreadGroupSize: optimalThreadGroupSize
            )
        }
    }
    
    func buildKernels(
        libraryURL: URL,
        kernelNames: [String],
        optimalThreadGroupSize: Bool)
    {
        let library = _buildLibrary(url: libraryURL)
        for name in kernelNames
        {
            _buildKernel(
                name: name,
                library: library,
                optimalThreadGroupSize: optimalThreadGroupSize
            )
        }
    }
    
    private func _buildLibrary(content: String) -> MTLLibrary
    {
        let library: MTLLibrary
        do
        {
            library = try _device.makeLibrary(source: content, options: nil)
        }
        catch let error
        {
            fatalError("Cannot find \(error.localizedDescription).")
        }
        return library
    }
    
    private func _buildLibrary(url: URL) -> MTLLibrary
    {
        let library: MTLLibrary
        do
        {
            library = try _device.makeLibrary(URL: url)
        }
        catch let error
        {
            fatalError("Cannot find \(error.localizedDescription).")
        }
        return library
    }
    
    private func _buildKernel(
        name: String,
        library: MTLLibrary,
        optimalThreadGroupSize: Bool)
    {
        do {
            let kernelFunction = library.makeFunction(name: name)!
            let pipelineTmp: MTLComputePipelineState
            if optimalThreadGroupSize
            {
                let computeDescriptor = MTLComputePipelineDescriptor()
                computeDescriptor.computeFunction = kernelFunction
                computeDescriptor
                    .threadGroupSizeIsMultipleOfThreadExecutionWidth = true
                
                pipelineTmp = try _device.makeComputePipelineState(
                    descriptor: computeDescriptor,
                    options: [],
                    reflection: nil
                )
            }
            else
            {
                pipelineTmp = try _device.makeComputePipelineState(
                    function: kernelFunction)
            }
            _pipelines[name] = pipelineTmp
        }
        catch {
            fatalError("Impossible to build kernel \(name).")
        }
    }
    
    func createCommand(_ pipeline: String) -> MetalCommand
    {
        if let pipelineTmp = _pipelines[pipeline]
        {
            return MetalCommand(queue: _queue, pipeline: pipelineTmp)
        }
        fatalError("Could not find pipeline: \(pipeline).")
    }
    
    func createBuffer<T>(metalBuffer: MetalSharedBuffer<T>,
                         options: MTLResourceOptions = [])
        -> MTLBuffer
    {
        return _device.makeBuffer(bytesNoCopy: metalBuffer.memory,
                                  length: metalBuffer.allocationSize,
                                  options: options)!
    }
    
    func createBuffer(_ byteLength: Int,
                      options: MTLResourceOptions)
        -> MTLBuffer
    {
        return _device.makeBuffer(length: byteLength, options: options)!
    }
    
    func download<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        var lastCommand: MTLCommandBuffer! = nil
        for metalBuffer in metalBuffers
        {
            // Buffer for storing encoded commands that are sent to GPU
            let command = _queue.makeCommandBuffer()!
            
            // Creates the command encoder from the command buffer
            let blitEncoder = command.makeBlitCommandEncoder()!
            
            // Command
            blitEncoder.synchronize(resource: metalBuffer.metal)
            
            // Finalize configuration
            blitEncoder.endEncoding()
            
            // Add command buffer to the queue
            command.enqueue()
            
            // Start job
            command.commit()
            
            lastCommand = command
        }
        
        if lastCommand != nil
        {
            // Wait for GPU to finish
            lastCommand.waitUntilCompleted()
        }
    }
    
    func download<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        var metalSharedBuffers = [MetalSharedBuffer<T>]()
        for metalBuffer in metalBuffers
        {
            // Buffer for storing encoded commands that are sent to GPU
            let command = _queue.makeCommandBuffer()!
            
            // Creates the command encoder from the command buffer
            let blitEncoder = command.makeBlitCommandEncoder()!
            
            // Command
            blitEncoder.copy(
                from: metalBuffer.metal,
                sourceOffset: 0,
                to: metalBuffer.shared.metal,
                destinationOffset: 0,
                size: metalBuffer.nbElems * MemoryLayout<T>.size)
            
            // Finalize configuration
            blitEncoder.endEncoding()
            
            // Add command buffer to the queue
            command.enqueue()
            
            // Start job
            command.commit()
            
            metalSharedBuffers.append(metalBuffer.shared)
        }
        
        download(metalSharedBuffers)
    }
    
    func upload<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        for metalBuffer in metalBuffers
        {
            metalBuffer.metal.didModifyRange(
                0..<metalBuffer.nbElems * MemoryLayout<T>.size)
        }
    }
    
    func upload<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        var metalSharedBuffers = [MetalSharedBuffer<T>]()
        for metalBuffer in metalBuffers
        {
            metalSharedBuffers.append(metalBuffer.shared)
        }
        upload(metalSharedBuffers)
        
        var lastCommand: MTLCommandBuffer! = nil
        for metalBuffer in metalBuffers
        {
            // Buffer for storing encoded commands that are sent to GPU
            let command = _queue.makeCommandBuffer()!
            
            // Creates the command encoder from the command buffer
            let blitEncoder = command.makeBlitCommandEncoder()!
            
            // Command
            blitEncoder.copy(
                from: metalBuffer.shared.metal,
                sourceOffset: 0,
                to: metalBuffer.metal,
                destinationOffset: 0,
                size: metalBuffer.nbElems * MemoryLayout<T>.size)
            
            // Finalize configuration
            blitEncoder.endEncoding()
            
            // Add command buffer to the queue
            command.enqueue()
            
            // Start job
            command.commit()
            
            lastCommand = command
        }
        
        if lastCommand != nil
        {
            // Wait for GPU to endIsh
            lastCommand.waitUntilCompleted()
        }
        
        // do not neet shared anymore
        for metalBuffer in metalBuffers
        {
            metalBuffer._shared = nil
        }
    }
}

public class MetalCommand
{
    var _command: MTLCommandBuffer!
    var _encoder: MTLComputeCommandEncoder!
    var _pipeline: MTLComputePipelineState
    
    public var threadExecutionWidth: Int
    {
        get {
            return _pipeline.threadExecutionWidth
        }
    }
    
    public var maxThreadsPerThreadgroup: Int
    {
        get {
            return _pipeline.maxTotalThreadsPerThreadgroup
        }
    }
    
    init(queue: MTLCommandQueue, pipeline: MTLComputePipelineState)
    {
        _pipeline = pipeline
        
        // Buffer for storing encoded commands that are sent to GPU
        _command = queue.makeCommandBuffer()
        
        // Creates the command encoder from the command buffer
        autoreleasepool()
        {
            _encoder = _command.makeComputeCommandEncoder()
        }
        
        // Encodes the pipeline state command
        _encoder.setComputePipelineState(pipeline)
    }
    
    public func setBytes<T>(_ data: [T], atIndex index: Int)
    {
        let byteLength = data.count * MemoryLayout<T>.size
        _encoder.setBytes(data, length: byteLength, index: index)
    }
    
    public func setBuffer(_ buffer: MTLBuffer, atIndex index: Int)
    {
        _encoder.setBuffer(buffer, offset: 0, index: index)
    }
    
    public func setTexture(_ texture: MTLTexture, atIndex index: Int)
    {
        _encoder.setTexture(texture, index: index)
    }
    
    public func dispatchThreads(threadsPerGrid: MTLSize,
                                threadsPerThreadgroup: MTLSize)
    {
        _encoder.dispatchThreads(threadsPerGrid,
                                 threadsPerThreadgroup: threadsPerThreadgroup)
    }
    
    public func enqueue()
    {
        // Finalize configuration
        _encoder.endEncoding()
        
        // Add command buffer to the queue
        _command.enqueue()
        
        // Start job
        _command.commit()
        
        if MAKit.Time.track
        {
            // Wait for it to endIsh
            _command.waitUntilCompleted()
        }
    }
}
