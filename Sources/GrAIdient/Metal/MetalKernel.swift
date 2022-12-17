//
// MetalKernel.swift
// GrAIdient
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Foundation
import Metal

/// Main class to interact with GPU.
public class MetalKernel
{
    /// Global state.
    static private var _metalKernel: MetalKernel! = nil
    
    /// Access to GPU devices.
    fileprivate let _context: MetalListDevices
    
    /// Get the main instance.
    public static var get: MetalKernel
    {
        get {
            if _metalKernel == nil
            {
                var listDevices = [MTLDevice]()
                
                let gpuNames = GrAI.Opti.gpuNamedPriority
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
                
                if listDevices.count == 0
                {
                    fatalError("Could not find any GPU device.")
                }
                _metalKernel = MetalKernel(listDevices)
            }
            return _metalKernel
        }
    }
    
    ///
    /// Build the main instance with multiple GPU devices.
    ///
    /// - Parameter listDevices: The list of GPU devices.
    ///
    private init(_ listDevices: [MTLDevice])
    {
        _context = MetalListDevices(listDevices)
    }
    
    ///
    /// Build GPU kernels.
    ///
    /// This API is especially usefull when Metal compiler cannot be invoked on the source files
    /// (example: in a Swift package).
    ///
    /// - Parameters:
    ///     - libraryContent: The source file.
    ///     - kernelNames: The list of kernel functions in the file.
    ///     - optimalThreadGroupSize:
    ///         Whether ThreadGroupSize is a multiple of ThreadExecutionWidth.
    ///
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
    
    ///
    /// Build GPU kernels.
    ///
    /// This API is especially preferred when Metal compiler can be invoked on the source files.
    ///
    /// - Parameters:
    ///     - libraryURL: The source file.
    ///     - kernelNames: The list of kernel functions in the file.
    ///     - optimalThreadGroupSize:
    ///         Whether ThreadGroupSize is a multiple of ThreadExecutionWidth.
    ///
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
    
    ///
    /// Get a GPU device.
    ///
    /// - Parameter deviceID: The id of the GPU to retrieve.
    /// - Returns: The GPU device.
    ///
    public func getDevice(deviceID: Int) -> MTLDevice
    {
        return _context.getDevice(deviceID: deviceID)
    }
    
    ///
    /// Get the command queue associated to a GPU device.
    ///
    /// - Parameter deviceID: The id of the GPU to retrieve the command queue from.
    /// - Returns: The command queue.
    ///
    public func getQueue(deviceID: Int) -> MTLCommandQueue
    {
        return _context.getQueue(deviceID: deviceID)
    }
    
    ///
    /// Create a new command to execute on the GPU.
    ///
    /// - Parameters:
    ///     - pipeline: The name of the command to create.
    ///     - deviceID: The GPU device where to execute the command.
    /// - Returns: The command.
    ///
    public func createCommand(_ pipeline: String, deviceID: Int) -> MetalCommand
    {
        return _context.createCommand(pipeline, deviceID: deviceID)
    }
    
    ///
    /// Create a buffer on a GPU device that is also accessible on the CPU.
    ///
    /// - Parameter metalBuffer: Object responsible for maintaining the buffer on the GPU.
    /// - Returns: The GPU buffer.
    ///
    func createBuffer<T>(_ metalBuffer: MetalSharedBuffer<T>) -> MTLBuffer
    {
        return _context.createBuffer(metalBuffer)
    }
    
    ///
    /// Create a buffer on a GPU device.
    ///
    /// - Parameters:
    ///     - byteLength: Size of the created buffer on the GPU.
    ///     - options: Option to specify storageMode (managed or private).
    ///     - deviceID: The GPU device to send the  buffer to.
    /// - Returns: The GPU buffer.
    ///
    func createBuffer(
        _ byteLength: Int,
        options: MTLResourceOptions,
        deviceID: Int) -> MTLBuffer
    {
        return _context.createBuffer(
            byteLength,
            options: options,
            deviceID: deviceID
        )
    }

    ///
    /// Download GPU buffers to the CPU.
    ///
    /// This provokes the synchronisation needed by our managed storageMode.
    /// MetalSharedBuffer maintains a MTLBuffer that may be duplicated on the CPU
    /// (especially when using external GPU).
    ///
    /// - Parameter metalBuffers: The list of shared buffers to download from GPU.
    ///
    public func download<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        _checkDevice(metalBuffers)
        _context.download(metalBuffers)
    }
    
    ///
    /// Download GPU buffers to the CPU.
    ///
    /// Force the creation of a MetalSharedBuffer.
    /// This is a proxy in order to transfer data back to the CPU
    /// (MetalPrivateBuffer cannot do it by itself because of private storageMode).
    ///
    /// - Parameter metalBuffers: The list of private buffers to download from GPU.
    ///
    public func download<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        _checkDevice(metalBuffers)
        _context.download(metalBuffers)
    }
    
    ///
    /// Upload CPU data to GPU buffers.
    ///
    /// This provokes the synchronisation needed by our managed storageMode.
    /// MetalSharedBuffer maintains a MTLBuffer that may be duplicated on the CPU
    /// (especially when using external GPU).
    ///
    /// - Parameter metalBuffers: The list of shared buffers to upload to the GPU.
    ///
    public func upload<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        _checkDevice(metalBuffers)
        _context.upload(metalBuffers)
    }
    
    ///
    /// Upload CPU data to GPU buffers.
    ///
    /// Force the creation of a MetalSharedBuffer.
    /// This is a proxy in order to transfer data back to the CPU
    /// (MetalPrivateBuffer cannot do it by itself because of private storageMode).
    ///
    /// - Parameter metalBuffers: The list of private buffers to upload to GPU.
    ///
    public func upload<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        _checkDevice(metalBuffers)
        _context.upload(metalBuffers)
    }
    
    ///
    /// Some sanity checks.
    ///
    /// - Parameter metalBuffers: List of buffers to check.
    ///
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

/// Allow access to GPU devices.
private class MetalListDevices
{
    /// The list of GPU devices.
    var _listDevices = [MetalDevice]()
    
    ///
    /// Create an instance with access to several GPU devices.
    ///
    /// - Parameter listDevices: The list of GPU devices.
    ///
    init(_ listDevices: [MTLDevice])
    {
        for device in listDevices
        {
            _listDevices.append(MetalDevice(device: device))
        }
    }
    
    ///
    /// Build GPU kernels.
    ///
    /// This API is especially usefull when Metal compiler cannot be invoked on the source files
    /// (example: in a Swift package).
    ///
    /// - Parameters:
    ///     - libraryContent: The source file.
    ///     - kernelNames: The list of kernel functions in the file.
    ///     - optimalThreadGroupSize:
    ///         Whether ThreadGroupSize is a multiple of ThreadExecutionWidth.
    ///
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
    
    ///
    /// Build GPU kernels.
    ///
    /// This API is especially preferred when Metal compiler can be invoked on the source files.
    ///
    /// - Parameters:
    ///     - libraryURL: The source file.
    ///     - kernelNames: The list of kernel functions in the file.
    ///     - optimalThreadGroupSize:
    ///         Whether ThreadGroupSize is a multiple of ThreadExecutionWidth.
    ///
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
    
    ///
    /// Get a GPU device.
    ///
    /// - Parameter deviceID: The id of the GPU to retrieve.
    /// - Returns: The GPU device.
    ///
    func getDevice(deviceID: Int) -> MTLDevice
    {
        return _listDevices[deviceID].device
    }
    
    ///
    /// Get the command queue associated to a GPU device.
    ///
    /// - Parameter deviceID: The id of the GPU to retrieve the command queue from.
    /// - Returns: The command queue.
    ///
    func getQueue(deviceID: Int) -> MTLCommandQueue
    {
        return _listDevices[deviceID].queue
    }
    
    ///
    /// Create a new command to execute on the GPU.
    ///
    /// - Parameters:
    ///     - pipeline: The name of the command to create.
    ///     - deviceID: The GPU device where to execute the command.
    /// - Returns: The command.
    ///
    func createCommand(_ pipeline: String, deviceID: Int) -> MetalCommand
    {
        return _listDevices[deviceID].createCommand(pipeline)
    }
    
    ///
    /// Create a buffer on a GPU device that is also accessible on the CPU.
    ///
    /// - Parameter metalBuffer: Object responsible for maintaining the buffer on the GPU.
    /// - Returns: The GPU buffer.
    ///
    func createBuffer<T>(_ metalBuffer: MetalSharedBuffer<T>) -> MTLBuffer
    {
        return _listDevices[metalBuffer.deviceID].createBuffer(metalBuffer)
    }
    
    ///
    /// Create a buffer on a GPU device.
    ///
    /// - Parameters:
    ///     - byteLength: Size of the created buffer on the GPU.
    ///     - options: Option to specify storageMode (managed or private).
    ///     - deviceID: The GPU device to send the  buffer to.
    /// - Returns: The GPU buffer.
    ///
    func createBuffer(
        _ byteLength: Int,
        options: MTLResourceOptions,
        deviceID: Int) -> MTLBuffer
    {
        return _listDevices[deviceID].createBuffer(
            byteLength,
            options: options
        )
    }
    
    ///
    /// Download GPU buffers to the CPU.
    ///
    /// This provokes the synchronisation needed by our managed storageMode.
    /// MetalSharedBuffer maintains a MTLBuffer that may be duplicated on the CPU
    /// (especially when using external GPU).
    ///
    /// - Parameter metalBuffers: The list of shared buffers to download from GPU.
    ///
    func download<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        _listDevices[metalBuffers[0].deviceID].download(metalBuffers)
    }
    
    ///
    /// Download GPU buffers to the CPU.
    ///
    /// Force the creation of a MetalSharedBuffer.
    /// This is a proxy in order to transfer data back to the CPU
    /// (MetalPrivateBuffer cannot do it by itself because of private storageMode).
    ///
    /// - Parameter metalBuffers: The list of private buffers to download from GPU.
    ///
    func download<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        _listDevices[metalBuffers[0].deviceID].download(metalBuffers)
    }
    
    ///
    /// Upload CPU data to GPU buffers.
    ///
    /// This provokes the synchronisation needed by our managed storageMode.
    /// MetalSharedBuffer maintains a MTLBuffer that may be duplicated on the CPU
    /// (especially when using external GPU).
    ///
    /// - Parameter metalBuffers: The list of shared buffers to upload to the GPU.
    ///
    func upload<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        _listDevices[metalBuffers[0].deviceID].upload(metalBuffers)
    }
    
    ///
    /// Upload CPU data to GPU buffers.
    ///
    /// Force the creation of a MetalSharedBuffer.
    /// This is a proxy in order to transfer data back to the CPU
    /// (MetalPrivateBuffer cannot do it by itself because of private storageMode).
    ///
    /// - Parameter metalBuffers: The list of private buffers to upload to GPU.
    ///
    func upload<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        _listDevices[metalBuffers[0].deviceID].upload(metalBuffers)
    }
}

/// Represents a single GPU device.
private class MetalDevice
{
    /// The GPU device.
    let _device: MTLDevice
    /// The queue associated with the GPU device.
    let _queue: MTLCommandQueue
    
    /// The different kernels' state available.
    var _pipelines: [String : MTLComputePipelineState] = [:]
    
    /// Get the GPU device.
    var device: MTLDevice
    {
        get {
            return _device
        }
    }
    
    /// Get the GPU command queue.
    var queue: MTLCommandQueue
    {
        get {
            return _queue
        }
    }
    
    ///
    /// Create an object with an access to one GPU device.
    ///
    /// - Parameter device: The GPU device.
    ///
    init(device: MTLDevice)
    {
        _device = device
        _queue = _device.makeCommandQueue()!
        
        _initKernels()
    }
    
    /// Initialize the GPU kernels' state.
    private func _initKernels()
    {
        let listKernels =
        [
            "Activation": [
                "forwardReLU",
                "backwardReLU",
                "forwardLeakyReLU",
                "backwardLeakyReLU",
                "forwardSoftReLU",
                "backwardSoftReLU",
                "forwardSigmoid",
                "backwardSigmoid",
            ],
            "Biases": [
                "reduceBiases",
            ],
            "BN": [
                "computeConvμ",
                "computeConvσ2",
                "forwardBNConvTraining",
                "forwardBNConvInference",
                "backwardWeightsBNConv",
                "backwardBNConvTraining",
                "backwardBNConvInference",
            ],
            "Convolution": [
                "convForward",
                "convBackward",
                "convBatchDerWeights",
                "convBatchDerBiases",
                "convDerWeights",
                "convDerBiases",
                "convReduceWeights",
            ],
            "FullyConnected": [
                "flForward",
                "flBackward",
                "flBatchDerWeights",
                "flBatchDerBiases",
                "flDerWeights",
                "flDerBiases",
                "flReduceWeights",
            ],
            "Layer1D": [
                "MSE1DLoss",
                "MSE1DLossDerivative",
                "linearErrorLoss",
                "linearErrorLossDerivative",
                "selectNeurons1DForward",
                "selectNeurons1DBackward",
            ],
            "Layer2D": [
                "avgPoolForward",
                "avgPoolBackward",
                "maxPoolForward",
                "maxPoolBackward",
                "adaptiveAvgPoolForward1",
                "adaptiveAvgPoolForward2",
                "adaptiveAvgPoolBackward1",
                "adaptiveAvgPoolBackward2",
                "selectNeurons2DForward",
                "selectNeurons2DBackward",
                "IRDFT2RGBForward",
                "IRDFT2RGBBackward",
                "decorrelateRGBForward",
                "decorrelateRGBBackward",
                "linearScale2DForward",
                "linearScale2DBackward",
                "setDataFTFrequences2D",
                "pad2DForward",
                "pad2DBackward",
                "crop2DForward",
                "crop2DBackward",
                "resizeBilinearPadForward",
                "resizeBilinearPadBackward",
                "rotate2DForward",
                "rotate2DBackward",
                "resizeBilinearCropForward",
                "resizeBilinearCropBackward",
            ],
            "Merge": [
                "sum1",
                "sum2",
                "multiplyForward",
                "multiplyBackward",
            ],
            "Optimizer": [
                "clipGradients",
                "multiplyGradients",
                "weightsSGD",
                "weightsMomentum",
                "weightsAdam",
                "weightsAMSGrad",
                "weightsAdamRectified",
                "weightsAdaBound",
                "weightsAMSBound",
            ],
            "Reset": [
                "reset"
            ]
        ]
        
        for (libName, kernelNames) in listKernels
        {
            let lib = Bundle.module.url(
                forResource: libName,
                withExtension: "metal",
                subdirectory: "Kernel"
            )!
            do
            {
                let content = try String(contentsOf: lib)
                let library = _buildLibrary(content: content)
                
                for kernelName in kernelNames
                {
                    _buildKernel(
                        name: kernelName,
                        library: library,
                        optimalThreadGroupSize: true
                    )
                }
            }
            catch let error
            {
                fatalError("Cannot init \(error.localizedDescription).")
            }
        }
    }
    
    ///
    /// Build GPU kernels.
    ///
    /// This API is especially usefull when Metal compiler cannot be invoked on the source files
    /// (example: in a Swift package).
    ///
    /// - Parameters:
    ///     - libraryContent: The source file.
    ///     - kernelNames: The list of kernel functions in the file.
    ///     - optimalThreadGroupSize:
    ///         Whether ThreadGroupSize is a multiple of ThreadExecutionWidth.
    ///
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
    
    ///
    /// Build GPU kernels.
    ///
    /// This API is especially preferred when Metal compiler can be invoked on the source files.
    ///
    /// - Parameters:
    ///     - libraryURL: The source file.
    ///     - kernelNames: The list of kernel functions in the file.
    ///     - optimalThreadGroupSize:
    ///         Whether ThreadGroupSize is a multiple of ThreadExecutionWidth.
    ///
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
    
    ///
    /// Build a library from a Metal source code content.
    ///
    /// - Parameter content: The source code content.
    /// - Returns: The library.
    ///
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
    
    ///
    /// Build a library from a Metal source code file.
    ///
    /// - Parameter url: The source code file url.
    /// - Returns: The library.
    ///
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
    
    ///
    /// Build GPU kernels from a Metal library.
    ///
    /// This function updates the different kernel's state `_pipelines`.
    ///
    /// - Parameters:
    ///     - name: The name of the kernel to build.
    ///     - library: The Metal library.
    ///     - optimalThreadGroupSize:
    ///         Whether ThreadGroupSize is a multiple of ThreadExecutionWidth.
    ///
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
    
    ///
    /// Create a new command to execute on the GPU.
    ///
    /// - Parameters:
    ///     - pipeline: The name of the command to create.
    ///     - deviceID: The GPU device where to execute the command.
    /// - Returns: The command.
    ///
    func createCommand(_ pipeline: String) -> MetalCommand
    {
        if let pipelineTmp = _pipelines[pipeline]
        {
            return MetalCommand(queue: _queue, pipeline: pipelineTmp)
        }
        fatalError("Could not find pipeline: \(pipeline).")
    }
    
    ///
    /// Create a buffer on a GPU device that is also accessible on the CPU.
    ///
    /// - Parameter metalBuffer: Object responsible for maintaining the buffer on the GPU.
    /// - Returns: The GPU buffer.
    ///
    func createBuffer<T>(_ metalBuffer: MetalSharedBuffer<T>) -> MTLBuffer
    {
        return _device.makeBuffer(
            bytesNoCopy: metalBuffer.memory,
            length: metalBuffer.allocationSize,
            options: MTLResourceOptions.storageModeManaged
        )!
    }
    
    ///
    /// Create a buffer on a GPU device.
    ///
    /// - Parameters:
    ///     - byteLength: Size of the created buffer on the GPU.
    ///     - options: Option to specify storageMode (managed or private).
    /// - Returns: The GPU buffer.
    ///
    func createBuffer(
        _ byteLength: Int,
        options: MTLResourceOptions) -> MTLBuffer
    {
        return _device.makeBuffer(length: byteLength, options: options)!
    }
    
    ///
    /// Download GPU buffers to the CPU.
    ///
    /// This provokes the synchronisation needed by our managed storageMode.
    /// MetalSharedBuffer maintains a MTLBuffer that may be duplicated on the CPU
    /// (especially when using external GPU).
    ///
    /// - Parameter metalBuffers: The list of shared buffers to download from GPU.
    ///
    func download<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        var lastCommand: MTLCommandBuffer! = nil
        for metalBuffer in metalBuffers
        {
            // Buffer for storing encoded commands that are sent to GPU.
            let command = _queue.makeCommandBuffer()!
            
            // Create the command encoder from the command buffer.
            let blitEncoder = command.makeBlitCommandEncoder()!
            
            // Command.
            blitEncoder.synchronize(resource: metalBuffer.metal)
            
            // Finalize configuration.
            blitEncoder.endEncoding()
            
            // Add command buffer to the queue.
            command.enqueue()
            
            // Start job.
            command.commit()
            
            lastCommand = command
        }
        
        if lastCommand != nil
        {
            // Wait for GPU to end.
            lastCommand.waitUntilCompleted()
        }
    }
    
    ///
    /// Download GPU buffers to the CPU.
    ///
    /// Force the creation of a MetalSharedBuffer.
    /// This is a proxy in order to transfer data back to the CPU
    /// (MetalPrivateBuffer cannot do it by itself because of private storageMode).
    ///
    /// - Parameter metalBuffers: The list of private buffers to download from GPU.
    ///
    func download<T>(_ metalBuffers: [MetalPrivateBuffer<T>])
    {
        var metalSharedBuffers = [MetalSharedBuffer<T>]()
        for metalBuffer in metalBuffers
        {
            // Buffer for storing encoded commands that are sent to GPU.
            let command = _queue.makeCommandBuffer()!
            
            // Create the command encoder from the command buffer.
            let blitEncoder = command.makeBlitCommandEncoder()!
            
            // Command.
            blitEncoder.copy(
                from: metalBuffer.metal,
                sourceOffset: 0,
                to: metalBuffer.shared.metal,
                destinationOffset: 0,
                size: metalBuffer.nbElems * MemoryLayout<T>.size
            )
            
            // Finalize configuration.
            blitEncoder.endEncoding()
            
            // Add command buffer to the queue.
            command.enqueue()
            
            // Start job.
            command.commit()
            
            metalSharedBuffers.append(metalBuffer.shared)
        }
        
        download(metalSharedBuffers)
    }
    
    ///
    /// Upload CPU data to GPU buffers.
    ///
    /// This provokes the synchronisation needed by our managed storageMode.
    /// MetalSharedBuffer maintains a MTLBuffer that may be duplicated on the CPU
    /// (especially when using external GPU).
    ///
    /// - Parameter metalBuffers: The list of shared buffers to upload to the GPU.
    ///
    func upload<T>(_ metalBuffers: [MetalSharedBuffer<T>])
    {
        for metalBuffer in metalBuffers
        {
            metalBuffer.metal.didModifyRange(
                0..<metalBuffer.nbElems * MemoryLayout<T>.size
            )
        }
    }
    
    ///
    /// Upload CPU data to GPU buffers.
    ///
    /// Force the creation of a MetalSharedBuffer.
    /// This is a proxy in order to transfer data back to the CPU
    /// (MetalPrivateBuffer cannot do it by itself because of private storageMode).
    ///
    /// - Parameter metalBuffers: The list of private buffers to upload to GPU.
    ///
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
            // Buffer for storing encoded commands that are sent to GPU.
            let command = _queue.makeCommandBuffer()!
            
            // Create the command encoder from the command buffer.
            let blitEncoder = command.makeBlitCommandEncoder()!
            
            // Command.
            blitEncoder.copy(
                from: metalBuffer.shared.metal,
                sourceOffset: 0,
                to: metalBuffer.metal,
                destinationOffset: 0,
                size: metalBuffer.nbElems * MemoryLayout<T>.size
            )
            
            // Finalize configuration.
            blitEncoder.endEncoding()
            
            // Add command buffer to the queue.
            command.enqueue()
            
            // Start job.
            command.commit()
            
            lastCommand = command
        }
        
        if lastCommand != nil
        {
            // Wait for GPU to end.
            lastCommand.waitUntilCompleted()
        }
        
        // We do not neet shared anymore.
        for metalBuffer in metalBuffers
        {
            metalBuffer._shared = nil
        }
    }
}

/// A command to run on the GPU
public class MetalCommand
{
    /// The command to run on the GPU.
    var _command: MTLCommandBuffer!
    /// The encoder used load parameters in the command.
    var _encoder: MTLComputeCommandEncoder!
    /// The state of the command.
    var _pipeline: MTLComputePipelineState
    
    /// For most efficient execution, the threadgroup size should be a multiple of this
    /// when executing the kernel.
    public var threadExecutionWidth: Int
    {
        get {
            return _pipeline.threadExecutionWidth
        }
    }
    
    /// The maximum total number of threads that can be in a single compute threadgroup.
    public var maxThreadsPerThreadgroup: Int
    {
        get {
            return _pipeline.maxTotalThreadsPerThreadgroup
        }
    }
    
    ///
    /// Create a new command on the GPU.
    ///
    /// - Parameters:
    ///     - queue: The command queue.
    ///     - pipeline: The command to run.
    ///
    init(queue: MTLCommandQueue, pipeline: MTLComputePipelineState)
    {
        _pipeline = pipeline
        
        // Buffer for storing encoded commands that are sent to GPU.
        _command = queue.makeCommandBuffer()
        
        // Create the command encoder from the command buffer.
        autoreleasepool()
        {
            _encoder = _command.makeComputeCommandEncoder()
        }
        
        // Encodes the pipeline state command.
        _encoder.setComputePipelineState(pipeline)
    }
    
    ///
    /// Send data to the command.
    ///
    /// - Parameters:
    ///     - data: The data to send to the command.
    ///     - index: The index of the placeholder receiving the data in the command definition.
    ///
    public func setBytes<T>(_ data: [T], atIndex index: Int)
    {
        let byteLength = data.count * MemoryLayout<T>.size
        _encoder.setBytes(data, length: byteLength, index: index)
    }
    
    ///
    /// Send a buffer to the command.
    ///
    /// - Parameters:
    ///     - buffer: The buffer to send to the command.
    ///     - index: The index of the placeholder receiving the buffer in the command definition.
    ///
    public func setBuffer(_ buffer: MTLBuffer, atIndex index: Int)
    {
        _encoder.setBuffer(buffer, offset: 0, index: index)
    }
    
    ///
    /// Send a texture to the command.
    ///
    /// - Parameters:
    ///     - texture: The texture to send to the command.
    ///     - index: The index of the placeholder receiving the texture in the command definition.
    ///
    public func setTexture(_ texture: MTLTexture, atIndex index: Int)
    {
        _encoder.setTexture(texture, index: index)
    }
    
    ///
    /// Dipsatch a "line" of parallel operations on the GPU.
    ///
    /// - Parameter nbThreads: The number of parallel operations.
    ///
    public func dispatchThreads(_ nbThreads: Int)
    {
        let threads = threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: nbThreads, height: 1, depth: 1)
        dispatchThreads(
            threadsPerGrid: threadsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }
    
    ///
    /// Dispatch a "grid" of parallel operations on the GPU.
    ///
    /// - Parameters:
    ///     - width: The grid's width.
    ///     - height: The grid's height.
    ///
    public func dispatchThreads(width: Int, height: Int)
    {
        if height == 1
        {
            dispatchThreads(width)
        }
        else if width == 1
        {
            let threads = threadExecutionWidth
            let threadsPerThreadgroup = MTLSizeMake(1, threads, 1)
            let threadsPerGrid = MTLSize(width: 1, height: height, depth: 1)
            dispatchThreads(
                threadsPerGrid: threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        else
        {
            let maxDim = max(width, height)
            let minDim = min(width, height)
            
            var ratio = Int(Double(maxDim) / Double(minDim))
            let maxRatio = maxThreadsPerThreadgroup / 64
            ratio = min(ratio, maxRatio, 4) // 4 is an hyper parameter
            // to try to optimize between local and eGPU.
            
            let threadsPerThreadgroup = width == maxDim ?
            MTLSizeMake(8 * ratio, 8, 1) :
            MTLSizeMake(8, 8 * ratio, 1)
            
            let threadsPerGrid = MTLSize(
                width: width,
                height: height,
                depth: 1
            )
            dispatchThreads(
                threadsPerGrid: threadsPerGrid,
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
    }
    
    ///
    /// Dispatch a "volume" of parallel operations on the GPU.
    ///
    /// A thread group contains multiple threads. The size of a thread group cannot exceed
    /// `maxThreadsPerThreadgroup`. This upper limit is usually shortened because
    /// of memory being shared between threads of the same thread group.
    /// Yet, a high number of threads in a thread group does not mean these many threads
    /// being executed at "the same time".
    /// The real concurrency comes from the SIMD groups. A SIMD group typically contains
    /// `threadExecutionWidth` threads that are executed at "the same time".
    /// For most efficient execution, the thread group size should be a multiple of
    /// `threadExecutionWidth`.
    /// Also note that `maxThreadsPerThreadgroup` is a multiple of the latter.
    ///
    /// - Parameters:
    ///     - threadsPerGrid: The total number of elements in grid.
    ///     - threadsPerThreadgroup: The number of elements in each thread group.
    ///
    public func dispatchThreads(
        threadsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize)
    {
        _encoder.dispatchThreads(
            threadsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }
    
    /// Enqueue command and start job.
    public func enqueue()
    {
        // Finalize configuration.
        _encoder.endEncoding()
        
        // Add command buffer to the queue.
        _command.enqueue()
        
        // Start job.
        _command.commit()
    }
}
