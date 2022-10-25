//
// MAKit.swift
// MAKit
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Foundation

/// Get access to some global functions and variables.
public class MAKit
{
    /// Get access to stored variables.
    private static var getCtx: MAKitContext
    {
        get {
            return MAKitContext.get
        }
    }
    private init() {}
    
    /// Namespace for optimization settings.
    public class Opti
    {
        /// Get/Set run on CPU.
        public static var CPU: Bool
        {
            get {
                return getCtx.optim == MAKitContext.Optimization.CPU
            }
            set {
                if newValue
                {
                    getCtx.optim = MAKitContext.Optimization.CPU
                }
                else
                {
                    getCtx.optim = MAKitContext.Optimization.GPU
                }
            }
        }
        /// Get/Set run on GPU.
        public static var GPU: Bool
        {
            get {
                return getCtx.optim == MAKitContext.Optimization.GPU
            }
            set {
                if newValue
                {
                    getCtx.optim = MAKitContext.Optimization.GPU
                }
                else
                {
                    getCtx.optim = MAKitContext.Optimization.CPU
                }
            }
        }
        
        /// Get/Set GPU names in the desired order (the first in the list is the first to be used).
        public static var gpuNamedPriority: [String]
        {
            get {
                return getCtx.gpuNamedPriority
            }
            set {
                getCtx.gpuNamedPriority = newValue
            }
        }
    }
    
    /// Namespace for gradient settings.
    public class Gradient
    {
        /// Get/Set compute the weights gradient per batch.
        public static var batch: Bool
        {
            get {
                return getCtx.gradient == MAKitContext.Gradient.Batch
            }
            set {
                if newValue
                {
                    getCtx.gradient = MAKitContext.Gradient.Batch
                }
                else
                {
                    getCtx.gradient = MAKitContext.Gradient.Sample
                }
            }
        }
        /// Get/Set compute the weights gradient per sample (will use more memory on GPU).
        public static var sample: Bool
        {
            get {
                return getCtx.gradient == MAKitContext.Gradient.Sample
            }
            set {
                if newValue
                {
                    getCtx.gradient = MAKitContext.Gradient.Sample
                }
                else
                {
                    getCtx.gradient = MAKitContext.Gradient.Batch
                }
            }
        }
    }
    
    /// Namespace for model settings.
    public class Model
    {
        /// Parameters allowing to discribute model context to the layers.
        public struct Params
        {
            /// The model context in which the layer is being created.
            public var context: ModelContext
            /// Whether the layer should be exposed as a layer of the model
            /// or not (case where the layer is contained in another one).
            public var hidden = false
            
            ///
            /// Create a parameter to distribute the model context.
            ///
            /// - Parameter context: The model context.
            ///
            public init(context: ModelContext)
            {
                self.context = context
            }
            
            ///
            /// Create a parameter to distribute the model context.
            ///
            /// - Parameter params: A parameter to copy.
            ///
            public init(params: Params)
            {
                self.context = params.context
                self.hidden = params.hidden
            }
        }
        
        /// Namespace for layer settings.
        public class Layer
        {
            ///
            /// Populate the registry of layers to serialize/deserialize.
            ///
            /// - Parameter registry: A dictionary to map serializable layers.
            ///
            public static func append(registry: [String: Codable.Type])
            {
                getCtx.layerRegistries.append(registry)
            }
            
            ///
            /// Retrieve the layer type from the registry.
            ///
            /// - Parameter layer: The string to look for in the registry.
            /// - Returns: The layer type associated.
            ///
            static func getType(layer: String) -> Codable.Type?
            {
                for registry in getCtx.layerRegistries
                {
                    if let type = registry[layer]
                    {
                        return type
                    }
                }
                return nil
            }
        }
        
        /// Namespace for activation function settings.
        public class Activation
        {
            ///
            /// Populate the registry of activation function factories.
            ///
            /// - Parameter kernel: A factory to build activation functions..
            ///
            public static func append(kernel: ActivationKernel)
            {
                getCtx.activationKernels.append(kernel)
            }
            
            ///
            /// Populate the registry of activation function to serialize/deserialize.
            ///
            /// - Parameter registry: A dictionary to map serializable activation functions.
            ///
            public static func append(registry: [String: Codable.Type])
            {
                getCtx.activationRegistries.append(registry)
            }
            
            ///
            /// Build the activation function.
            ///
            /// - Parameter name: The activation function string to build.
            /// - Returns: The activation function built.
            ///
            static func build(_ name: String) -> ActivationFunction?
            {
                for kernel in getCtx.activationKernels.reversed()
                {
                    if let activation = kernel.build(name)
                    {
                        return activation
                    }
                }
                return nil
            }
            
            ///
            /// Retrieve the activation function type from the registry.
            ///
            /// - Parameter layer: The string to look for in the registry.
            /// - Returns: The activation function type associated.
            ///
            static func getType(activation: String) -> Codable.Type?
            {
                for registry in getCtx.activationRegistries.reversed()
                {
                    if let type = registry[activation]
                    {
                        return type
                    }
                }
                return nil
            }
        }
    }
    
    /// Namespace for optimizer settings.
    public class Optimizer
    {
        /// Optimizer algorithm.
        public enum Class
        {
            case SGD, SGDMomentum
            case Adam, AMSGrad, AdamRectified
            case AdaBound, AMSBound
        }
        
        /// Parameters needed to run optimizers.
        public struct Params
        {
            /// Time step used by certain optimizers (example: Adam)
            var t: Int = 0
            /// Step of the model training.
            public var step: Int = 0
            
            /// Number of steps per epoch.
            public var nbLoops: Int = -1
            
            /// Optimizer's scheduler.
            public var optimizer: TimeScheduler! = nil
            /// Dictionary of variables' scheduler.
            public var variables: [String:TimeVariable] = [:]
            
            /// Use gradient clipping during training.
            public var gradientClipping: Bool = false
            /// Threshold above which to cut the gradients.
            public var normThreshold: Double = 1.0
            
            /// Create parameters for optimizer.
            public init() {}
        }
        
        /* example:
         
        var optimizerParams = MAKit.Optimizer.Params()
         
        optimizerParams.batchSize = batchSize
        optimizerParams.nbSamples = nbSamples
        optimizerParams.nbLoops = nbLoops
         
        optimizerParams.optimizer = ConstEpochsScheduler(
            MAKit.Optimizer.Class.AdamRectified
        )
         
        optimizerParams.variables =
        [
            "alpha": MultEpochsVar(
                epoch0: 5,
                epochMul: 2,
                value: LinearDescending(min: 0.0001, max: 0.005)
            ),
            "lambda": ConstEpochsVar(value: ConstVal(0.003))
        ] */
    }
    
    /// Namespace for run settings.
    public class Loop
    {
        /// Get/Set run the gradient checking algorithm.
        public static var gradientChecking: Bool
        {
            get {
                return getCtx.gc
            }
            set {
                getCtx.gc = newValue
            }
        }
    }
    
    /// Namespace for dump settings.
    public class Dump
    {
        /// Get/Set the directory where to read data.
        public static var inputDir: URL
        {
            get {
                return getCtx.inputDir
            }
            set {
                getCtx.inputDir = newValue
            }
        }
        /// Get/Set the directory where to dump outputs.
        public static var outputDir: URL
        {
            get {
                return getCtx.outputDir
            }
            set {
                getCtx.outputDir = newValue
            }
        }
        /// Get/Set the directory where to read and write models.
        public static var modeleDir: URL
        {
            get {
                return getCtx.modeleDir
            }
            set {
                getCtx.modeleDir = newValue
            }
        }
    }
    
    /// Namespace for time tracking settings.
    public class Time
    {
        /// Get/Set track time.
        public static var track: Bool
        {
            get {
                return getCtx.trackTime
            }
            set {
                getCtx.trackTime = newValue
            }
        }
        
        /// Start trackking time.
        public static func start()
        {
            if track
            {
                TimeTransaction.get.start()
            }
        }
        
        ///
        /// Stop tracking time.
        ///
        /// Throw an error when time is not being tracked or the output cannot be written.
        ///
        /// - Parameters:
        ///     - id: The id of the function tracked.
        ///     - description: A short description of the function.
        ///
        public static func stop(id: String, description: String) throws
        {
            if track
            {
                try TimeTransaction.get.stop(id: id, description: description)
            }
        }
        
        ///
        /// Dump aggreegated tracked time.
        ///
        /// Throw an error when time is not being tracked.
        ///
        /// - Parameters:
        ///     - id: The id of the function tracked.
        ///     - description: A short description of the function.
        ///
        public static func dumpStacked() throws
        {
            if track
            {
                try TimeTransaction.get.dumpStacked()
            }
            else
            {
                throw TimeError.TrackTime
            }
        }
    }
}

/// A global context with stored variables.
fileprivate class MAKitContext
{
    private static let _ctx = MAKitContext()
    
    /// Access to the context.
    static var get: MAKitContext
    {
        get {
            return _ctx
        }
    }
    
    //--------------------------------------------------------------------------
    // OPTI
    //--------------------------------------------------------------------------
    /// Optimization variable.
    var optim = Optimization.CPU
    enum Optimization
    {
        case CPU
        case GPU
    }
    
    var gpuNamedPriority = [String]()
    
    //--------------------------------------------------------------------------
    // GRADIENT
    //--------------------------------------------------------------------------
    /// Gradient variable.
    var gradient = Gradient.Batch
    enum Gradient
    {
        case Batch
        case Sample
    }
    
    //--------------------------------------------------------------------------
    // MODEL
    //--------------------------------------------------------------------------
    /// Activation function factories.
    var activationKernels: [ActivationKernel] = [ActivationKernelImpl()]
    /// Serializable activation function registry.
    var activationRegistries: [[String: Codable.Type]] = [ACTIVATION_REGISTRY]
    
    /// Serializable layer registry.
    var layerRegistries: [[String: Codable.Type]] = [LAYER_REGISTRY]
    
    //--------------------------------------------------------------------------
    // LOOP
    //--------------------------------------------------------------------------
    /// Gradient Checking variable.
    var gc = false
    
    //--------------------------------------------------------------------------
    // DUMP
    //--------------------------------------------------------------------------
    /// Dirctory where to read data.
    var inputDir: URL! = nil
    /// Directory where to dump outputs.
    var outputDir: URL! = nil
    /// Dirctory wheeree to read and write models.
    var modeleDir: URL! = nil
    
    //--------------------------------------------------------------------------
    // TIME
    //--------------------------------------------------------------------------
    /// Time tracking variable.
    var trackTime = false
}

///
/// Dump content to the disk.
///
/// Throw an error when it is imposible to write the file.
///
/// - Parameters:
///     - directory: The directory where to write the file.
///     - name: The file name to dump the content into.
///     - content: The content to write on the disk.
///
func appendTxtFile(directory: URL, name: String, content: String) throws
{
    let outputFile = directory.appendingPathComponent("\(name).txt")
    if FileManager.default.fileExists(atPath: outputFile.path)
    {
        let fileHandle = try FileHandle(forWritingTo: outputFile)
        fileHandle.seekToEndOfFile()
        fileHandle.write(content.data(using: .utf8)!)
        fileHandle.closeFile()
    }
    else
    {
        try content.write(
            to: outputFile,
            atomically: true,
            encoding: String.Encoding.utf8
        )
    }
}
