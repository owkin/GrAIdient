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
    
    public class Model
    {
        public struct Params
        {
            public var context: ModelContext
            public var hidden = false
            
            public init(context: ModelContext)
            {
                self.context = context
            }
            
            public init(params: Params)
            {
                self.context = params.context
                self.hidden = params.hidden
            }
        }
        
        public class Layer
        {
            public static func append(registry: [String: Codable.Type])
            {
                getCtx.layerRegistries.append(registry)
            }
            
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
        
        public class Activation
        {
            public static func append(kernel: IActivationKernel)
            {
                getCtx.activationKernels.append(kernel)
            }
            
            public static func append(registry: [String: Codable.Type])
            {
                getCtx.activationRegistries.append(registry)
            }
            
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
    var activationKernels: [IActivationKernel] = [ActivationKernel()]
    var activationRegistries: [[String: Codable.Type]] = [ACTIVATION_REGISTRY]
    
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
