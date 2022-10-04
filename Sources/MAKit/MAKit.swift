//
// MAKit.swift
// MAKit
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Foundation

public class MAKit
{
    private static var getCtx: MAKitContext
    {
        get {
            return MAKitContext.get
        }
    }
    private init() {}
    
    public class Opti
    {
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
    
    public class Gradient
    {
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
    
    public class Loop
    {
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
    
    public class Dump
    {
        public static var inputDir: String
        {
            get {
                return getCtx.inputDir
            }
            set {
                getCtx.inputDir = newValue
            }
        }
        public static var outputDir: String
        {
            get {
                return getCtx.outputDir
            }
            set {
                getCtx.outputDir = newValue
            }
        }
        public static var modeleDir: String
        {
            get {
                return getCtx.modeleDir
            }
            set {
                getCtx.modeleDir = newValue
            }
        }
    }
    
    public class Time
    {
        public static var track: Bool
        {
            get {
                return getCtx.trackTime
            }
            set {
                getCtx.trackTime = newValue
            }
        }
        
        public static func start()
        {
            if track
            {
                TimeTransaction.get.start()
            }
        }
        
        public static func stop(id: String, description: String) throws
        {
            if track
            {
                try TimeTransaction.get.stop(id: id, description: description)
            }
        }
        
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

fileprivate class MAKitContext
{
    private static let _ctx = MAKitContext()
    
    static var get: MAKitContext
    {
        get {
            return _ctx
        }
    }
    
    //--------------------------------------------------------------------------
    // OPTI
    //--------------------------------------------------------------------------
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
    var gradient = Gradient.Batch
    enum Gradient
    {
        case Batch
        case Sample
    }
    
    //--------------------------------------------------------------------------
    // LOOP
    //--------------------------------------------------------------------------
    var gc = false
    
    //--------------------------------------------------------------------------
    // DUMP
    //--------------------------------------------------------------------------
    var inputDir = ""
    var outputDir = ""
    var modeleDir = ""
    
    //--------------------------------------------------------------------------
    // TIME
    //--------------------------------------------------------------------------
    var trackTime = false
}

func appendTxtFile(directory: String, name: String, content: String) throws
{
    let outputFile = directory + "/" + name + ".txt"
    if FileManager.default.fileExists(atPath: outputFile)
    {
        let fileHandle = try FileHandle(
            forWritingTo: URL(fileURLWithPath: outputFile)
        )
        fileHandle.seekToEndOfFile()
        fileHandle.write(content.data(using: .utf8)!)
        fileHandle.closeFile()
    }
    else
    {
        try content.write(
            to: URL(fileURLWithPath: outputFile),
            atomically: true,
            encoding: String.Encoding.utf8
        )
    }
}
