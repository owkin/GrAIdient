//
// Transaction.swift
// MAKit
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Foundation

class Transaction
{
    var _nbRunning = 0
    
    var isRunning: Bool
    {
        get {
            return _nbRunning > 0
        }
    }
}

public enum TimeError: Error
{
    case TrackTime
    case Transaction
}

extension TimeError: CustomStringConvertible
{
    public var description: String
    {
        switch self
        {
        case .TrackTime:
            return "Time is not being tracked."
        case .Transaction:
            return "No time transaction is running."
        }
    }
}

class TimeTransaction: Transaction
{
    static let get = TimeTransaction()
    
    var _startTime: DispatchTime? = nil
    var _stopTime: DispatchTime? = nil
    
    var _stackedTimes: [String: Double] = [:]
    
    var elpased: String
    {
        get {
            let nanoTime = _stopTime!.uptimeNanoseconds -
                           _startTime!.uptimeNanoseconds
            let timeSeconds = Double(nanoTime) / 1_000_000_000
            let timeMinutes = timeSeconds / 60.0
            
            if timeMinutes >= 1
            {
                return String(timeMinutes) + "min" + String(timeSeconds) + "s"
            }
            return String(timeSeconds) + "s"
        }
    }
    
    var stacked: [String: String]
    {
        get {
            var ret_dict: [String: String] = [:]
            for (identifier, timeSeconds) in _stackedTimes
            {
                var retVal: String = ""
                let timeMinutes = timeSeconds / 60.0
                
                if timeMinutes >= 1
                {
                    retVal = String(timeMinutes) + "min" +
                        String(timeSeconds) + "s"
                }
                else
                {
                    retVal = String(timeSeconds) + "s"
                }
                
                ret_dict[identifier] = retVal
            }
            return ret_dict
        }
    }
    
    func start()
    {
        if !isRunning
        {
            _nbRunning += 1
            _startTime = DispatchTime.now()
        }
    }
    
    func stop(id: String, description: String) throws
    {
        if isRunning
        {
            _nbRunning -= 1
            if !isRunning
            {
                _stopTime = DispatchTime.now()
                try _dumpTrack(id: id, description: description)
                _stackIdentifier(id: id)
            }
        }
        else
        {
            throw TimeError.Transaction
        }
    }
    
    private func _resetStack()
    {
        _stackedTimes = [:]
    }
    
    private func _stackIdentifier(id: String)
    {
        let nanoTime = _stopTime!.uptimeNanoseconds -
                       _startTime!.uptimeNanoseconds
        let timeSeconds = Double(nanoTime) / 1_000_000_000
        if let _ = _stackedTimes[id]
        {
            _stackedTimes[id]! += timeSeconds
        }
        else
        {
            _stackedTimes[id] = timeSeconds
        }
    }
    
    private func _dumpTrack(id: String, description: String) throws
    {
        var content = id + " " + description + "\n"
        content += elpased + "\n\n"
        try appendTxtFile(
            directory: MAKit.Dump.outputDir,
            name: "ElapsedTime",
            content: content
        )
    }
    
    func dumpStacked() throws
    {
        var content: String = ""
        for (id, time) in stacked
        {
            content += id + "\n"
            content += time + "\n\n"
        }
        content += "-------------------------------------------------\n"
        
        try appendTxtFile(
            directory: MAKit.Dump.outputDir,
            name: "StackedTime",
            content: content
        )
        _resetStack()
    }
}
