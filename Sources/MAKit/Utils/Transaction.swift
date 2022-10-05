//
// Transaction.swift
// MAKit
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Foundation

///
/// A context that goes from a start to an end.
///
/// Several start without an end do not create any new transaction.
/// They are just wrapped in the already existing transaction until it ends.
///
class Transaction
{
    var _nbRunning = 0
    
    /// Whether the transaction is running or not.
    var isRunning: Bool
    {
        get {
            return _nbRunning > 0
        }
    }
}

/// Error occuring during a Time transaction.
public enum TimeError: Error
{
    /// Time  is not being tracked.
    case TrackTime
    /// No time transaction is running.
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

/// A tarnsaction to handle time.
class TimeTransaction: Transaction
{
    /// Access the time transaction.
    static let get = TimeTransaction()
    
    var _startTime: DispatchTime? = nil
    var _stopTime: DispatchTime? = nil
    
    /// The elapsed time for the different functions.
    var _stackedTimes: [String: Double] = [:]
    
    /// A description showing the elapsed time between `_startTime` and `_stopTime`.
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
    
    /// A description showing the elapsed time for the different functions.
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
    
    /// Start the time transaction.
    func start()
    {
        if !isRunning
        {
            _nbRunning += 1
            _startTime = DispatchTime.now()
        }
    }
    
    ///
    /// Stop the time transaction.
    ///
    /// Throw an error when time is not being tracked.
    ///
    /// - parameters:
    ///     - id: The id of the function tracked.
    ///     - description: A short description of the function.
    ///
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
    
    /// Reset the elapsed time for the different functions.
    private func _resetStack()
    {
        _stackedTimes = [:]
    }
    
    ///
    /// Add elapsed time to a function.
    ///
    /// - parameter id: The id of the function tracked.
    ///
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
    
    ///
    /// Dump elapsed time.
    ///
    /// - parameters:
    ///     -  id: The id of the function tracked.
    ///     - description: A short description of the function.
    ///
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
    
    ///
    /// Dump aggreegated tracked time.
    ///
    /// Throw an error when time is not being tracked.
    ///
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
