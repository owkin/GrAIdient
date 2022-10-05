//
// Concurrency.swift
// MAKit
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Foundation

/// A namespace  to execute functions in parallel.
public class Concurrency
{
    ///
    /// Execute the same functions on multiple elements.
    ///
    /// - Parameters:
    ///     - nbElems: Number of elements to process.
    ///     - block: The function to execute on the different elements.
    ///     
    public static func slice(_ nbElems: Int, _ block: (Int) -> ())
    {
        let nbThreads = ProcessInfo.processInfo.activeProcessorCount
        if nbElems >= nbThreads
        {
            DispatchQueue.concurrentPerform(iterations: nbThreads)
            {
                (thread: Int) in
                
                let nbElemsPerThread = nbElems / nbThreads
                let start = thread * nbElemsPerThread
                let end = min(nbElems, (thread+1) * nbElemsPerThread)
                
                for elem in start..<end
                {
                    block(elem)
                }
            }
        }
        else if nbElems > 1
        {
            DispatchQueue.concurrentPerform(iterations: nbElems)
            {
                (thread: Int) in
                block(thread)
            }
        }
        else if nbElems == 1
        {
            block(0)
        }
        else if nbElems == 0
        {
            return
        }
        else if nbElems < 0
        {
            fatalError("'nbElems' should be positive, received \(nbElems).")
        }
    }
}
