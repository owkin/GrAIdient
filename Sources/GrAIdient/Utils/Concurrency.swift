//
// Concurrency.swift
// GrAIdient
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Foundation

///
/// Split an ensemble of elements into "balanced" batches.
///
/// - Parameters :
///     - nbElems: The number of elements in the ensemble.
///     - nbSplits: The number of batch splits.
/// - Returns: The list of (start, end) indices for the different batches.
///
func splitBatch(
    nbElems: Int, nbSplits: Int
) -> [(start: Int, end: Int)]
{
    var batchRanges = [(start: Int, end: Int)]()
    let batchSize = nbElems / nbSplits
    let remaining = nbElems % nbSplits
    
    var cur = 0
    for block in 0..<nbSplits
    {
        var batchSizeTmp = min(batchSize, nbElems - cur)
        if block < remaining
        {
            if batchSizeTmp != batchSize
            {
                fatalError()
            }
            batchSizeTmp += 1
        }
        batchRanges.append((start: cur, end: cur + batchSizeTmp))
        cur += batchSizeTmp
    }
    return batchRanges
}

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
        let nbThreads = min(
            nbElems, ProcessInfo.processInfo.activeProcessorCount
        )
        if nbThreads > 1
        {
            let batchRanges = splitBatch(nbElems: nbElems, nbSplits: nbThreads)
            DispatchQueue.concurrentPerform(iterations: nbThreads)
            {
                (thread: Int) in
                
                let range = batchRanges[thread]
                for elem in range.start..<range.end
                {
                    block(elem)
                }
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
