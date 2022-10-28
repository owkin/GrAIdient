//
// Ensemble.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// A collection of elements.
public protocol Ensemble
{
    /// The unit type in the collection.
    associatedtype T
    
    /// Get the different elements as an array.
    var all: [T] { get }
    /// Number of elements in the collection.
    var nbElems: Int { get }
    
    ///
    /// Get one element.
    ///
    /// - Parameter i: The index of the element to retrieve.
    /// - Returns: The element retrieved.
    ///
    func get(_ i: Int) -> T?
}

extension Ensemble
{
    /// Number of elements in the collection.
    public var nbElems: Int
    {
        return all.count
    }
    
    ///
    /// Get one element.
    ///
    /// - Parameter i: The index of the element to retrieve.
    /// - Returns: The element retrieved.
    ///
    public func get(_ i: Int) -> T?
    {
        let elems = all
        if i >= 0 && i < elems.count
        {
            return elems[i]
        }
        return nil
    }
    
    ///
    /// Set one element.
    ///
    /// - Parameters
    ///     - i: The index of the element to set.
    ///     - elem: The value to set.
    ///
    public func set(_ i: Int, _ elem: T)
    {
        var elems = all
        if i >= 0 && i < elems.count
        {
            elems[i] = elem
        }
    }
}

/// A collection of elements, shaped in a grid.
public protocol Grid: Ensemble
{
    //             j  0 ---------->[width-1,0]
    //0           i|
    //             |
    //             |
    //             |
    //             |
    //[0,height-1] v               [width-1,height-1]
    
    /// The width of the grid.
    var width: Int { get }
    /// The height of the grid.
    var height: Int { get }
    
    ///
    /// Get one element with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row of the element to retrieve.
    ///     - j: The column of the element to retrieve.
    /// - Returns: The element retrieved.
    ///
    func get(_ i: Int, _ j: Int) -> T?
}

extension Grid
{
    ///
    /// Get one element with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row of the element to retrieve.
    ///     - j: The column of the element to retrieve.
    /// - Returns: The element retrieved.
    ///
    public func get(_ i: Int, _ j: Int) -> T?
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            return get(j + i * width)
        }
        return nil
    }
    
    ///
    /// Set one element with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row of the element to retrieve.
    ///     - j: The column of the element to retrieve.
    ///     - elem: The value to set.
    ///
    public func set(_ i: Int, _ j: Int, _ elem: T)
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            set(j + i * width, elem)
        }
    }
}
