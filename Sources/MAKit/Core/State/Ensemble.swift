//
// Ensemble.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

public protocol Ensemble
{
    associatedtype T
    
    var all: [T] { get }
    var nbElems: Int { get }
    func get(_ i: Int) -> T?
}

extension Ensemble
{
    public var nbElems: Int
    {
        return all.count
    }
    
    public func get(_ i: Int) -> T?
    {
        let elems = all
        if i >= 0 && i < elems.count
        {
            return elems[i]
        }
        return nil
    }
    
    public func set(_ i: Int, _ elem: T)
    {
        var elems = all
        if i >= 0 && i < elems.count
        {
            elems[i] = elem
        }
    }
}

public protocol Grid: Ensemble
{
    var width: Int { get }
    var height: Int { get }
    func get(_ i: Int, _ j: Int) -> T?
}

extension Grid
{
    //             j  0 ---------->[width-1,0]
    //0           i|
    //             |
    //             |
    //             |
    //             |
    //[0,height-1] v               [width-1,height-1]
    public func get(_ i: Int, _ j: Int) -> T?
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            return get(j + i * width)
        }
        return nil
    }
    
    public func set(_ i: Int, _ j: Int, _ elem: T)
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            set(j + i * width, elem)
        }
    }
}
