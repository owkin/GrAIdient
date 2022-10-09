//
// Neurones.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

public class Neurone
{
    public var v = [Var]()
    public var gc = [[GC]]()
    
    public var nbGC: Int
    {
        get {
            if gc.count == 0
            {
                return 0
            }
            return gc[0].count
        }
    }
    
    public struct Var
    {
        public var out: Double = 0.0
        public var tmp: Double = 0.0
        public var delta: Double = 0.0
    }
    public struct GC
    {
        public var out: Double = 0.0
    }
    
    public init() {}
    
    public func initGC(batchSize: Int, nbGC: Int)
    {
        gc = [[GC]]()
        for _ in 0..<batchSize
        {
            let gcTmp = [GC](repeating: GC(), count: nbGC)
            gc.append(gcTmp)
        }
    }
    
    public func initBatch(_ batchSize: Int)
    {
        v = [Var](repeating: Var(), count: batchSize)
    }
}

public class EnsembleNeurones: Ensemble
{
    public typealias T = Neurone
    
    var _neurones: [Neurone]
    var _nbNeurones = 0
    
    public var all: [Neurone]
    {
        return _neurones
    }
    
    public init(_ nbNeurones: Int)
    {
        _nbNeurones = nbNeurones
        _neurones = [Neurone]()
        for _ in 0..<_nbNeurones
        {
            _neurones.append(Neurone())
        }
    }
    
    public func initGC(batchSize: Int, nbGC: Int)
    {
        for neurone in _neurones
        {
            neurone.initGC(batchSize: batchSize, nbGC: nbGC)
        }
    }
    
    public func initBatch(_ batchSize: Int)
    {
        for neurone in _neurones
        {
            neurone.initBatch(batchSize)
        }
    }
}

public class GridNeurones: EnsembleNeurones, Grid
{
    public let width: Int
    public let height: Int
    
    public init(width: Int, height: Int)
    {
        self.width = width
        self.height = height
        super.init(height * width)
    }
}
