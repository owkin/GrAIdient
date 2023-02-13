//
// Neurons.swift
// GrAIdient
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// Unit of a neural structure in the CPU execution context.
public class Neuron
{
    /// List of state variables of size: (batch size,).
    public var v = [Var]()
    /// List of gradient checking variables of size: (batch size, number of weight modifications).
    public var gc = [[GC]]()
    
    /// Number of different weigths for which we are estimating the gradient during Gradient Checking.
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
    
    /// State variables for forward and backward in CPU execution context.
    public struct Var
    {
        /// Output (result of the forward pass).
        public var out: Double = 0.0
        /// Temporary value (result of the forward pass before activation function).
        public var tmp: Double = 0.0
        /// Gradient (result of the backward pass).
        public var delta: Double = 0.0
    }
    /// Gradient Checking variables.
    public struct GC
    {
        /// Output (result of the forward gradient checking).
        public var out: Double = 0.0
    }
    
    /// Create one Neuron.
    public init() {}
    
    ///
    /// Initialize internal state variables for gradient checking.
    ///
    /// - Parameters:
    ///     - batchSize: The batch size of data.
    ///     - nbGC: Number of different weigths for which we are estimating the gradient
    ///     during Gradient Checking.
    ///
    public func initGC(batchSize: Int, nbGC: Int)
    {
        gc = [[GC]]()
        for _ in 0..<batchSize
        {
            let gcTmp = [GC](repeating: GC(), count: nbGC)
            gc.append(gcTmp)
        }
    }
    
    ///
    /// Initialize internal state variables.
    ///
    /// - Parameter batchSize: The batch size of data.
    ///
    public func initBatch(_ batchSize: Int)
    {
        v = [Var](repeating: Var(), count: batchSize)
    }
}

/// 1D shape collection of neurons.
public class EnsembleNeurons: Ensemble
{
    /// The unit type in the collection is a neuron.
    public typealias T = Neuron
    
    /// The list of internal neurons.
    var _neurons: [Neuron]
    /// Number of neurons.
    var _nbNeurons = 0
    
    /// Get the different neurons in the array.
    public var all: [Neuron]
    {
        return _neurons
    }
    
    ///
    /// Create an array of neurons.
    ///
    /// - Parameter nbNeurons: Number of neurons in the array.
    ///
    public init(_ nbNeurons: Int)
    {
        _nbNeurons = nbNeurons
        _neurons = [Neuron]()
        for _ in 0..<_nbNeurons
        {
            _neurons.append(Neuron())
        }
    }
    
    ///
    /// Initialize internal state variables for gradient checking.
    ///
    /// - Parameters:
    ///     - batchSize: The batch size of data.
    ///     - nbGC: Number of different weigths for which we are estimating the gradient
    ///     during Gradient Checking.
    ///
    public func initGC(batchSize: Int, nbGC: Int)
    {
        for neuron in _neurons
        {
            neuron.initGC(batchSize: batchSize, nbGC: nbGC)
        }
    }
    
    ///
    /// Initialize internal state variables.
    ///
    /// - Parameter batchSize: The batch size of data.
    ///
    public func initBatch(_ batchSize: Int)
    {
        for neuron in _neurons
        {
            neuron.initBatch(batchSize)
        }
    }
}

/// 2D shape collection of neurons.
public class GridNeurons: EnsembleNeurons, Grid
{
    /// The width of the grid.
    public let width: Int
    /// The height of the grid.
    public let height: Int
    
    ///
    /// Create a grid of neurons.
    ///
    /// - Parameters:
    ///     - width: The widht of the grid of neurons.
    ///     - height: The height of the grid of neurons.
    ///
    public init(width: Int, height: Int)
    {
        self.width = width
        self.height = height
        super.init(height * width)
    }
}