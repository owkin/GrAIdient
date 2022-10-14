//
// Sum2D.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

import MetalKit

public class Sum2D: LayerMerge2D
{
    public init(layersPrev: [Layer2D], params: MAKit.Model.Params)
    {
        let layer0 = layersPrev[0]
        for layerPrev in layersPrev
        {
            if layerPrev.nbFilters != layer0.nbFilters ||
               layerPrev.height != layer0.height ||
               layerPrev.width != layer0.width
            {
                fatalError("Layer structure error.")
            }
        }
        
        super.init(layersPrev: layersPrev,
                   nbFilters: layer0.nbFilters,
                   height: layer0.height,
                   width: layer0.width,
                   params: params)
    }
    
    public required init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    public override func copy(
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
        
        var layersPrev = [Layer2D]()
        for idPrev in _idsPrev
        {
            layersPrev.append(mapping[idPrev] as! Layer2D)
        }
        
        let layer = Sum2D(layersPrev: layersPrev, params: params)
        return layer
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for depth in 0..<nbFilters
        {
            for i in 0..<height {
            for j in 0..<width
            {
                neurones[depth].get(i, j)!.initGC(batchSize: batchSize,
                                                  nbGC: nbGC)
            }}
        }
        
        for batch in 0..<batchSize {
        for elem in 0..<nbSameElems {
        for depth in 0..<nbFilters
        {
            for i in 0..<height {
            for j in 0..<width
            {
                var sum = 0.0
                for num in 0..<_layersPrev.count
                {
                    let neuronesPrev = (_layersPrev[num] as! Layer2D).neurones
                    sum += neuronesPrev[depth].get(i, j)!.gc[batch][elem].out
                }
                
                neurones[depth].get(i, j)!.gc[batch][elem].out = sum
            }}
        }}}
    
        for batch in 0..<batchSize {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        for depth in 0..<nbFilters
        {
            for i in 0..<height {
            for j in 0..<width
            {
                var sum = 0.0
                for num in 0..<_layersPrev.count
                {
                    let neuronesPrev = (_layersPrev[num] as! Layer2D).neurones
                    
                    if num == index
                    {
                        sum += neuronesPrev[depth].get(i, j)!
                            .gc[batch][nbLastElems[index]+elem].out
                    }
                    else
                    {
                        sum += neuronesPrev[depth].get(i, j)!.v[batch].out
                    }
                }
                
                neurones[depth].get(i, j)!
                    .gc[batch][offset+elem].out = sum
            }}
        }}
            
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        for num in 0..<_layersPrev.count
        {
            MetalKernel.get.download([(_layersPrev[num] as! Layer2D).outs])
        }
        
        let (nbSameElems, layersIndex, nbElems) = getMergedGraph()
        
        var nbGC = nbSameElems
        for nbElemsTmp in nbElems
        {
            nbGC += nbElemsTmp
        }
        
        for depth in 0..<nbFilters
        {
            for i in 0..<height {
            for j in 0..<width
            {
                neurones[depth].get(i, j)!.initGC(batchSize: batchSize,
                                                  nbGC: nbGC)
            }}
        }
        
        for batch in 0..<batchSize {
        for elem in 0..<nbSameElems {
        for depth in 0..<nbFilters
        {
            for i in 0..<height {
            for j in 0..<width
            {
                var sum = 0.0
                for num in 0..<_layersPrev.count
                {
                    let neuronesPrev = (_layersPrev[num] as! Layer2D).neurones
                    sum += neuronesPrev[depth].get(i, j)!.gc[batch][elem].out
                }
                
                neurones[depth].get(i, j)!.gc[batch][elem].out = sum
            }}
        }}}
        
        for batch in 0..<batchSize {
        var offset = nbSameElems
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
        for (index, nbElemsTmp) in zip(layersIndex, nbElems) {
        for elem in 0..<nbElemsTmp {
        for depth in 0..<nbFilters
        {
            for i in 0..<height {
            for j in 0..<width
            {
                var sum = 0.0
                for num in 0..<_layersPrev.count
                {
                    let outsPrevPtr =
                        (_layersPrev[num] as! Layer2D).outs.shared.buffer
                    let neuronesPrev =
                        (_layersPrev[num] as! Layer2D).neurones
                    
                    if num == index
                    {
                        sum += neuronesPrev[depth].get(i, j)!
                            .gc[batch][nbLastElems[index]+elem].out
                    }
                    else
                    {
                        let offsetStart = (depth + nbFilters * batch) * height
                        let offsetTmp = j + (offsetStart + i) * width
                        
                        sum += Double(outsPrevPtr[offsetTmp])
                    }
                }
                
                neurones[depth].get(i, j)!.gc[batch][offset+elem].out = sum
            }}
        }}
            
        offset += nbElemsTmp
        nbLastElems[index] += nbElemsTmp
        }}
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        for elem in 0..<batchSize {
        for depth in 0..<nbFilters
        {
            for i in 0..<height {
            for j in 0..<width
            {
                var sum = 0.0
                for num in 0..<_layersPrev.count
                {
                    let neuronesPrev =
                        (_layersPrev[num] as! Layer2D).neurones
                    sum += neuronesPrev[depth].get(i, j)!.v[elem].out
                }
                
                neurones[depth].get(i, j)!.v[elem].out = sum
            }}
        }}
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        try checkStateForwardGPU(batchSize: batchSize)
        
        var first = true
        for num in 0..<_layersPrev.count
        {
            let nbElems = (_layersPrev[num] as! Layer2D).outs.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command: MetalCommand
            if first
            {
                command = MetalKernel.get.createCommand(
                    "sum1", deviceID: deviceID)
                first = false
            }
            else
            {
                command = MetalKernel.get.createCommand(
                    "sum2", deviceID: deviceID)
            }
            
            command.setBuffer((_layersPrev[num] as! Layer2D).outs.metal,
                              atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer(outs.metal, atIndex: 2)
            
            let threads = command.threadExecutionWidth
            let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
            let threadsPerGrid = MTLSize(width: nbElems,
                                         height: 1,
                                         depth: 1)
            command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                               threadsPerThreadgroup: threadsPerThreadgroup)
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if !mustComputeBackward
        {
            return
        }
        
        for num in 0..<_layersPrev.count
        {
            if !_layersPrev[num].computeDelta
            {
                continue
            }
            
            let neuronesPrev = (_layersPrev[num] as! Layer2D).neurones
            for elem in 0..<batchSize {
            for depth in 0..<nbFilters
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let deltaCur = neurones[depth].get(i, j)!.v[elem].delta
                    
                    if _layersPrev[num].dirty
                    {
                        neuronesPrev[depth].get(i, j)!.v[elem].delta =
                            deltaCur
                    }
                    else
                    {
                        neuronesPrev[depth].get(i, j)!.v[elem].delta +=
                            deltaCur
                    }
                }}
            }}
        }
        
        propagateDirty()
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        if !mustComputeBackward
        {
            return
        }
        
        for num in 0..<_layersPrev.count
        {
            if !_layersPrev[num].computeDelta
            {
                continue
            }
            
            try (_layersPrev[num] as! Layer2D).checkStateBackwardGPU(
                batchSize: batchSize
            )
            
            let nbElems = delta.nbElems
            let pNbElems: [UInt32] = [UInt32(nbElems)]
            
            let command: MetalCommand
            if _layersPrev[num].dirty
            {
                command = MetalKernel.get.createCommand(
                    "sum1", deviceID: deviceID)
            }
            else
            {
                command = MetalKernel.get.createCommand(
                    "sum2", deviceID: deviceID)
            }
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbElems, atIndex: 1)
            command.setBuffer((_layersPrev[num] as! Layer2D).delta.metal,
                              atIndex: 2)
            
            let threads = command.threadExecutionWidth
            let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
            let threadsPerGrid = MTLSize(width: nbElems,
                                         height: 1,
                                         depth: 1)
            command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                               threadsPerThreadgroup: threadsPerThreadgroup)
            command.enqueue()
        }
        
        propagateDirty()
    }
}
