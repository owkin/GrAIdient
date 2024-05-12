//
// LayerMerge.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void sum1Half(
    const device half * ins,
    constant uint * pNbElems,
    device half * outs,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems && ins && outs)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    outs[id] = ins[id];
}

kernel void sum14Half(
    const device half4 * ins,
    constant uint * pNbElems,
    device half4 * outs,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems && ins && outs)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id * 4 >= nbElems)
    {
        return ;
    }
    
    outs[id] = ins[id];
}

kernel void sum2Half(
    const device half * ins,
    constant uint * pNbElems,
    device half * outs,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems && ins && outs)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    outs[id] += ins[id];
}

kernel void sum24Half(
    const device half4 * ins,
    constant uint * pNbElems,
    device half4 * outs,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems && ins && outs)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id * 4 >= nbElems)
    {
        return ;
    }
    
    outs[id] += ins[id];
}

kernel void multiplyForwardHalf(
    const device half * outsPrev,
    constant uint * pNbElems,
    device half * outs,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems && outsPrev && outs)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    outs[id] *= outsPrev[id];
}

kernel void multiplyBackwardHalf(
    const device half * outs,
    const device half * delta,
    constant uint * pNbElems,
    constant uint * pDirty,
    device half * deltaPrev,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    uint dirty;
    
    if (pNbElems && pDirty && outs && delta && deltaPrev)
    {
        nbElems = pNbElems[0];
        dirty = *pDirty;
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float tmp = outs[id];
    float deltaCur = delta[id];
    
    if (dirty)
    {
        deltaPrev[id] = deltaCur * tmp;
    }
    else
    {
        deltaPrev[id] += deltaCur * tmp;
    }
}
