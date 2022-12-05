//
// Sum.metal
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void sum1(
    const device float * ins,
    constant uint * pNbElems,
    device float * outs,
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

kernel void sum2(
    const device float * ins,
    constant uint * pNbElems,
    device float * outs,
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

kernel void multiplyForward(
    const device float * outsPrev,
    constant uint * pNbElems,
    device float * outs,
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

kernel void multiplyBackward(
    const device float * outs,
    const device float * delta,
    constant uint * pNbElems,
    constant uint * pDirty,
    device float * deltaPrev,
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
