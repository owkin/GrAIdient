//
// Reset.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void resetFloat(
    constant uint * pNbElems,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems && outs)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    outs[id] = 0.0;
}
