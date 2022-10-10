//
//  Reset.metal
//
//  Created by Jean-François Reboud on 25/07/2022.
//  Copyright © 2022 Jean-François Reboud. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void reset(
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
