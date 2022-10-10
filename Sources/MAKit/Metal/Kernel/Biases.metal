//
//  Biases.metal
//
//  Created by Jean-François Reboud on 25/07/2022.
//  Copyright © 2022 Jean-François Reboud. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void reduceBiases(
    const device float * deltaWeights,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    uint accumulate;
    
    if (pNbNeurones && pNbBatch && pAccumulate && deltaWeights && grads)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id;
    if (depth >= nbNeurones)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = depth + elem * nbNeurones;
        tmp += deltaWeights[offset];
    }
    
    if (accumulate)
    {
        grads[depth] += tmp;
    }
    else
    {
        grads[depth] = tmp;
    }
}
