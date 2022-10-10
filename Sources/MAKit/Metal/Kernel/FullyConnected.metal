//
//  FullyConnected.metal
//
//  Created by Jean-François Reboud on 25/07/2022.
//  Copyright © 2022 Jean-François Reboud. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void flForward(
    const device float * outsPrev,
    const device float * weights,
    const device float * biases,
    constant uint * pNbNeurones,
    constant uint * pNbNeuronesPrev,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbNeuronesPrev;
    uint nbBatch;
    
    if (pNbNeurones && pNbNeuronesPrev && pNbBatch &&
        outsPrev && weights && biases && outs)
    {
        nbNeurones = *pNbNeurones;
        nbNeuronesPrev = *pNbNeuronesPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurones || elem >= nbBatch)
    {
        return ;
    }
    
    float tmp = biases[depth];
    for (uint depthPrev=0; depthPrev<nbNeuronesPrev; depthPrev++)
    {
        uint offsetPrev = depthPrev + nbNeuronesPrev * elem;
        uint offsetWeights = depthPrev + nbNeuronesPrev * depth;
        
        float outPrev = outsPrev[offsetPrev];
        float w = weights[offsetWeights];
                
        tmp += outPrev * w;
    }
    
    uint offset = depth + nbNeurones * elem;
    outs[offset] = tmp;
}

kernel void flBackward(
    const device float * delta,
    const device float * weights,
    constant uint * pNbNeurones,
    constant uint * pNbNeuronesPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbNeuronesPrev;
    uint nbBatch;
    uint dirty;
    
    if (pNbNeurones && pNbNeuronesPrev && pNbBatch && pDirty &&
        deltaPrev && weights && delta)
    {
        nbNeurones = *pNbNeurones;
        nbNeuronesPrev = *pNbNeuronesPrev;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depthPrev = id[0];
    uint elem = id[1];
    
    if (depthPrev >= nbNeuronesPrev || elem >= nbBatch)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint depth=0; depth<nbNeurones; depth++)
    {
        uint offset = depth + nbNeurones * elem;
        uint offsetWeights = depthPrev + nbNeuronesPrev * depth;
        
        float deltaCur = delta[offset];
        float w = weights[offsetWeights];
        
        tmp += w * deltaCur;
    }
    
    uint offsetPrev = depthPrev + nbNeuronesPrev * elem;
    if (dirty)
    {
        deltaPrev[offsetPrev] = tmp;
    }
    else
    {
        deltaPrev[offsetPrev] += tmp;
    }
}

kernel void flBatchDerWeights(
    const device float * outsPrev,
    const device float * delta,
    constant uint * pNbNeurones,
    constant uint * pNbNeuronesPrev,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbNeuronesPrev;
    uint nbBatch;
    uint accumulate;
    
    if (pNbNeurones && pNbNeuronesPrev && pNbBatch && pAccumulate &&
        outsPrev && delta && grads)
    {
        nbNeurones = *pNbNeurones;
        nbNeuronesPrev = *pNbNeuronesPrev;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id[0];
    uint depthPrev = id[1];
    
    if (depth >= nbNeurones || depthPrev >= nbNeuronesPrev)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = depth + nbNeurones * elem;
        float deltaCur = delta[offset];
        
        uint offsetPrev = depthPrev + nbNeuronesPrev * elem;
        float outPrev = outsPrev[offsetPrev];
        
        tmp += deltaCur * outPrev;
    }
    
    uint offsetWeights = depthPrev + nbNeuronesPrev * depth;
    if (accumulate)
    {
        grads[offsetWeights] += tmp;
    }
    else
    {
        grads[offsetWeights] = tmp;
    }
}

kernel void flBatchDerBiases(
    const device float * delta,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    uint accumulate;
    
    if (pNbNeurones && pNbBatch && pAccumulate && delta && grads)
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
        uint offset = depth + nbNeurones * elem;
        tmp += delta[offset];
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

kernel void flDerWeights(
    const device float * outsPrev,
    const device float * delta,
    constant uint * pNbNeurones,
    constant uint * pNbNeuronesPrev,
    constant uint * pNbBatch,
    device float * deltaWeights,
    uint3 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbNeuronesPrev;
    uint nbBatch;
    
    if (pNbNeurones && pNbNeuronesPrev && pNbBatch &&
        outsPrev && delta && deltaWeights)
    {
        nbNeurones = *pNbNeurones;
        nbNeuronesPrev = *pNbNeuronesPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint depthPrev = id[1];
    uint elem = id[2];
    
    if (depth >= nbNeurones || depthPrev >= nbNeuronesPrev || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurones * elem;
    float deltaCur = delta[offset];
    
    uint offsetPrev = depthPrev + nbNeuronesPrev * elem;
    float outPrev = outsPrev[offsetPrev];
    
    float tmp = deltaCur * outPrev;
    
    uint offsetStartWeights = elem * nbNeurones * nbNeuronesPrev;
    uint offsetWeights = offsetStartWeights +
        depthPrev + nbNeuronesPrev * depth;
    deltaWeights[offsetWeights] = tmp;
}

kernel void flDerBiases(
    const device float * delta,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    
    if (pNbNeurones && pNbBatch && delta && deltaWeights)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurones || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurones * elem;
    float deltaCur = delta[offset];
    
    uint offsetWeights = elem * nbNeurones + depth;
    deltaWeights[offsetWeights] = deltaCur;
}

kernel void flReduceWeights(
    const device float * deltaWeights,
    constant uint * pNbNeurones,
    constant uint * pNbNeuronesPrev,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbNeuronesPrev;
    uint nbBatch;
    uint accumulate;
    
    if (pNbNeurones && pNbNeuronesPrev && pNbBatch && pAccumulate &&
        deltaWeights && grads)
    {
        nbNeurones = *pNbNeurones;
        nbNeuronesPrev = *pNbNeuronesPrev;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id[0];
    uint depthPrev = id[1];
    
    if (depth >= nbNeurones || depthPrev >= nbNeuronesPrev)
    {
        return ;
    }
    
    uint offsetWeights = depthPrev + nbNeuronesPrev * depth;
        
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = elem * nbNeurones * nbNeuronesPrev + offsetWeights;
        tmp += deltaWeights[offset];
    }
    
    if (accumulate)
    {
        grads[offsetWeights] += tmp;
    }
    else
    {
        grads[offsetWeights] = tmp;
    }
}
