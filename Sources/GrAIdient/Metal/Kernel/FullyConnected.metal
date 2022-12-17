//
// FullyConnected.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void flForward(
    const device float * outsPrev,
    const device float * weights,
    const device float * biases,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    
    if (pNbNeurons && pNbNeuronsPrev && pNbBatch &&
        outsPrev && weights && biases && outs)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    float tmp = biases[depth];
    for (uint depthPrev=0; depthPrev<nbNeuronsPrev; depthPrev++)
    {
        uint offsetPrev = depthPrev + nbNeuronsPrev * elem;
        uint offsetWeights = depthPrev + nbNeuronsPrev * depth;
        
        float outPrev = outsPrev[offsetPrev];
        float w = weights[offsetWeights];
                
        tmp += outPrev * w;
    }
    
    uint offset = depth + nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void flBackward(
    const device float * delta,
    const device float * weights,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint dirty;
    
    if (pNbNeurons && pNbNeuronsPrev && pNbBatch && pDirty &&
        deltaPrev && weights && delta)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depthPrev = id[0];
    uint elem = id[1];
    
    if (depthPrev >= nbNeuronsPrev || elem >= nbBatch)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * elem;
        uint offsetWeights = depthPrev + nbNeuronsPrev * depth;
        
        float deltaCur = delta[offset];
        float w = weights[offsetWeights];
        
        tmp += w * deltaCur;
    }
    
    uint offsetPrev = depthPrev + nbNeuronsPrev * elem;
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
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint accumulate;
    
    if (pNbNeurons && pNbNeuronsPrev && pNbBatch && pAccumulate &&
        outsPrev && delta && grads)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id[0];
    uint depthPrev = id[1];
    
    if (depth >= nbNeurons || depthPrev >= nbNeuronsPrev)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = depth + nbNeurons * elem;
        float deltaCur = delta[offset];
        
        uint offsetPrev = depthPrev + nbNeuronsPrev * elem;
        float outPrev = outsPrev[offsetPrev];
        
        tmp += deltaCur * outPrev;
    }
    
    uint offsetWeights = depthPrev + nbNeuronsPrev * depth;
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
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint accumulate;
    
    if (pNbNeurons && pNbBatch && pAccumulate && delta && grads)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id;
    if (depth >= nbNeurons)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = depth + nbNeurons * elem;
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
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    
    if (pNbNeurons && pNbNeuronsPrev && pNbBatch &&
        outsPrev && delta && deltaWeights)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem = id[0] / nbNeurons;
    uint depth = id[0] % nbNeurons;
    uint depthPrev = id[1];
    
    if (depth * elem >= nbNeurons * nbBatch ||
        depthPrev >= nbNeuronsPrev)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    float deltaCur = delta[offset];
    
    uint offsetPrev = depthPrev + nbNeuronsPrev * elem;
    float outPrev = outsPrev[offsetPrev];
    
    float tmp = deltaCur * outPrev;
    
    uint offsetStartWeights = elem * nbNeurons * nbNeuronsPrev;
    uint offsetWeights = offsetStartWeights +
        depthPrev + nbNeuronsPrev * depth;
    deltaWeights[offsetWeights] = tmp;
}

kernel void flDerBiases(
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && delta && deltaWeights)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    float deltaCur = delta[offset];
    
    uint offsetWeights = elem * nbNeurons + depth;
    deltaWeights[offsetWeights] = deltaCur;
}

kernel void flReduceWeights(
    const device float * deltaWeights,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint accumulate;
    
    if (pNbNeurons && pNbNeuronsPrev && pNbBatch && pAccumulate &&
        deltaWeights && grads)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id[0];
    uint depthPrev = id[1];
    
    if (depth >= nbNeurons || depthPrev >= nbNeuronsPrev)
    {
        return ;
    }
    
    uint offsetWeights = depthPrev + nbNeuronsPrev * depth;
        
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = elem * nbNeurons * nbNeuronsPrev + offsetWeights;
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
