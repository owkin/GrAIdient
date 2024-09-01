//
// FullyConnected.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void flForwardHalf(
    const device half * outsPrev,
    const device half * weights,
    const device half * biases,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    device half * outs,
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
    
    half tmp = biases[depth];
    for (uint depthPrev=0; depthPrev<nbNeuronsPrev; depthPrev++)
    {
        uint offsetPrev = depthPrev + nbNeuronsPrev * elem;
        uint offsetWeights = depthPrev + nbNeuronsPrev * depth;
        
        half outPrev = outsPrev[offsetPrev];
        half w = weights[offsetWeights];
                
        tmp += outPrev * w;
    }
    
    uint offset = depth + nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void flBackwardHalf(
    const device half * delta,
    const device half * weights,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
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
    
    half tmp = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * elem;
        uint offsetWeights = depthPrev + nbNeuronsPrev * depth;
        
        half deltaCur = delta[offset];
        half w = weights[offsetWeights];
        
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

kernel void flBatchDerWeightsHalf(
    const device half * outsPrev,
    const device half * delta,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device half * grads,
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
    
    half tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = depth + nbNeurons * elem;
        half deltaCur = delta[offset];
        
        uint offsetPrev = depthPrev + nbNeuronsPrev * elem;
        half outPrev = outsPrev[offsetPrev];
        
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

kernel void flBatchDerBiasesHalf(
    const device half * delta,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device half * grads,
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
    
    half tmp = 0.0;
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

kernel void flDerWeightsHalf(
    const device half * outsPrev,
    const device half * delta,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    device half * deltaWeights,
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
    half deltaCur = delta[offset];
    
    uint offsetPrev = depthPrev + nbNeuronsPrev * elem;
    half outPrev = outsPrev[offsetPrev];
    
    half tmp = deltaCur * outPrev;
    
    uint offsetStartWeights = elem * nbNeurons * nbNeuronsPrev;
    uint offsetWeights = offsetStartWeights +
        depthPrev + nbNeuronsPrev * depth;
    deltaWeights[offsetWeights] = tmp;
}

kernel void flDerBiasesHalf(
    const device half * delta,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device half * deltaWeights,
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
    half deltaCur = delta[offset];
    
    uint offsetWeights = elem * nbNeurons + depth;
    deltaWeights[offsetWeights] = deltaCur;
}

kernel void flReduceWeightsHalf(
    const device half * deltaWeights,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device half * grads,
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
        
    half tmp = 0.0;
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
