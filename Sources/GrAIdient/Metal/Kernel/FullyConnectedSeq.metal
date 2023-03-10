//
// FullyConnectedPatch.metal
// GrAIdient
//
// Created by Jean-François Reboud on 08/03/2023.
//

#include <metal_stdlib>
using namespace metal;

kernel void flSeqForward(
    const device float * outsPrev,
    const device float * weights,
    const device float * biases,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbNeuronsPrev && pNbBatch && pSequence &&
        outsPrev && weights && biases && outs)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depth >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    float tmp = biases[depth];
    for (uint depthPrev=0; depthPrev<nbNeuronsPrev; depthPrev++)
    {
        uint offsetPrev = depthPrev + nbNeuronsPrev * seq +
            sequence * nbNeuronsPrev * elem;
        float outPrev = outsPrev[offsetPrev];
        
        uint offsetWeights = depthPrev + nbNeuronsPrev * depth;
        float w = weights[offsetWeights];
        
        tmp += outPrev * w;
    }
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void flSeqBackward(
    const device float * delta,
    const device float * weights,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint dirty;
    
    if (pNbNeurons && pNbNeuronsPrev && pNbBatch && pDirty &&
        deltaPrev && weights && delta)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depthPrev = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depthPrev >= nbNeuronsPrev || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offsetWeights = depthPrev + nbNeuronsPrev * depth;
        float w = weights[offsetWeights];
        
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        float deltaCur = delta[offset];
        
        tmp += w * deltaCur;
    }
    
    uint offsetPrev = depthPrev + nbNeuronsPrev * seq +
        sequence * nbNeuronsPrev * elem;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = tmp;
    }
    else
    {
        deltaPrev[offsetPrev] += tmp;
    }
}

kernel void flSeqBatchDerWeights(
    const device float * outsPrev,
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint accumulate;
    
    if (pNbNeurons && pNbNeuronsPrev && pNbBatch && pSequence && pAccumulate &&
        outsPrev && delta && grads)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
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
    for (uint elem=0; elem<nbBatch; elem++) {
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        float deltaCur = delta[offset];
        
        uint offsetPrev = depthPrev + nbNeuronsPrev * seq +
            sequence * nbNeuronsPrev * elem;
        float outPrev = outsPrev[offsetPrev];
        
        tmp += outPrev * deltaCur;
    }}
    
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

kernel void flSeqDerWeights(
    const device float * outsPrev,
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbNeuronsPrev && pNbBatch && pSequence &&
        outsPrev && delta && deltaWeights)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint depth = id[0] % nbNeurons;
    uint depthPrev = id[1];
    uint elem = id[0] / nbNeurons;
    
    if (depth * elem >= nbNeurons * nbBatch ||
        depthPrev >= nbNeuronsPrev)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        float deltaCur = delta[offset];
        
        uint offsetPrev = depthPrev + nbNeuronsPrev * seq +
            sequence * nbNeuronsPrev * elem;
        float outPrev = outsPrev[offsetPrev];
        
        tmp += outPrev * deltaCur;
    }
    
    uint offsetStartWeights = elem * nbNeurons * nbNeuronsPrev;
    uint offsetWeights = offsetStartWeights + depthPrev + nbNeuronsPrev * depth;
    deltaWeights[offsetWeights] = tmp;
}

kernel void flSeqReduceWeights(
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