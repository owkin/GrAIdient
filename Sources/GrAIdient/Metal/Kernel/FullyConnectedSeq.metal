//
// FullyConnectedSeq.metal
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

kernel void flSeq4Forward(
    const device float4 * outsPrev,
    const device float4 * weights,
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
    
    if (depth >= nbNeurons || elem * 4 >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    float4 tmp [4] = {0};
    for (uint depthPrev=0; depthPrev<nbNeuronsPrev/4; depthPrev++)
    {
        uint offsetWeights = (depthPrev * 4 + nbNeuronsPrev * depth) / 4;
        float4 w = weights[offsetWeights];
        
        for (uint i=0; i<4; i++)
        {
            uint offsetPrev = (depthPrev * 4 + nbNeuronsPrev * seq +
                sequence * nbNeuronsPrev * (elem*4+i)) / 4;
            float4 outPrev = outsPrev[offsetPrev];
            
            tmp[i] += outPrev * w;
        }
    }
    
    float bias = biases[depth];
    for (uint i=0; i<4; i++)
    {
        uint offset = depth + nbNeurons * seq +
            sequence * nbNeurons * (elem*4+i);
        outs[offset] = tmp[i][0] + tmp[i][1] + tmp[i][2] + tmp[i][3] + bias;
    }
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

kernel void flSeq4Backward(
    const device float * delta,
    const device float4 * weights,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float4 * deltaPrev,
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
    
    if (depthPrev * 4 >= nbNeuronsPrev ||
        elem * 4 >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    float4 tmp [4] = {0};
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offsetWeights = (depthPrev * 4 + nbNeuronsPrev * depth) / 4;
        float4 w = weights[offsetWeights];
        
        for (uint i=0; i<4; i++)
        {
            uint offset = depth + nbNeurons * seq +
                sequence * nbNeurons * (elem*4+i);
            float deltaCur = delta[offset];
            
            tmp[i] += w * deltaCur;
        }
    }
    
    if (dirty)
    {
        for (uint i=0; i<4; i++)
        {
            uint offsetPrev = (depthPrev * 4 + nbNeuronsPrev * seq +
                sequence * nbNeuronsPrev * (elem*4+i)) / 4;
            deltaPrev[offsetPrev] = tmp[i];
        }
    }
    else
    {
        for (uint i=0; i<4; i++)
        {
            uint offsetPrev = (depthPrev * 4 + nbNeuronsPrev * seq +
                sequence * nbNeuronsPrev * (elem*4+i)) / 4;
            deltaPrev[offsetPrev] += tmp[i];
        }
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

kernel void flSeqBatch4DerWeights(
    const device float4 * outsPrev,
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pAccumulate,
    device float4 * grads,
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
    
    if (depth >= nbNeurons || depthPrev * 4 >= nbNeuronsPrev)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++) {
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        float deltaCur = delta[offset];
        
        uint offsetPrev = (depthPrev * 4 + nbNeuronsPrev * seq +
            sequence * nbNeuronsPrev * elem) / 4;
        float4 outPrev = outsPrev[offsetPrev];
        
        tmp += outPrev * deltaCur;
    }}
    
    uint offsetWeights = (depthPrev * 4 + nbNeuronsPrev * depth) / 4;
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
