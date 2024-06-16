//
// FullyConnectedSeq.metal
// GrAIdient
//
// Created by Jean-François Reboud on 08/03/2023.
//

#include <metal_stdlib>
using namespace metal;

kernel void flSeqForwardHalf(
    const device half * outsPrev,
    const device half * weights,
    const device half * biases,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrev,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depth >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    half tmp = biases[depth];
    uint offsetPrev = nbNeuronsPrev * seq + sequence * nbNeuronsPrev * elem;
    uint offsetWeights = nbNeuronsPrev * depth;
    
    for (uint depthPrev=0; depthPrev<nbNeuronsPrev; depthPrev++)
    {
        half outPrev = outsPrev[depthPrev + offsetPrev];
        half w = weights[depthPrev + offsetWeights];
        
        tmp += outPrev * w;
    }
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void flSeq48ForwardHalf(
    const device half4 * outsPrev,
    const device half4 * weights,
    const device half * biases,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrev,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint coeff = 8;
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depth >= nbNeurons || elem * coeff >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    half4 tmp[8] = {0};
    uint offsetPrev = nbNeuronsPrev * seq / 4;
    uint offsetWeights = (nbNeuronsPrev * depth) / 4;
    
    for (uint depthPrev=0; depthPrev<nbNeuronsPrev/4; depthPrev++)
    {
        half4 w = weights[depthPrev + offsetWeights];
        uint offsetPrevTmp1 = depthPrev + offsetPrev;
        
        for (uint i=0; i<coeff; i++)
        {
            uint offsetPrevTmp2 = offsetPrevTmp1 +
                sequence * nbNeuronsPrev * (elem*coeff+i) / 4;
            half4 outPrev = outsPrev[offsetPrevTmp2];
            
            tmp[i] += outPrev * w;
        }
    }
    
    half bias = biases[depth];
    uint offset = depth + nbNeurons * seq;
    
    for (uint i=0; i<coeff; i++)
    {
        uint offsetTmp = offset + sequence * nbNeurons * (elem*coeff+i);
        outs[offsetTmp] = tmp[i][0] + tmp[i][1] + tmp[i][2] + tmp[i][3] + bias;
    }
}

kernel void flSeq4ForwardHalf(
    const device half4 * outsPrev,
    const device half4 * weights,
    const device half * biases,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrev,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depth >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    half4 tmp = 0;
    uint offsetPrev =
        (nbNeuronsPrev * seq + sequence * nbNeuronsPrev * elem) / 4;
    uint offsetWeights = (nbNeuronsPrev * depth) / 4;
    
    for (uint depthPrev=0; depthPrev<nbNeuronsPrev/4; depthPrev++)
    {
        uint offsetPrevTmp = depthPrev + offsetPrev;
        half4 outPrev = outsPrev[offsetPrevTmp];
        
        uint offsetWeightsTmp = depthPrev + offsetWeights;
        half4 w = weights[offsetWeightsTmp];
        
        tmp += outPrev * w;
    }
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    outs[offset] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + biases[depth];
}

kernel void flSeqBackwardHalf(
    const device half * delta,
    const device half * weights,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrev,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint depthPrev = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depthPrev >= nbNeuronsPrev || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    half tmp = 0.0;
    uint offset = nbNeurons * seq + sequence * nbNeurons * elem;
    
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offsetWeights = depthPrev + nbNeuronsPrev * depth;
        half w = weights[offsetWeights];
        
        uint offsetTmp = depth + offset;
        half deltaCur = delta[offsetTmp];
        
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

kernel void flSeq48BackwardHalf(
    const device half * delta,
    const device half4 * weights,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrev,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half4 * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint coeff = 8;
    uint depthPrev = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depthPrev * 4 >= nbNeuronsPrev ||
        elem * coeff >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    half4 tmp[8] = {0};
    uint offset = nbNeurons * seq;
    
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offsetTmp1 = depth + offset;
        uint offsetWeights = (depthPrev * 4 + nbNeuronsPrev * depth) / 4;
        half4 w = weights[offsetWeights];
        
        for (uint i=0; i<coeff; i++)
        {
            uint offsetTmp2 = offsetTmp1 +
                sequence * nbNeurons * (elem*coeff+i);
            half deltaCur = delta[offsetTmp2];
            
            tmp[i] += w * deltaCur;
        }
    }
    
    uint offsetPrev = depthPrev + (nbNeuronsPrev * seq) / 4;
    if (dirty)
    {
        for (uint i=0; i<coeff; i++)
        {
            uint offsetPrevTmp = offsetPrev +
                sequence * nbNeuronsPrev * (elem*coeff+i) / 4;
            deltaPrev[offsetPrevTmp] = tmp[i];
        }
    }
    else
    {
        for (uint i=0; i<coeff; i++)
        {
            uint offsetPrevTmp = offsetPrev +
                sequence * nbNeuronsPrev * (elem*coeff+i) / 4;
            deltaPrev[offsetPrevTmp] += tmp[i];
        }
    }
}

kernel void flSeq4BackwardHalf(
    const device half * delta,
    const device half4 * weights,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrev,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half4 * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint depthPrev = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depthPrev * 4 >= nbNeuronsPrev || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    half4 tmp = 0.0;
    uint offset = nbNeurons * seq + sequence * nbNeurons * elem;
    
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offsetWeights = depthPrev + nbNeuronsPrev * depth / 4;
        half4 w = weights[offsetWeights];
        
        uint offsetTmp = depth + offset;
        half deltaCur = delta[offsetTmp];
        
        tmp += w * deltaCur;
    }
    
    uint offsetPrev = (depthPrev * 4 + nbNeuronsPrev * seq +
        sequence * nbNeuronsPrev * elem) / 4;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = tmp;
    }
    else
    {
        deltaPrev[offsetPrev] += tmp;
    }
}

kernel void flSeqBatchDerWeightsHalf(
    const device half * outsPrev,
    const device half * delta,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrev,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & accumulate,
    device half * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint depth = id[0];
    uint depthPrev = id[1];
    
    if (depth >= nbNeurons || depthPrev >= nbNeuronsPrev)
    {
        return ;
    }
    
    half tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++) {
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        half deltaCur = delta[offset];
        
        uint offsetPrev = depthPrev + nbNeuronsPrev * seq +
            sequence * nbNeuronsPrev * elem;
        half outPrev = outsPrev[offsetPrev];
        
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

kernel void flSeqBatch4DerWeightsHalf(
    const device half4 * outsPrev,
    const device half * delta,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrev,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & accumulate,
    device half4 * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint depth = id[0];
    uint depthPrev = id[1];
    
    if (depth >= nbNeurons || depthPrev * 4 >= nbNeuronsPrev)
    {
        return ;
    }
    
    half4 tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++) {
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        half deltaCur = delta[offset];
        
        uint offsetPrev = depthPrev + (nbNeuronsPrev * seq +
            sequence * nbNeuronsPrev * elem) / 4;
        half4 outPrev = outsPrev[offsetPrev];
        
        tmp += outPrev * deltaCur;
    }}
    
    uint offsetWeights = depthPrev + (nbNeuronsPrev * depth) / 4;
    if (accumulate)
    {
        grads[offsetWeights] += tmp;
    }
    else
    {
        grads[offsetWeights] = tmp;
    }
}

kernel void flSeqDerWeightsHalf(
    const device half * outsPrev,
    const device half * delta,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrev,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint depth = id[0] % nbNeurons;
    uint depthPrev = id[1];
    uint elem = id[0] / nbNeurons;
    
    if (depth * elem >= nbNeurons * nbBatch ||
        depthPrev >= nbNeuronsPrev)
    {
        return ;
    }
    
    half tmp = 0.0;
    uint offset = depth + sequence * nbNeurons * elem;
    uint offsetPrev = depthPrev + sequence * nbNeuronsPrev * elem;
    
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offsetTmp = nbNeurons * seq + offset;
        half deltaCur = delta[offsetTmp];
        
        uint offsetPrevTmp = nbNeuronsPrev * seq + offsetPrev;
        half outPrev = outsPrev[offsetPrevTmp];
        
        tmp += outPrev * deltaCur;
    }
    
    uint offsetStartWeights = elem * nbNeurons * nbNeuronsPrev;
    uint offsetWeights = offsetStartWeights + depthPrev + nbNeuronsPrev * depth;
    deltaWeights[offsetWeights] = tmp;
}

kernel void flSeqReduceWeightsHalf(
    const device half * deltaWeights,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrev,
    constant uint & nbBatch,
    constant uint & accumulate,
    device half * grads,
    uint2 id [[ thread_position_in_grid ]])
{
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
