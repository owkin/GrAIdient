//
// Layer1D.metal
// GrAIdient
//
// Created by Jean-François Reboud on 27/02/2023.
//

#include <metal_stdlib>
using namespace metal;

kernel void avgPoolSeqForward(
    const device float * outsPrev,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence &&
        outsPrev && outs)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offsetPrev = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        tmp += outsPrev[offsetPrev];
    }
    tmp /= sequence;
    
    uint offset = depth + nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void avgPoolSeqBackward(
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint dirty;
    
    if (pNbNeurons && pNbBatch && pSequence && pDirty &&
        delta && deltaPrev)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
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
    
    uint offset = depth + nbNeurons * elem;
    float deltaCur = delta[offset];
    
    uint offsetPrev = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    if (dirty)
    {
        deltaPrev[offsetPrev] = deltaCur / sequence;
    }
    else
    {
        deltaPrev[offsetPrev] += deltaCur / sequence;
    }
}

kernel void concat1SeqForward(
    const device float * outsPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pSequencePrev,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint sequencePrev;
    uint globalOffset;
    
    if (pGlobalOffset && pNbNeurons &&
        pNbBatch && pSequence && pSequencePrev && outsPrev && outs)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        sequencePrev = *pSequencePrev;
        globalOffset = *pGlobalOffset;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequencePrev;
    uint seq = id[1] % sequencePrev;
    
    if (depth >= nbNeurons || elem >= nbBatch || seq >= sequencePrev)
    {
        return ;
    }
    
    uint offsetPrev = depth +
        nbNeurons * seq + sequencePrev * nbNeurons * elem;
    uint offset = depth +
        nbNeurons * (globalOffset+seq) + sequence * nbNeurons * elem;
    
    outs[offset] = outsPrev[offsetPrev];
}

kernel void concat1SeqBackward(
    const device float * delta,
    constant uint * pGlobalOffset,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pSequencePrev,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint sequencePrev;
    uint globalOffset;
    uint dirty;
    
    if (pGlobalOffset && pNbNeurons &&
        pNbBatch && pSequence && pSequencePrev && pDirty && deltaPrev && delta)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        sequencePrev = *pSequencePrev;
        globalOffset = *pGlobalOffset;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequencePrev;
    uint seq = id[1] % sequencePrev;
    
    if (depth >= nbNeurons || elem >= nbBatch || seq >= sequencePrev)
    {
        return ;
    }
    
    uint offsetPrev = depth +
        nbNeurons * seq + sequencePrev * nbNeurons * elem;
    uint offset = depth +
        nbNeurons * (globalOffset+seq) + sequence * nbNeurons * elem;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = delta[offset];
    }
    else
    {
        deltaPrev[offsetPrev] += delta[offset];
    }
}

kernel void concat2SeqForward(
    const device float * outsPrev,
    constant uint * pGlobalOffset,
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
    uint globalOffset;
    
    if (pGlobalOffset && pNbNeurons && pNbNeuronsPrev &&
        pNbBatch && pSequence && outsPrev && outs)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        globalOffset = *pGlobalOffset;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depth >= nbNeuronsPrev || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offsetPrev = depth +
        nbNeuronsPrev * seq + sequence * nbNeuronsPrev * elem;
    uint offset = globalOffset+depth +
        nbNeurons * seq + sequence * nbNeurons * elem;
    
    outs[offset] = outsPrev[offsetPrev];
}

kernel void concat2SeqBackward(
    const device float * delta,
    constant uint * pGlobalOffset,
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
    uint globalOffset;
    uint dirty;
    
    if (pGlobalOffset && pNbNeurons && pNbNeuronsPrev &&
        pNbBatch && pSequence && pDirty && deltaPrev && delta)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        globalOffset = *pGlobalOffset;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depth >= nbNeuronsPrev || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offsetPrev = depth +
        nbNeuronsPrev * seq + sequence * nbNeuronsPrev * elem;
    uint offset = globalOffset+depth +
        nbNeurons * seq + sequence * nbNeurons * elem;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = delta[offset];
    }
    else
    {
        deltaPrev[offsetPrev] += delta[offset];
    }
}

kernel void constant12SeqForward(
    const device float * weights,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence && weights && outs)
    {
        nbNeurons = *pNbNeurons;
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
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    outs[offset] = weights[depth + nbNeurons * seq];
}

kernel void constant12SeqBackward(
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint accumulate;
    
    if (pNbNeurons && pNbBatch && pSequence && pAccumulate && delta && grads)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id[0];
    uint seq = id[1];
    if (depth >= nbNeurons || seq >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        tmp += delta[offset];
    }
    
    if (accumulate)
    {
        grads[depth + nbNeurons * seq] += tmp;
    }
    else
    {
        grads[depth + nbNeurons * seq] = tmp;
    }
}

kernel void constant2SeqForward(
    const device float * weights,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence && weights && outs)
    {
        nbNeurons = *pNbNeurons;
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
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    outs[offset] = weights[depth];
}

kernel void querySeqForward(
    const device float * query,
    const device float * key,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeuronsPrev && pNbBatch && pSequence &&
        query && key && outs)
    {
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint seqK = id[0];
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (seqK >= sequence || elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint depthPrev=0; depthPrev<nbNeuronsPrev; depthPrev++)
    {
        uint offsetQuery = depthPrev +
            nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
        uint offsetKey = depthPrev +
            nbNeuronsPrev * seqK + sequence * nbNeuronsPrev * elem;
        
        tmp += query[offsetQuery] * key[offsetKey];
    }
    tmp /= sqrt((float)nbNeuronsPrev);
    
    uint offset = seqK + sequence * seqQ + sequence * sequence * elem;
    outs[offset] = tmp;
}

kernel void queryQuerySeqBackward(
    const device float * delta,
    const device float * key,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * query,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint dirty;
    
    if (pNbNeuronsPrev && pNbBatch && pSequence && pDirty &&
        query && key && delta)
    {
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depthPrev = id[0];
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (depthPrev >= nbNeuronsPrev || elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offset = seqK + sequence * seqQ + sequence * sequence * elem;
        uint offsetKey = depthPrev +
            nbNeuronsPrev * seqK + sequence * nbNeuronsPrev * elem;
        
        tmp += delta[offset] * key[offsetKey];
    }
    tmp /= sqrt((float)nbNeuronsPrev);
    
    uint offsetQuery = depthPrev +
        nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
    
    if (dirty)
    {
        query[offsetQuery] = tmp;
    }
    else
    {
        query[offsetQuery] += tmp;
    }
}

kernel void queryKeySeqBackward(
    const device float * delta,
    const device float * query,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * key,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint dirty;
    
    if (pNbNeuronsPrev && pNbBatch && pSequence && pDirty &&
        query && key && delta)
    {
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depthPrev = id[0];
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    
    if (depthPrev >= nbNeuronsPrev || elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqQ=0; seqQ<sequence; seqQ++)
    {
        uint offset = seqK + sequence * seqQ + sequence * sequence * elem;
        uint offsetQuery = depthPrev +
            nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
        
        tmp += delta[offset] * query[offsetQuery];
    }
    tmp /= sqrt((float)nbNeuronsPrev);
    
    uint offsetKey = depthPrev +
        nbNeuronsPrev * seqK + sequence * nbNeuronsPrev * elem;
    
    if (dirty)
    {
        key[offsetKey] = tmp;
    }
    else
    {
        key[offsetKey] += tmp;
    }
}

kernel void softmaxSeqForward(
    const device float * outsPrev,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence && outsPrev && outs)
    {
        nbNeurons = *pNbNeurons;
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
    
    float cMax = 0.0;
    for (uint depth1=0; depth1<nbNeurons; depth1++)
    {
        uint offset1 = depth1 +
            nbNeurons * seq + sequence * nbNeurons * elem;
        
        float outPrev = outsPrev[offset1];
        if (outPrev > cMax)
        {
            cMax = outPrev;
        }
    }
    
    float sum1 = 0.0;
    for (uint depth1=0; depth1<nbNeurons; depth1++)
    {
        uint offset1 = depth1 +
            nbNeurons * seq + sequence * nbNeurons * elem;
        
        float outPrev = outsPrev[offset1];
        sum1 += exp(outPrev - cMax);
    }
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    float outPrev = outsPrev[offset];
    outs[offset] = exp(outPrev - cMax) / sum1;
}

kernel void softmaxSeqBackward(
    const device float * outsPrev,
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint dirty;
    
    if (pNbNeurons && pNbBatch && pSequence && pDirty &&
        outsPrev && deltaPrev && delta)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
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
    
    float cMax = 0.0;
    for (uint depth1=0; depth1<nbNeurons; depth1++)
    {
        uint offset1 = depth1 +
            nbNeurons * seq + sequence * nbNeurons * elem;
        
        float outPrev = outsPrev[offset1];
        if (outPrev > cMax)
        {
            cMax = outPrev;
        }
    }
    
    float sum1 = 0.0;
    for (uint depth1=0; depth1<nbNeurons; depth1++)
    {
        uint offset1 = depth1 +
            nbNeurons * seq + sequence * nbNeurons * elem;
        float outPrev = outsPrev[offset1];
        sum1 += exp(outPrev - cMax);
    }
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    float outPrev = outsPrev[offset];
    float deltaCur = delta[offset];
    
    float sum2 = 0.0;
    for (uint depth2=0; depth2<nbNeurons; depth2++)
    {
        uint offset2 = depth2 +
            nbNeurons * seq + sequence * nbNeurons * elem;
        
        float outPrev2 = outsPrev[offset2];
        float deltaCur2 = delta[offset2];
        sum2 += exp(outPrev + outPrev2 - 2 * cMax) * deltaCur2;
    }
    
    if (dirty)
    {
        deltaPrev[offset] = -sum2 / (sum1 * sum1) +
            exp(outPrev - cMax) * deltaCur / sum1;
    }
    else
    {
        deltaPrev[offset] += -sum2 / (sum1 * sum1) +
            exp(outPrev - cMax) * deltaCur / sum1;
    }
}

kernel void valueSeqForward(
    const device float * value,
    const device float * score,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence &&
        value && score && outs)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (depth >= nbNeurons || elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offsetValue = depth +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetScore = seqK +
            sequence * seqQ + sequence * sequence * elem;
        
        tmp += value[offsetValue] * score[offsetScore];
    }
    
    uint offset = depth + nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void valueValueSeqBackward(
    const device float * delta,
    const device float * score,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * value,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint dirty;
    
    if (pNbNeurons && pNbBatch && pSequence && pDirty &&
        value && score && delta)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (depth >= nbNeurons || elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offset = depth + nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetScore = seqK +
            sequence * seqQ + sequence * sequence * elem;
        
        tmp += delta[offset] * score[offsetScore];
    }
    
    uint offsetValue = depth +
        nbNeurons * seqQ + sequence * nbNeurons * elem;
    
    if (dirty)
    {
        value[offsetValue] = tmp;
    }
    else
    {
        value[offsetValue] += tmp;
    }
}

kernel void valueScoreSeqBackward(
    const device float * delta,
    const device float * value,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * score,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint dirty;
    
    if (pNbNeurons && pNbBatch && pSequence && pDirty &&
        value && score && delta)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint seqK = id[0];
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (seqK >= sequence || elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetValue = depth +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        
        tmp += delta[offset] * value[offsetValue];
    }
    
    uint offsetScore = seqK +
        sequence * seqQ + sequence * sequence * elem;
    
    if (dirty)
    {
        score[offsetScore] = tmp;
    }
    else
    {
        score[offsetScore] += tmp;
    }
}
