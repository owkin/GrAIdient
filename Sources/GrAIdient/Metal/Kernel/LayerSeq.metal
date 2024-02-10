//
// LayerSeq.metal
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

kernel void selectSeqForward(
    const device float * outsPrev,
    constant uint * pNbNeurons,
    constant uint * pTargetSeq,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint targetSeq;
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pTargetSeq && pNbNeurons && pNbBatch && pSequence &&
        outsPrev && outs)
    {
        targetSeq = *pTargetSeq;
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
    
    uint offset = depth + nbNeurons * elem;
    uint offsetPrev = depth +
        nbNeurons * targetSeq + sequence * nbNeurons * elem;
    outs[offset] = outsPrev[offsetPrev];
}

kernel void selectSeqBackward(
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pTargetSeq,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint targetSeq;
    
    if (pNbNeurons && pTargetSeq && pNbBatch && pSequence &&
        deltaPrev && delta)
    {
        targetSeq = *pTargetSeq;
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
    
    uint offset = depth + nbNeurons * elem;
    uint offsetPrev = depth +
        nbNeurons * targetSeq + sequence * nbNeurons * elem;
    deltaPrev[offsetPrev] += delta[offset];
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

kernel void concat1Seq4Forward(
    const device float4 * outsPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pSequencePrev,
    device float4 * outs,
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
    
    if (depth * 4 >= nbNeurons || elem >= nbBatch || seq >= sequencePrev)
    {
        return ;
    }
    
    uint offsetPrev = (depth * 4 +
        nbNeurons * seq + sequencePrev * nbNeurons * elem) / 4;
    uint offset = (depth * 4 +
        nbNeurons * (globalOffset+seq) + sequence * nbNeurons * elem) / 4;
    
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

kernel void concat1Seq4Backward(
    const device float4 * delta,
    constant uint * pGlobalOffset,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pSequencePrev,
    constant uint * pDirty,
    device float4 * deltaPrev,
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
    
    if (depth * 4 >= nbNeurons || elem >= nbBatch || seq >= sequencePrev)
    {
        return ;
    }
    
    uint offsetPrev = (depth * 4 +
        nbNeurons * seq + sequencePrev * nbNeurons * elem) / 4;
    uint offset = (depth * 4 +
        nbNeurons * (globalOffset+seq) + sequence * nbNeurons * elem) / 4;
    
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

kernel void constant12Seq4Forward(
    const device float4 * weights,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float4 * outs,
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
    
    if (depth * 4 >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offset =
        (depth * 4 + nbNeurons * seq + sequence * nbNeurons * elem) / 4;
    outs[offset] = weights[(depth * 4 + nbNeurons * seq) / 4];
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

kernel void constant12Seq4Backward(
    const device float4 * delta,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pAccumulate,
    device float4 * grads,
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
    if (depth * 4 >= nbNeurons || seq >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset =
            (depth * 4 + nbNeurons * seq + sequence * nbNeurons * elem) / 4;
        tmp += delta[offset];
    }
    
    if (accumulate)
    {
        grads[(depth * 4 + nbNeurons * seq) / 4] += tmp;
    }
    else
    {
        grads[(depth * 4 + nbNeurons * seq) / 4] = tmp;
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

kernel void constant2Seq4Forward(
    const device float4 * weights,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float4 * outs,
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
    
    if (depth * 4 >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offset =
        (depth * 4 + nbNeurons * seq + sequence * nbNeurons * elem) / 4;
    outs[offset] = weights[depth];
}

kernel void querySeqForward(
    const device float * query,
    const device float * key,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev && pNbBatch && pSequence &&
        query && key && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeuronsPrev / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (head >= nbHeads || seqK >= sequence ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint depthPrev = j + head * size;
        
        uint offsetQuery = depthPrev +
            nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
        uint offsetKey = depthPrev +
            nbNeuronsPrev * seqK + sequence * nbNeuronsPrev * elem;
        
        tmp += query[offsetQuery] * key[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offset = seqK + head * sequence +
        nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void querySeq4Forward(
    const device float4 * query,
    const device float4 * key,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev && pNbBatch && pSequence &&
        query && key && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeuronsPrev / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (head >= nbHeads || seqK >= sequence ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint j=0; j<size/4; j++)
    {
        uint depthPrev = j * 4 + head * size;
        
        uint offsetQuery = (depthPrev +
            nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem) / 4;
        uint offsetKey = (depthPrev +
            nbNeuronsPrev * seqK + sequence * nbNeuronsPrev * elem) / 4;
        
        tmp += query[offsetQuery] * key[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offset = seqK + head * sequence +
        nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

kernel void queryQuerySeqBackward(
    const device float * delta,
    const device float * key,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * query,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBatch && pSequence && pDirty &&
        query && key && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
        size = nbNeuronsPrev / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    uint depthPrev = j + head * size;
    
    if (head >= nbHeads || j >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offset = seqK + head * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetKey = depthPrev +
            nbNeuronsPrev * seqK + sequence * nbNeuronsPrev * elem;
        
        tmp += delta[offset] * key[offsetKey];
    }
    tmp /= sqrt((float)size);
    
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

kernel void queryQuerySeq4Backward(
    const device float * delta,
    const device float4 * key,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float4 * query,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBatch && pSequence && pDirty &&
        query && key && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
        size = nbNeuronsPrev / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    uint depthPrev = j * 4 + head * size;
    
    if (head >= nbHeads || j * 4 >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offset = seqK + head * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetKey = (depthPrev +
            nbNeuronsPrev * seqK + sequence * nbNeuronsPrev * elem) / 4;
        
        tmp += delta[offset] * key[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offsetQuery = (depthPrev +
        nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem) / 4;
    
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
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * key,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBatch && pSequence && pDirty &&
        query && key && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
        size = nbNeuronsPrev / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    uint depthPrev = j + head * size;
    
    if (head >= nbHeads || j >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqQ=0; seqQ<sequence; seqQ++)
    {
        uint offset = seqK + head * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetQuery = depthPrev +
            nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
        
        tmp += delta[offset] * query[offsetQuery];
    }
    tmp /= sqrt((float)size);
    
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

kernel void queryKeySeq4Backward(
    const device float * delta,
    const device float4 * query,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float4 * key,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBatch && pSequence && pDirty &&
        query && key && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
        size = nbNeuronsPrev / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    uint depthPrev = j * 4 + head * size;
    
    if (head >= nbHeads || j * 4 >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint seqQ=0; seqQ<sequence; seqQ++)
    {
        uint offset = seqK + head * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetQuery = (depthPrev +
            nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem) / 4;
        
        tmp += delta[offset] * query[offsetQuery];
    }
    tmp /= sqrt((float)size);
    
    uint offsetKey = (depthPrev +
        nbNeuronsPrev * seqK + sequence * nbNeuronsPrev * elem) / 4;
    
    if (dirty)
    {
        key[offsetKey] = tmp;
    }
    else
    {
        key[offsetKey] += tmp;
    }
}

kernel void querySelfSeqForward(
    const device float * outsPrev,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev1;
    uint nbNeuronsPrev2;
    uint nbBlocksPrev;
    uint queryOffset, keyOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence &&
        outsPrev && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev1 = *pNbNeuronsPrev;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeuronsPrev2 = nbNeuronsPrev1 / nbBlocksPrev;
        queryOffset = pGlobalOffset[0];
        keyOffset = pGlobalOffset[1];
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeuronsPrev2 / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (head >= nbHeads || seqK >= sequence ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint depthPrev = j + head * size;
        
        uint offsetQuery = depthPrev + queryOffset * nbNeuronsPrev2 +
            nbNeuronsPrev1 * seqQ + sequence * nbNeuronsPrev1 * elem;
        uint offsetKey = depthPrev + keyOffset * nbNeuronsPrev2 +
            nbNeuronsPrev1 * seqK + sequence * nbNeuronsPrev1 * elem;
        
        tmp += outsPrev[offsetQuery] * outsPrev[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offset = seqK + head * sequence +
        nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void querySelfSeq4Forward(
    const device float4 * outsPrev,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev1;
    uint nbNeuronsPrev2;
    uint nbBlocksPrev;
    uint queryOffset, keyOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence &&
        outsPrev && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev1 = *pNbNeuronsPrev;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeuronsPrev2 = nbNeuronsPrev1 / nbBlocksPrev;
        queryOffset = pGlobalOffset[0];
        keyOffset = pGlobalOffset[1];
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeuronsPrev2 / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (head >= nbHeads || seqK >= sequence ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint j=0; j<size/4; j++)
    {
        uint depthPrev = j * 4 + head * size;
        
        uint offsetQuery = (depthPrev + queryOffset * nbNeuronsPrev2 +
            nbNeuronsPrev1 * seqQ + sequence * nbNeuronsPrev1 * elem) / 4;
        uint offsetKey = (depthPrev + keyOffset * nbNeuronsPrev2 +
            nbNeuronsPrev1 * seqK + sequence * nbNeuronsPrev1 * elem) / 4;
        
        tmp += outsPrev[offsetQuery] * outsPrev[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offset = seqK + head * sequence +
        nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

kernel void querySelfQuerySeqBackward(
    const device float * outsPrev,
    const device float * delta,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev1;
    uint nbNeuronsPrev2;
    uint nbBlocksPrev;
    uint queryOffset, keyOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence && pDirty &&
        outsPrev && delta && deltaPrev)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev1 = *pNbNeuronsPrev;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeuronsPrev2 = nbNeuronsPrev1 / nbBlocksPrev;
        queryOffset = pGlobalOffset[0];
        keyOffset = pGlobalOffset[1];
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
        size = nbNeuronsPrev2 / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    uint depthPrev = j + head * size;
    
    if (head >= nbHeads || j >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offset = seqK + head * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetKey = depthPrev + keyOffset * nbNeuronsPrev2 +
            nbNeuronsPrev1 * seqK + sequence * nbNeuronsPrev1 * elem;
        
        tmp += delta[offset] * outsPrev[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offsetQuery = depthPrev + queryOffset * nbNeuronsPrev2 +
        nbNeuronsPrev1 * seqQ + sequence * nbNeuronsPrev1 * elem;
    
    if (dirty)
    {
        deltaPrev[offsetQuery] = tmp;
    }
    else
    {
        deltaPrev[offsetQuery] += tmp;
    }
}

kernel void querySelfQuerySeq4Backward(
    const device float4 * outsPrev,
    const device float * delta,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float4 * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev1;
    uint nbNeuronsPrev2;
    uint nbBlocksPrev;
    uint queryOffset, keyOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence && pDirty &&
        outsPrev && delta && deltaPrev)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev1 = *pNbNeuronsPrev;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeuronsPrev2 = nbNeuronsPrev1 / nbBlocksPrev;
        queryOffset = pGlobalOffset[0];
        keyOffset = pGlobalOffset[1];
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
        size = nbNeuronsPrev2 / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    uint depthPrev = j * 4 + head * size;
    
    if (head >= nbHeads || j * 4 >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offset = seqK + head * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetKey = (depthPrev + keyOffset * nbNeuronsPrev2 +
            nbNeuronsPrev1 * seqK + sequence * nbNeuronsPrev1 * elem) / 4;
        
        tmp += delta[offset] * outsPrev[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offsetQuery = (depthPrev + queryOffset * nbNeuronsPrev2 +
        nbNeuronsPrev1 * seqQ + sequence * nbNeuronsPrev1 * elem) / 4;
    
    if (dirty)
    {
        deltaPrev[offsetQuery] = tmp;
    }
    else
    {
        deltaPrev[offsetQuery] += tmp;
    }
}

kernel void querySelfKeySeqBackward(
    const device float * outsPrev,
    const device float * delta,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev1;
    uint nbNeuronsPrev2;
    uint nbBlocksPrev;
    uint queryOffset, keyOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence && pDirty &&
        outsPrev && delta && deltaPrev)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev1 = *pNbNeuronsPrev;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeuronsPrev2 = nbNeuronsPrev1 / nbBlocksPrev;
        queryOffset = pGlobalOffset[0];
        keyOffset = pGlobalOffset[1];
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
        size = nbNeuronsPrev2 / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    uint depthPrev = j + head * size;
    
    if (head >= nbHeads || j >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqQ=0; seqQ<sequence; seqQ++)
    {
        uint offset = seqK + head * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetQuery = depthPrev + queryOffset * nbNeuronsPrev2 +
            nbNeuronsPrev1 * seqQ + sequence * nbNeuronsPrev1 * elem;
        
        tmp += delta[offset] * outsPrev[offsetQuery];
    }
    tmp /= sqrt((float)size);
    
    uint offsetKey = depthPrev + keyOffset * nbNeuronsPrev2 +
        nbNeuronsPrev1 * seqK + sequence * nbNeuronsPrev1 * elem;
    
    if (dirty)
    {
        deltaPrev[offsetKey] = tmp;
    }
    else
    {
        deltaPrev[offsetKey] += tmp;
    }
}

kernel void querySelfKeySeq4Backward(
    const device float4 * outsPrev,
    const device float * delta,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float4 * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev1;
    uint nbNeuronsPrev2;
    uint nbBlocksPrev;
    uint queryOffset, keyOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence && pDirty &&
        outsPrev && delta && deltaPrev)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev1 = *pNbNeuronsPrev;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeuronsPrev2 = nbNeuronsPrev1 / nbBlocksPrev;
        queryOffset = pGlobalOffset[0];
        keyOffset = pGlobalOffset[1];
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
        size = nbNeuronsPrev2 / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    uint depthPrev = j * 4 + head * size;
    
    if (head >= nbHeads || j * 4 >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint seqQ=0; seqQ<sequence; seqQ++)
    {
        uint offset = seqK + head * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetQuery = (depthPrev + queryOffset * nbNeuronsPrev2 +
            nbNeuronsPrev1 * seqQ + sequence * nbNeuronsPrev1 * elem) / 4;
        
        tmp += delta[offset] * outsPrev[offsetQuery];
    }
    tmp /= sqrt((float)size);
    
    uint offsetKey = (depthPrev + keyOffset * nbNeuronsPrev2 +
        nbNeuronsPrev1 * seqK + sequence * nbNeuronsPrev1 * elem) / 4;
    
    if (dirty)
    {
        deltaPrev[offsetKey] = tmp;
    }
    else
    {
        deltaPrev[offsetKey] += tmp;
    }
}

kernel void softmaxSeqForward(
    const device float * outsPrev,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbBatch && pSequence && outsPrev && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons / nbHeads;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    uint head = depth / size;
    
    if (depth >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    float cMax = outsPrev[
        0+head*size + nbNeurons * seq + sequence * nbNeurons * elem
    ];
    for (uint j=0; j<size; j++)
    {
        uint offset1 = j+head*size +
            nbNeurons * seq + sequence * nbNeurons * elem;
        
        float outPrev = outsPrev[offset1];
        if (outPrev > cMax)
        {
            cMax = outPrev;
        }
    }
    
    float sum1 = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint offset1 = j+head*size +
            nbNeurons * seq + sequence * nbNeurons * elem;
        
        float outPrev = outsPrev[offset1];
        sum1 += exp(outPrev - cMax);
    }
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    float outPrev = outsPrev[offset];
    outs[offset] = exp(outPrev - cMax) / sum1;
}

kernel void softmaxSeq4Forward(
    const device float4 * outsPrev,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float4 * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbBatch && pSequence && outsPrev && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons / nbHeads;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    uint head = depth / (size / 4);
    
    if (depth * 4 >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    float cMax = outsPrev[
        (depth * 4 + nbNeurons * seq + sequence * nbNeurons * elem) / 4
    ][0];
    for (uint j=0; j<size/4; j++)
    {
        uint offset1 = (j*4+head*size +
            nbNeurons * seq + sequence * nbNeurons * elem) / 4;
        
        float4 outPrev = outsPrev[offset1];
        float max1 = max(outPrev[0], outPrev[1]);
        float max2 = max(outPrev[2], outPrev[3]);
        float max3 = max(max1, max2);
        if (max3 > cMax)
        {
            cMax = max3;
        }
    }
    
    float4 sum1 = 0.0;
    for (uint j=0; j<size/4; j++)
    {
        uint offset1 = (j*4+head*size +
            nbNeurons * seq + sequence * nbNeurons * elem) / 4;
        
        float4 outPrev = outsPrev[offset1];
        sum1 += exp(outPrev - cMax);
    }
    
    float sum2 = sum1[0] + sum1[1] + sum1[2] + sum1[3];
    
    uint offset =
        (depth * 4 + nbNeurons * seq + sequence * nbNeurons * elem) / 4;
    float4 outPrev = outsPrev[offset];
    outs[offset] = exp(outPrev - cMax) / sum2;
}

kernel void softmaxSeqBackward(
    const device float * outs,
    const device float * delta,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbBatch && pSequence && pDirty &&
        deltaPrev && outs && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons / nbHeads;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    uint head = depth / size;
    
    if (depth >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    float outCur = outs[offset];
    float deltaCur = delta[offset];
    
    float sum1 = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint offset1 = j+head*size +
            nbNeurons * seq + sequence * nbNeurons * elem;
        
        float outCur1 = outs[offset1];
        float deltaCur1 = delta[offset1];
        sum1 += outCur1 * deltaCur1;
    }
    
    if (dirty)
    {
        deltaPrev[offset] = outCur * (deltaCur - sum1);
    }
    else
    {
        deltaPrev[offset] += outCur * (deltaCur - sum1);
    }
}

kernel void softmaxSeq4Backward(
    const device float4 * outs,
    const device float4 * delta,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float4 * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbBatch && pSequence && pDirty &&
        deltaPrev && outs && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons / nbHeads;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    uint head = depth / (size / 4);
    
    if (depth * 4 >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offset =
        (depth * 4 + nbNeurons * seq + sequence * nbNeurons * elem) / 4;
    float4 outCur = outs[offset];
    float4 deltaCur = delta[offset];
    
    float4 sum1 = 0.0;
    for (uint j=0; j<size/4; j++)
    {
        uint offset1 = (j*4+head*size +
            nbNeurons * seq + sequence * nbNeurons * elem) / 4;
        
        float4 outCur1 = outs[offset1];
        float4 deltaCur1 = delta[offset1];
        sum1 += outCur1 * deltaCur1;
    }
    
    float sum2 = sum1[0] + sum1[1] + sum1[2] + sum1[3];
    if (dirty)
    {
        deltaPrev[offset] = outCur * (deltaCur - sum2);
    }
    else
    {
        deltaPrev[offset] += outCur * (deltaCur - sum2);
    }
}

kernel void valueSeqForward(
    const device float * value,
    const device float * score,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev && pNbBatch && pSequence &&
        value && score && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    uint depth = j + head * size;
    
    if (head >= nbHeads || j >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offsetValue = depth +
            nbNeurons * seqK + sequence * nbNeurons * elem;
        uint offsetScore = seqK + head * sequence +
            nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
        
        tmp += value[offsetValue] * score[offsetScore];
    }
    
    uint offset = depth + nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void valueSeq4Forward(
    const device float4 * value,
    const device float * score,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float4 * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev && pNbBatch && pSequence &&
        value && score && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    uint depth = j * 4 + head * size;
    
    if (head >= nbHeads || j >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offsetValue = (depth +
            nbNeurons * seqK + sequence * nbNeurons * elem) / 4;
        uint offsetScore = seqK + head * sequence +
            nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
        
        tmp += value[offsetValue] * score[offsetScore];
    }
    
    uint offset = (depth + nbNeurons * seqQ + sequence * nbNeurons * elem) / 4;
    outs[offset] = tmp;
}

kernel void valueValueSeqBackward(
    const device float * delta,
    const device float * score,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * value,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBatch && pSequence && pDirty &&
        value && score && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons / nbHeads;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint head = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    uint depth = j + head * size;
    
    if (head >= nbHeads || j >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqQ=0; seqQ<sequence; seqQ++)
    {
        uint offset = depth + nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetScore = seqK + head * sequence +
            nbNeuronsPrev * seqQ +
            sequence * nbNeuronsPrev * elem;
        
        tmp += delta[offset] * score[offsetScore];
    }
    
    uint offsetValue = depth + nbNeurons * seqK + sequence * nbNeurons * elem;
    if (dirty)
    {
        value[offsetValue] = tmp;
    }
    else
    {
        value[offsetValue] += tmp;
    }
}

kernel void valueValueSeq4Backward(
    const device float4 * delta,
    const device float * score,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float4 * value,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBatch && pSequence && pDirty &&
        value && score && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons / nbHeads;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint head = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    uint depth = j * 4 + head * size;
    
    if (head >= nbHeads || j * 4 >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint seqQ=0; seqQ<sequence; seqQ++)
    {
        uint offset =
            (depth + nbNeurons * seqQ + sequence * nbNeurons * elem) / 4;
        uint offsetScore = seqK + head * sequence +
            nbNeuronsPrev * seqQ +
            sequence * nbNeuronsPrev * elem;
        
        tmp += delta[offset] * score[offsetScore];
    }
    
    uint offsetValue =
        (depth + nbNeurons * seqK + sequence * nbNeurons * elem) / 4;
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
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * score,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBatch && pSequence && pDirty &&
        value && score && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons / nbHeads;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint head = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (head >= nbHeads || seqK >= sequence ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint depth = j + head * size;
        
        uint offset = depth + nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetValue = depth +
            nbNeurons * seqK + sequence * nbNeurons * elem;
        
        tmp += delta[offset] * value[offsetValue];
    }
    
    uint offsetScore = seqK + head * sequence +
        nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
    
    if (dirty)
    {
        score[offsetScore] = tmp;
    }
    else
    {
        score[offsetScore] += tmp;
    }
}

kernel void valueScoreSeq4Backward(
    const device float4 * delta,
    const device float4 * value,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * score,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBatch && pSequence && pDirty &&
        value && score && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons / nbHeads;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint head = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (head >= nbHeads || seqK >= sequence ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint j=0; j<size/4; j++)
    {
        uint depth = j * 4 + head * size;
        
        uint offset =
            (depth + nbNeurons * seqQ + sequence * nbNeurons * elem) / 4;
        uint offsetValue = (depth +
            nbNeurons * seqK + sequence * nbNeurons * elem) / 4;
        
        tmp += delta[offset] * value[offsetValue];
    }
    
    uint offsetScore = seqK + head * sequence +
        nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
    
    if (dirty)
    {
        score[offsetScore] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
    else
    {
        score[offsetScore] += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
}

kernel void valueSelfSeqForward(
    const device float * value,
    const device float * score,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons1;
    uint nbNeurons2;
    uint nbNeuronsPrev;
    uint nbBlocksPrev;
    uint valueOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence &&
        value && score && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons2 = *pNbNeurons;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeurons1 = nbNeurons2 * nbBlocksPrev;
        nbNeuronsPrev = *pNbNeuronsPrev;
        valueOffset = *pGlobalOffset;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons2 / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    uint depth = j + head * size;
    
    if (head >= nbHeads || j >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offsetValue = depth + valueOffset * nbNeurons2 +
            nbNeurons1 * seqK + sequence * nbNeurons1 * elem;
        uint offsetScore = seqK + head * sequence +
            nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
        
        tmp += value[offsetValue] * score[offsetScore];
    }
    
    uint offset = depth + nbNeurons2 * seqQ + sequence * nbNeurons2 * elem;
    outs[offset] = tmp;
}

kernel void valueSelfSeq4Forward(
    const device float4 * value,
    const device float * score,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float4 * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons1;
    uint nbNeurons2;
    uint nbNeuronsPrev;
    uint nbBlocksPrev;
    uint valueOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence &&
        value && score && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons2 = *pNbNeurons;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeurons1 = nbNeurons2 * nbBlocksPrev;
        nbNeuronsPrev = *pNbNeuronsPrev;
        valueOffset = *pGlobalOffset;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons2 / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    uint depth = j * 4 + head * size;
    
    if (head >= nbHeads || j >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint seqK=0; seqK<sequence; seqK++)
    {
        uint offsetValue = (depth + valueOffset * nbNeurons2 +
            nbNeurons1 * seqK + sequence * nbNeurons1 * elem) / 4;
        uint offsetScore = seqK + head * sequence +
            nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
        
        tmp += value[offsetValue] * score[offsetScore];
    }
    
    uint offset =
        (depth + nbNeurons2 * seqQ + sequence * nbNeurons2 * elem) / 4;
    outs[offset] = tmp;
}

kernel void valueSelfValueSeqBackward(
    const device float * delta,
    const device float * score,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * value,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons1;
    uint nbNeurons2;
    uint nbNeuronsPrev;
    uint nbBlocksPrev;
    uint valueOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence &&
        value && score && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons2 = *pNbNeurons;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeurons1 = nbNeurons2 * nbBlocksPrev;
        nbNeuronsPrev = *pNbNeuronsPrev;
        valueOffset = *pGlobalOffset;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons2 / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    uint depth = j + head * size;
    
    if (head >= nbHeads || j >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint seqQ=0; seqQ<sequence; seqQ++)
    {
        uint offset = depth + nbNeurons2 * seqQ + sequence * nbNeurons2 * elem;
        uint offsetScore = seqK + head * sequence +
            nbNeuronsPrev * seqQ +
            sequence * nbNeuronsPrev * elem;
        
        tmp += delta[offset] * score[offsetScore];
    }
    
    uint offsetValue = depth + valueOffset * nbNeurons2 +
        nbNeurons1 * seqK + sequence * nbNeurons1 * elem;
    value[offsetValue] += tmp;
}

kernel void valueSelfValueSeq4Backward(
    const device float4 * delta,
    const device float * score,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float4 * value,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons1;
    uint nbNeurons2;
    uint nbNeuronsPrev;
    uint nbBlocksPrev;
    uint valueOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence &&
        value && score && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons2 = *pNbNeurons;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeurons1 = nbNeurons2 * nbBlocksPrev;
        nbNeuronsPrev = *pNbNeuronsPrev;
        valueOffset = *pGlobalOffset;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons2 / nbHeads;
    }
    else
        return ;
    
    uint head = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    uint depth = j * 4 + head * size;
    
    if (head >= nbHeads || j * 4 >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint seqQ=0; seqQ<sequence; seqQ++)
    {
        uint offset =
            (depth + nbNeurons2 * seqQ + sequence * nbNeurons2 * elem) / 4;
        uint offsetScore = seqK + head * sequence +
            nbNeuronsPrev * seqQ +
            sequence * nbNeuronsPrev * elem;
        
        tmp += delta[offset] * score[offsetScore];
    }
    
    uint offsetValue = (depth + valueOffset * nbNeurons2 +
        nbNeurons1 * seqK + sequence * nbNeurons1 * elem) / 4;
    value[offsetValue] += tmp;
}

kernel void valueSelfScoreSeqBackward(
    const device float * delta,
    const device float * value,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * score,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons1;
    uint nbNeurons2;
    uint nbNeuronsPrev;
    uint nbBlocksPrev;
    uint valueOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence && pDirty &&
        value && score && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons2 = *pNbNeurons;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeurons1 = nbNeurons2 * nbBlocksPrev;
        nbNeuronsPrev = *pNbNeuronsPrev;
        valueOffset = *pGlobalOffset;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons2 / nbHeads;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint head = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (head >= nbHeads || seqK >= sequence ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint depth = j + head * size;
        
        uint offset = depth + nbNeurons2 * seqQ + sequence * nbNeurons2 * elem;
        uint offsetValue = depth + valueOffset * nbNeurons2 +
            nbNeurons1 * seqK + sequence * nbNeurons1 * elem;
        
        tmp += delta[offset] * value[offsetValue];
    }
    
    uint offsetScore = seqK + head * sequence +
        nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
    
    if (dirty)
    {
        score[offsetScore] = tmp;
    }
    else
    {
        score[offsetScore] += tmp;
    }
}

kernel void valueSelfScoreSeq4Backward(
    const device float4 * delta,
    const device float4 * value,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBlocksPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * score,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint nbNeurons1;
    uint nbNeurons2;
    uint nbNeuronsPrev;
    uint nbBlocksPrev;
    uint valueOffset;
    uint nbBatch;
    uint sequence;
    uint size;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbNeuronsPrev &&
        pNbBlocksPrev && pGlobalOffset && pNbBatch && pSequence && pDirty &&
        value && score && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons2 = *pNbNeurons;
        nbBlocksPrev = *pNbBlocksPrev;
        nbNeurons1 = nbNeurons2 * nbBlocksPrev;
        nbNeuronsPrev = *pNbNeuronsPrev;
        valueOffset = *pGlobalOffset;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        size = nbNeurons2 / nbHeads;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint head = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (head >= nbHeads || seqK >= sequence ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    float4 tmp = 0.0;
    for (uint j=0; j<size/4; j++)
    {
        uint depth = j * 4 + head * size;
        
        uint offset =
            (depth + nbNeurons2 * seqQ + sequence * nbNeurons2 * elem) / 4;
        uint offsetValue = (depth + valueOffset * nbNeurons2 +
            nbNeurons1 * seqK + sequence * nbNeurons1 * elem) / 4;
        
        tmp += delta[offset] * value[offsetValue];
    }
    
    uint offsetScore = seqK + head * sequence +
        nbNeuronsPrev * seqQ + sequence * nbNeuronsPrev * elem;
    
    if (dirty)
    {
        score[offsetScore] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
    else
    {
        score[offsetScore] += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
}

kernel void layerCAMSeqForward(
    const device float * outsPrev,
    const device float * deltaPrev,
    constant uint * pNbNeuronsPrev,
    constant uint * pKeepPositive,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbBatch;
    uint sequence;
    uint nbNeuronsPrev;
    uint keepPositive;
    
    if (pNbNeuronsPrev && pKeepPositive && pNbBatch && pSequence &&
        outsPrev && outs)
    {
        nbNeuronsPrev = *pNbNeuronsPrev;
        keepPositive = *pKeepPositive;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint seq = id[0];
    uint elem = id[1];
    
    if (seq >= sequence || elem >= nbBatch)
    {
        return ;
    }
        
    float sum = 0.0;
    for (uint depthPrev=0; depthPrev<nbNeuronsPrev; depthPrev++)
    {
        uint offsetPrev = depthPrev + nbNeuronsPrev * seq +
            sequence * nbNeuronsPrev * elem;
        
        float deltaPrevTmp = deltaPrev[offsetPrev];
        if (!keepPositive)
        {
            deltaPrevTmp = -deltaPrevTmp;
        }
        if (deltaPrevTmp < 0)
        {
            deltaPrevTmp = 0.0;
        }
        
        sum = deltaPrevTmp * outsPrev[offsetPrev];
    }
    
    uint offset = seq + sequence * elem;
    outs[offset] = sum;
}
