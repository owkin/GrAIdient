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
