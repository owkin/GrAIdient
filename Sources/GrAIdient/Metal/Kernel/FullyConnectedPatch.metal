//
// FullyConnectedPatch.metal
// GrAIdient
//
// Created by Jean-François Reboud on 25/02/2023.
//

#include <metal_stdlib>
using namespace metal;

kernel void flPatchForward(
    const device float * outsPrev,
    const device float * weights,
    const device float * biases,
    constant uint * pNbNeurons,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensionsPrev,
    constant uint * pPatch,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbChannelsPrev;
    uint heightPrev, widthPrev;
    uint patch;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbChannelsPrev && pDimensionsPrev && pPatch &&
        pNbBatch && pSequence &&
        outsPrev && weights && biases && outs)
    {
        nbNeurons = *pNbNeurons;
        nbChannelsPrev = *pNbChannelsPrev;
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        patch = *pPatch;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint weightWidth = nbChannelsPrev * patch * patch;
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depth >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint nbSeqPerRow = heightPrev / patch;
    uint nbSeqPerCol = widthPrev / patch;
    uint seqI = seq / nbSeqPerCol;
    uint seqJ = seq % nbSeqPerCol;
    
    uint iStart = seqI * patch;
    uint jStart = seqJ * patch;
    
    float tmp = biases[depth];
    for (uint depthPrev=0; depthPrev<nbChannelsPrev; depthPrev++)
    {
        uint offsetStartPrev =
            (depthPrev + nbChannelsPrev * elem) * heightPrev;
        
        for (uint i=iStart; i<iStart+patch; i++) {
        for (uint j=jStart; j<jStart+patch; j++)
        {
            uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
            float outPrev = outsPrev[offsetPrev];
            
            uint offsetWeight = j +
                i * _patch + depthPrev * _patch * _patch;
            uint offsetWeights = offsetWeight + weightWidth * depth;
            float w = weights[offsetWeights];
            
            tmp += outPrev * w;
        }}
    }
    
    uint offset = depth + nbNeurons * seq + nbBatch * nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void flPatchBackward(
    const device float * delta,
    const device float * weights,
    constant uint * pNbNeurons,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensionsPrev,
    constant uint * pPatch,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbChannelsPrev;
    uint heightPrev, widthPrev;
    uint patch;
    uint nbBatch;
    uint sequence;
    uint dirty;
    
    if (pNbNeurons && pNbNeuronsPrev && pNbBatch && pDirty &&
        deltaPrev && weights && delta)
    {
        nbNeurons = *pNbNeurons;
        nbChannelsPrev = *pNbChannelsPrev;
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        patch = *pPatch;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint weightWidth = nbChannelsPrev * patch * patch;
    uint offsetWeight = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (offsetWeight >= weightWidth || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint nbSeqPerRow = heightPrev / patch;
    uint nbSeqPerCol = widthPrev / patch;
    uint seqI = seq / nbSeqPerCol;
    uint seqJ = seq % nbSeqPerCol;
    
    uint iStart = seqI * patch;
    uint jStart = seqJ * patch;
    
    uint res = offsetWeight;
    uint depthPrev = res / (patch * patch);
    res -= depthPrev * patch * patch;
    uint i = res / patch;
    res -= i * patch;
    uint j = res;
    
    float tmp = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offsetWeights = offsetWeight + weightWidth * depth;
        float w = weights[offsetWeights];
        
        uint offset = depth + nbNeurons * seq + nbBatch * nbNeurons * elem;
        float deltaCur = delta[offset];
        
        tmp += w * deltaCur;
    }
    
    uint offsetStartPrev = (depthPrev + nbChannelsPrev * elem) * heightPrev;
    uint offsetPrev = jStart+j + (offsetStartPrev + iStart+i) * widthPrev;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = tmp;
    }
    else
    {
        deltaPrev[offsetPrev] += tmp;
    }
}

kernel void flPatchBatchDerWeights(
    const device float * outsPrev,
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensionsPrev,
    constant uint * pPatch,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbChannelsPrev;
    uint heightPrev, widthPrev;
    uint patch;
    uint nbBatch;
    uint sequence;
    uint accumulate;
    
    if (pNbNeurons && pNbChannelsPrev && pDimensionsPrev && pPatch &&
        pNbBatch && pSequence && pAccumulate &&
        outsPrev && delta && grads)
    {
        nbNeurons = *pNbNeurons;
        nbChannelsPrev = *pNbChannelsPrev;
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        patch = *pPatch;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint weightWidth = nbChannelsPrev * patch * patch;
    uint depth = id[0];
    uint offsetWeight = id[1];
    
    if (depth >= nbNeurons || offsetWeight >= weightWidth)
    {
        return ;
    }
    
    uint nbSeqPerRow = heightPrev / patch;
    uint nbSeqPerCol = widthPrev / patch;
    
    uint res = offsetWeight;
    uint depthPrev = res / (patch * patch);
    res -= depthPrev * patch * patch;
    uint i = res / patch;
    res -= i * patch;
    uint j = res;
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++) {
    for (uint seq=0; seq<sequence; seq++)
    {
        uint seqI = seq / nbSeqPerCol;
        uint seqJ = seq % nbSeqPerCol;
        
        uint iStart = seqI * patch;
        uint jStart = seqJ * patch;
        
        uint offset = depth + nbNeurons * seq + nbBatch * nbNeurons * elem;
        float deltaCur = delta[offset];
        
        uint offsetStartPrev = (depthPrev + nbChannelsPrev * elem) * heightPrev;
        uint offsetPrev = jStart+j + (offsetStartPrev + iStart+i) * widthPrev;
        float outPrev = outsPrev[offsetPrev];
        
        tmp += outPrev * deltaCur;
    }}
    
    uint offsetWeights = offsetWeight + weightWidth * depth;
    if (accumulate)
    {
        grads[offsetWeights] += tmp;
    }
    else
    {
        grads[offsetWeights] = tmp;
    }
}

kernel void flPatchBatchDerBiases(
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pAccumulate,
    device float * grads,
    uint id [[ thread_position_in_grid ]])
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
    
    uint depth = id;
    if (depth >= nbNeurons)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++) {
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offset = depth + nbNeurons * seq + nbBatch * nbNeurons * elem;
        tmp += delta[offset];
    }}
    
    if (accumulate)
    {
        grads[depth] += tmp;
    }
    else
    {
        grads[depth] = tmp;
    }
}

kernel void flPatchDerWeights(
    const device float * outsPrev,
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensionsPrev,
    constant uint * pPatch,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbChannelsPrev;
    uint heightPrev, widthPrev;
    uint patch;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbChannelsPrev && pDimensionsPrev && pPatch &&
        pNbBatch && pSequence &&
        outsPrev && delta && deltaWeights)
    {
        nbNeurons = *pNbNeurons;
        nbChannelsPrev = *pNbChannelsPrev;
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        patch = *pPatch;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint weightWidth = nbChannelsPrev * patch * patch;
    uint depth = id[0] % nbNeurons;
    uint offsetWeight = id[1];
    uint elem = id[0] / nbNeurons;
    
    if (depth * elem >= nbNeurons * nbBatch ||
        offsetWeight >= weightWidth)
    {
        return ;
    }
    
    uint nbSeqPerRow = heightPrev / patch;
    uint nbSeqPerCol = widthPrev / patch;
    
    uint res = offsetWeight;
    uint depthPrev = res / (patch * patch);
    res -= depthPrev * patch * patch;
    uint i = res / patch;
    res -= i * patch;
    uint j = res;
    
    float tmp = 0.0;
    for (uint seq=0; seq<sequence; seq++)
    {
        uint seqI = seq / nbSeqPerCol;
        uint seqJ = seq % nbSeqPerCol;
        
        uint iStart = seqI * patch;
        uint jStart = seqJ * patch;
        
        uint offset = depth + nbNeurons * seq + nbBatch * nbNeurons * elem;
        float deltaCur = delta[offset];
        
        uint offsetStartPrev = (depthPrev + nbChannelsPrev * elem) * heightPrev;
        uint offsetPrev = jStart+j + (offsetStartPrev + iStart+i) * widthPrev;
        float outPrev = outsPrev[offsetPrev];
        
        tmp += outPrev * deltaCur;
    }
    
    uint offsetStartWeights = elem * nbNeurons * weightWidth;
    uint offsetWeights = offsetStartWeights +
        offsetWeight + weightWidth * depth;
    deltaWeights[offsetWeights] = tmp;
}

kernel void flPatchDerBiases(
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence && delta && deltaWeights)
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
        uint offset = depth + nbNeurons * seq + nbBatch * nbNeurons * elem;
        tmp += delta[offset];
    }
    
    uint offsetWeights = elem * nbNeurons + depth;
    deltaWeights[offsetWeights] = tmp;
}

kernel void flPatchReduceWeights(
    const device float * deltaWeights,
    constant uint * pNbNeurons,
    constant uint * pNbChannelsPrev,
    constant uint * pPatch,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbChannelsPrev;
    uint patch;
    uint nbBatch;
    uint accumulate;
    
    if (pNbNeurons && pNbChannelsPrev && pPatch && pNbBatch && pAccumulate &&
        deltaWeights && grads)
    {
        nbNeurons = *pNbNeurons;
        nbChannelsPrev = *pNbChannelsPrev;
        nbBatch = *pNbBatch;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint weightWidth = nbChannelsPrev * patch * patch;
    uint depth = id[0];
    uint offsetWeight = id[1];
    
    if (depth >= nbNeurons || offsetWeight >= weightWidth)
    {
        return ;
    }
    
    uint offsetWeights = offsetWeight + weightWidth * depth;
        
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = elem * nbNeurons * weightWidth + offsetWeights;
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
