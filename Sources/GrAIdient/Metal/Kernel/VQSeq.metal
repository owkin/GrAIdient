//
// VQSeq.metal
// GrAIdient
//
// Created by Jean-François Reboud on 18/06/2023.
//

#include <metal_stdlib>
using namespace metal;

kernel void vqSeqForward(
    const device float * outsPrev,
    const device float * weights,
    constant uint * pNbNeurons,
    constant uint * pK,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * outs,
    device int * indices,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint K;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pK && pNbBatch && pSequence &&
        weights && outsPrev && outs && indices)
    {
        nbNeurons = *pNbNeurons;
        K = *pK;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint elem = id[1];
    uint seq = id[0];
    
    if (seq >= sequence || elem >= nbBatch)
    {
        return ;
    }
    
    int minIndex = -1;
    float minValue = 0.0;
    for (uint k=0; k<K; k++)
    {
        float value = 0.0;
        for (uint depth=0; depth<nbNeurons; depth++)
        {
            uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
            
            uint offsetWeights = depth + nbNeurons * k;
            
            float outPrev = outsPrev[offset];
            float vq = weights[offsetWeights];
            value += pow(outPrev - vq, 2.0);
        }
        
        if (minIndex < 0 || value < minValue)
        {
            minValue = value;
            minIndex = k;
        }
    }
    
    if (minIndex >= 0)
    {
        for (uint depth=0; depth<nbNeurons; depth++)
        {
            uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
            
            uint offsetWeights = depth + nbNeurons * minIndex;
            outs[offset] = weights[offsetWeights];
        }
        indices[seq + elem * sequence] = minIndex;
    }
}

kernel void vqSeqBackward(
    const device float * outsPrev,
    const device float * delta,
    const device float * weights,
    const device int * indices,
    constant uint * pNbNeurons,
    constant uint * pK,
    constant float * pBeta,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint K;
    float beta;
    uint nbBatch;
    uint sequence;
    uint dirty;
    
    if (pNbNeurons && pK && pBeta && pNbBatch && pSequence && pDirty &&
        outsPrev && delta && weights && indices && deltaPrev)
    {
        nbNeurons = *pNbNeurons;
        K = *pK;
        beta = *pBeta;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (seq * elem >= sequence * nbBatch ||
        depth >= nbNeurons)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    
    int minIndex = indices[seq + elem * sequence];
    uint offsetWeights = depth + nbNeurons * minIndex;
    
    float vq = weights[offsetWeights];
    float deltaCur = delta[offset];
    float outPrev = outsPrev[offset];
    
    if (dirty)
    {
        deltaPrev[offset] = deltaCur;
    }
    else
    {
        deltaPrev[offset] += deltaCur;
    }
    
    // Commitment term.
    deltaPrev[offset] += beta * 2.0 * (outPrev - vq);
}

kernel void vqSeqBatchDerWeights(
    const device float * outsPrev,
    const device float * weights,
    const device int * indices,
    constant uint * pNbNeurons,
    constant uint * pK,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint K;
    float coeff;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pK && pCoeff && pNbBatch && pSequence &&
        outsPrev && weights && indices && grads)
    {
        nbNeurons = *pNbNeurons;
        K = *pK;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint k = id[1];
    uint depth = id[0];
    
    if (depth >= nbNeurons || k >= K)
    {
        return ;
    }
    
    float sum = 0.0;
    for (uint elem=0; elem<nbBatch; elem++){
    for (uint seq=0; seq<sequence; seq++)
    {
        int minIndex = indices[seq + elem * sequence];
        if (minIndex == (int)k)
        {
            uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
            
            uint offsetWeights = depth + nbNeurons * minIndex;
            
            float vq = weights[offsetWeights];
            float outPrev = outsPrev[offset];
            
            sum += vq - outPrev;
        }
    }}
    sum *= coeff / (float)(nbBatch * sequence) * 2.0;
    
    grads[depth + nbNeurons * k] += sum;
}

kernel void vqSeqDerWeights(
    const device float * outsPrev,
    const device float * weights,
    const device int * indices,
    constant uint * pNbNeurons,
    constant uint * pK,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint K;
    float coeff;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pK && pCoeff && pNbBatch && pSequence &&
        outsPrev && weights && indices && deltaWeights)
    {
        nbNeurons = *pNbNeurons;
        K = *pK;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint elem = id[1] / K;
    uint k = id[1] % K;
    uint depth = id[0];
    
    if (depth >= nbNeurons || elem * k >= nbBatch * K)
    {
        return ;
    }
    
    float sum = 0.0;
    for (uint seq=0; seq<sequence; seq++)
    {
        int minIndex = indices[seq + elem * sequence];
        if (minIndex == (int)k)
        {
            uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
            
            uint offsetWeights = depth + nbNeurons * minIndex;
            
            float vq = weights[offsetWeights];
            float outPrev = outsPrev[offset];
            
            sum += vq - outPrev;
        }
    }
    sum *= coeff / (float)(nbBatch * sequence) * 2.0;
    
    deltaWeights[depth + nbNeurons * k + K * nbNeurons * elem] += sum;
}

kernel void vqSeqLoss(
    const device float * outsPrev,
    const device float * outs,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * losses,
    uint id [[ thread_position_in_grid ]])
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
    
    uint elem = id;
    if (elem >= nbBatch)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++) {
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        
        float outPrev = outsPrev[offset];
        float vq = outs[offset];
        float diff = outPrev - vq;
        
        tmp += diff * diff;
    }}
    losses[elem] = tmp;
}
