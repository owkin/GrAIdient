//
// VQSeq.metal
// GrAIdient
//
// Created by Jean-François Reboud on 18/06/2023.
//

#include <metal_stdlib>
using namespace metal;

kernel void vqSeqForwardHalf(
    const device half * outsPrev,
    const device half * weights,
    constant uint * pNbNeurons,
    constant uint * pK,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device half * outs,
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
    half minValue = 0.0;
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

kernel void vqSeqBackwardHalf(
    const device half * outsPrev,
    const device half * delta,
    const device half * weights,
    const device int * indices,
    constant uint * pNbNeurons,
    constant uint * pK,
    constant float * pBeta,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pDirty,
    device half * deltaPrev,
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
    if (minIndex >= 0)
    {
        uint offsetWeights = depth + nbNeurons * minIndex;
        
        float vq = weights[offsetWeights];
        half deltaCur = delta[offset];
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
        deltaPrev[offset] += beta / (float)(nbBatch * sequence) *
            2.0 * (outPrev - vq);
    }
    else if (dirty)
    {
        deltaPrev[offset] = 0.0;
    }
}

kernel void vqSeqBatchDerWeightsHalf(
    const device half * outsPrev,
    const device half * weights,
    const device int * indices,
    constant uint * pNbNeurons,
    constant uint * pK,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device half * grads,
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

kernel void vqSeqDerWeightsHalf(
    const device half * outsPrev,
    const device half * weights,
    const device int * indices,
    constant uint * pNbNeurons,
    constant uint * pK,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device half * deltaWeights,
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

kernel void vqSeqLossHalf(
    const device half * outsPrev,
    const device half * outs,
    const device int * indices,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device half * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence &&
        outsPrev && outs && indices && losses)
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
        int minIndex = indices[seq + elem * sequence];
        if (minIndex >= 0)
        {
            uint offset =
                depth + nbNeurons * seq + sequence * nbNeurons * elem;
            
            float outPrev = outsPrev[offset];
            float vq = outs[offset];
            float diff = outPrev - vq;
            
            tmp += diff * diff;
        }
    }}
    losses[elem] = tmp;
}

kernel void vqLayerCAMMaxSeqHalf(
     const device half * camLayer,
     constant uint * pNbNeurons,
     constant uint * pNbThreadgroups,
     constant uint * pNbBatch,
     constant uint * pSequence,
     device half * camMax,
     uint2 groupId [[ threadgroup_position_in_grid ]],
     uint2 threadId [[ thread_position_in_threadgroup ]],
     uint2 id [[ thread_position_in_grid ]])
{
    constexpr uint threadsPerThreadgroup = 64;
    threadgroup float camShared[threadsPerThreadgroup];
    
    uint nbNeurons;
    uint nbThreadgroups;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbThreadgroups && pNbBatch && pSequence &&
        camLayer && camMax)
    {
        nbNeurons = *pNbNeurons;
        nbThreadgroups = *pNbThreadgroups;
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
    
    camShared[threadId[0]] = camLayer[seq + sequence * elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride=threadsPerThreadgroup/2; stride>0; stride>>=1)
    {
        uint index = threadId[0] + groupId[0] * threadsPerThreadgroup;
        if (threadId[0] < stride &&
            (index + stride) < sequence)
        {
            camShared[threadId[0]] = max(
                camShared[threadId[0] + stride],
                camShared[threadId[0]]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (threadId[0] == 0)
    {
        uint offset = elem * nbThreadgroups + groupId[0];
        camMax[offset] = camShared[0];
    }
}

kernel void vqGradSeqForwardHalf(
    const device half * outsPrev,
    const device half * camLayer,
    const device half * camMax,
    const device half * weights,
    constant uint * pNbNeurons,
    constant uint * pK,
    constant float * pMagnitudeCoeff,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device half * outs,
    device int * indices,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint K;
    float magnitudeCoeff;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pK && pMagnitudeCoeff && pNbBatch && pSequence &&
        outsPrev && camLayer && camMax && weights && outs && indices)
    {
        nbNeurons = *pNbNeurons;
        K = *pK;
        magnitudeCoeff = *pMagnitudeCoeff;
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
    
    float cam = camLayer[seq + sequence * elem];
    if (cam / camMax[elem] >= magnitudeCoeff)
    {
        int minIndex = -1;
        float minValue = 0.0;
        for (uint k=0; k<K; k++)
        {
            float value = 0.0;
            for (uint depth=0; depth<nbNeurons; depth++)
            {
                uint offset =
                    depth + nbNeurons * seq + sequence * nbNeurons * elem;
                
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
                uint offset =
                    depth + nbNeurons * seq + sequence * nbNeurons * elem;
                
                uint offsetWeights = depth + nbNeurons * minIndex;
                outs[offset] = weights[offsetWeights];
            }
            indices[seq + elem * sequence] = minIndex;
        }
    }
    else
    {
        indices[seq + elem * sequence] = -1;
    }
}
