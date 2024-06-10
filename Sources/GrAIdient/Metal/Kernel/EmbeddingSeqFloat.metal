//
// EmbeddingSeqFloat.metal
// GrAIdient
//
// Created by Jean-François Reboud on 10/06/2024.
//

#include <metal_stdlib>
using namespace metal;

kernel void embeddingSeqForwardFloat(
    const device int * ins,
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
    
    if (pNbNeurons && pNbBatch && pSequence &&
        weights && ins && outs)
    {
        nbNeurons = *pNbNeurons;
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
    
    int index = ins[seq + sequence * elem];
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        
        uint offsetWeights = depth + nbNeurons * index;
        outs[offset] = weights[offsetWeights];
    }
}

kernel void embeddingSeqBatchDerWeightsFloat(
    const device int * ins,
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pVocabularySize,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint vocabularySize;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pVocabularySize && pNbBatch && pSequence &&
        ins && delta && grads)
    {
        nbNeurons = *pNbNeurons;
        vocabularySize = *pVocabularySize;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint embedding = id[1];
    uint depth = id[0];
    
    if (depth >= nbNeurons || embedding >= vocabularySize)
    {
        return ;
    }
    
    float sum = 0.0;
    for (uint elem=0; elem<nbBatch; elem++){
    for (uint seq=0; seq<sequence; seq++)
    {
        int index = ins[seq + elem * sequence];
        if (index == (int)embedding)
        {
            uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
            float deltaCur = delta[offset];
            
            sum += deltaCur;
        }
    }}
    
    uint offsetWeights = depth + nbNeurons * embedding;
    grads[offsetWeights] += sum;
}

kernel void embeddingSeqDerWeightsFloat(
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
