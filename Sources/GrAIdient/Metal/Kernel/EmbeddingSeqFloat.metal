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
    const device int * ins,
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pVocabularySize,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint vocabularySize;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pVocabularySize && pNbBatch && pSequence &&
        ins && delta && deltaWeights)
    {
        nbNeurons = *pNbNeurons;
        vocabularySize = *pVocabularySize;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint elem = id[1] / vocabularySize;
    uint embedding = id[1] % vocabularySize;
    uint depth = id[0];
    
    if (depth >= nbNeurons || elem * embedding >= nbBatch * vocabularySize)
    {
        return ;
    }
    
    float sum = 0.0;
    for (uint seq=0; seq<sequence; seq++)
    {
        int minIndex = ins[seq + elem * sequence];
        if (minIndex == (int)embedding)
        {
            uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
            float deltaCur = delta[offset];
            
            sum += deltaCur;
        }
    }
    
    uint offsetWeights = depth +
        nbNeurons * embedding + vocabularySize * nbNeurons * elem;
    deltaWeights[offsetWeights] += sum;
}
