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
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint elem = id[1];
    uint seq = id[0];
    
    if (seq >= sequence || elem >= nbBatch)
    {
        return ;
    }
    
    int index = ins[seq + sequence * elem];
    uint offset = nbNeurons * seq + sequence * nbNeurons * elem;
    uint offsetWeights = nbNeurons * index;
    
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        outs[depth + offset] = weights[depth + offsetWeights];
    }
}

kernel void embeddingSeqBatchDerWeightsFloat(
    const device int * ins,
    const device float * delta,
    constant uint & nbNeurons,
    constant uint & vocabularySize,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint embedding = id[1];
    uint depth = id[0];
    
    if (depth >= nbNeurons || embedding >= vocabularySize)
    {
        return ;
    }
    
    float sum = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = depth + sequence * nbNeurons * elem;
        for (uint seq=0; seq<sequence; seq++)
        {
            int index = ins[seq + elem * sequence];
            if (index == (int)embedding)
            {
                uint offsetTmp = nbNeurons * seq + offset;
                float deltaCur = delta[offsetTmp];
                
                sum += deltaCur;
            }
        }
    }
    
    uint offsetWeights = depth + nbNeurons * embedding;
    grads[offsetWeights] += sum;
}

kernel void embeddingSeqDerWeightsFloat(
    const device int * ins,
    const device float * delta,
    constant uint & nbNeurons,
    constant uint & vocabularySize,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint elem = id[1] / vocabularySize;
    uint embedding = id[1] % vocabularySize;
    uint depth = id[0];
    
    if (depth >= nbNeurons || elem * embedding >= nbBatch * vocabularySize)
    {
        return ;
    }
    
    float sum = 0.0;
    uint offset = depth + sequence * nbNeurons * elem;
    
    for (uint seq=0; seq<sequence; seq++)
    {
        int minIndex = ins[seq + elem * sequence];
        if (minIndex == (int)embedding)
        {
            uint offsetTmp = nbNeurons * seq + offset;
            float deltaCur = delta[offsetTmp];
            
            sum += deltaCur;
        }
    }
    
    uint offsetWeights = depth +
        nbNeurons * embedding + vocabularySize * nbNeurons * elem;
    deltaWeights[offsetWeights] += sum;
}
