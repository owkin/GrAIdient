//
// RMSNormSeqFloat.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/06/2024.
//

#include <metal_stdlib>
using namespace metal;

kernel void computeRMSNormSeqσ2Float(
    const device float * tmps,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * σ2,
    uint2 id [[ thread_position_in_grid ]])
{
    uint elem = id[1];
    uint seq = id[0];
    if (elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint nbElems = nbNeurons;
    float sum = 0.0;
    
    uint offset = nbNeurons * seq + sequence * nbNeurons * elem;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        float tmp = tmps[offset + depth];
        sum += tmp * tmp;
    }
    
    σ2[seq + sequence * elem] = sum / nbElems;
}

kernel void forwardRMSNormSeqFloat(
    const device float * Ɣ,
    const device float * σ2,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * tmps,
    device float * xHat,
    uint2 id [[ thread_position_in_grid ]])
{
    float Ɛ = 1e-5;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (depth >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    
    float tmp1 = tmps[offset];
    float tmp2 = sqrt(σ2[seq + sequence * elem] + Ɛ);
    float xhat = tmp1 / tmp2;
    xHat[offset] = xhat;
    tmps[offset] = Ɣ[depth] * xhat;
}

kernel void backwardWeights1RMSNormSeqFloat(
    const device float * delta,
    const device float * xHat,
    const device float * Ɣ,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * sum2,
    uint2 id [[ thread_position_in_grid ]])
{
    uint elem = id[1];
    uint seq = id[0];
    if (elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    float tmp = 0.0;
    uint offset = nbNeurons * seq + sequence * nbNeurons * elem;
    
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offsetTmp = depth + offset;
        
        float deltaTmp = delta[offsetTmp];
        float xHatTmp = xHat[offsetTmp];
        float dxHat = Ɣ[depth] * deltaTmp;
        tmp += dxHat * xHatTmp;
    }
    sum2[seq + sequence * elem] = tmp;
}

kernel void backwardWeights2RMSNormSeqFloat(
    const device float * delta,
    const device float * xHat,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & accumulate,
    device float * dƔ,
    uint id [[ thread_position_in_grid ]])
{
    uint depth = id;
    if (depth >= nbNeurons)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++) {
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offset = depth +
            nbNeurons * seq + sequence * nbNeurons * elem;
            
        float deltaTmp = delta[offset];
        float xHatTmp = xHat[offset];
        
        tmp += deltaTmp * xHatTmp;
    }}
    
    if (accumulate)
    {
        dƔ[depth] += tmp;
    }
    else
    {
        dƔ[depth] = tmp;
    }
}

kernel void backwardRMSNormSeqFloat(
    const device float * σ2,
    const device float * xHat,
    const device float * Ɣ,
    const device float * sum2,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * delta,
    uint2 id [[ thread_position_in_grid ]])
{
    float Ɛ = 1e-5;
    
    uint depth = id[0];
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    uint nbElems = nbNeurons;
    
    if (depth >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
    
    float mult =
        1.0 / ((float)nbElems * sqrt(σ2[seq + sequence * elem] + Ɛ));
    float dxHat = Ɣ[depth] * delta[offset];
    float tmp1 = nbElems * dxHat;
    float tmp3 = xHat[offset] * sum2[seq + sequence * elem];
    
    delta[offset] = mult * (tmp1 - tmp3);
}
