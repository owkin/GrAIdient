//
// LayerNorm.metal
// GrAIdient
//
// Created by Jean-François Reboud on 09/03/2023.
//

#include <metal_stdlib>
using namespace metal;

kernel void computeLayerNormSeqμFloat(
    const device float * tmps,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * μ,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence && tmps && μ)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint elem = id[1];
    uint seq = id[0];
    if (elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint nbElems = nbNeurons;
    float sum = 0.0;
    
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        sum += tmps[offset];
    }
    
    μ[seq + sequence * elem] = sum / nbElems;
}

kernel void computeLayerNormSeqμ4Float(
    const device float4 * tmps,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * μ,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence && tmps && μ)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint elem = id[1];
    uint seq = id[0];
    if (elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint nbElems = nbNeurons;
    float4 sum = 0.0;
    
    for (uint depth=0; depth<nbNeurons/4; depth++)
    {
        uint offset =
            (depth * 4 + nbNeurons * seq + sequence * nbNeurons * elem) / 4;
        sum += tmps[offset];
    }
    
    μ[seq + sequence * elem] = (sum[0] + sum[1] + sum[2] + sum[3]) / nbElems;
}

kernel void computeLayerNormSeqσ2Float(
    const device float * tmps,
    const device float * μ,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * σ2,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence && tmps && μ && σ2)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint elem = id[1];
    uint seq = id[0];
    if (elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint nbElems = nbNeurons;
    float sum = 0.0;
    
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * seq + sequence * nbNeurons * elem;
        
        float tmp = tmps[offset] - μ[seq + sequence * elem];
        sum += tmp * tmp;
    }
    
    σ2[seq + sequence * elem] = sum / nbElems;
}

kernel void computeLayerNormSeqσ24Float(
    const device float4 * tmps,
    const device float * μ,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * σ2,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence && tmps && μ && σ2)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint elem = id[1];
    uint seq = id[0];
    if (elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint nbElems = nbNeurons;
    float4 sum = 0.0;
    
    for (uint depth=0; depth<nbNeurons/4; depth++)
    {
        uint offset =
            (depth * 4 + nbNeurons * seq + sequence * nbNeurons * elem) / 4;
        
        float4 tmp = tmps[offset] - μ[seq + sequence * elem];
        sum += tmp * tmp;
    }
    
    σ2[seq + sequence * elem] = (sum[0] + sum[1] + sum[2] + sum[3]) / nbElems;
}

kernel void forwardLayerNormSeqFloat(
    const device float * β,
    const device float * Ɣ,
    const device float * μ,
    const device float * σ2,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * tmps,
    device float * xHat,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    float Ɛ = 1e-5;
    
    if (pNbNeurons && pNbBatch && pSequence && β && Ɣ &&
        tmps && xHat && μ && σ2)
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
    
    float tmp1 = tmps[offset] - μ[seq + sequence * elem];
    float tmp2 = sqrt(σ2[seq + sequence * elem] + Ɛ);
    float xhat = tmp1 / tmp2;
    xHat[offset] = xhat;
    tmps[offset] = Ɣ[depth] * xhat + β[depth];
}

kernel void forwardLayerNormSeq4Float(
    const device float4 * β,
    const device float4 * Ɣ,
    const device float * μ,
    const device float * σ2,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float4 * tmps,
    device float4 * xHat,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    float Ɛ = 1e-5;
    
    if (pNbNeurons && pNbBatch && pSequence && β && Ɣ &&
        tmps && xHat && μ && σ2)
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
    
    float4 tmp1 = tmps[offset] - μ[seq + sequence * elem];
    float tmp2 = sqrt(σ2[seq + sequence * elem] + Ɛ);
    float4 xhat = tmp1 / tmp2;
    xHat[offset] = xhat;
    tmps[offset] = Ɣ[depth] * xhat + β[depth];
}

kernel void backwardWeights1LayerNormSeqFloat(
    const device float * delta,
    const device float * xHat,
    const device float * Ɣ,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * sum1,
    device float * sum2,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence &&
        delta && xHat && Ɣ && sum1 && sum2)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint elem = id[1];
    uint seq = id[0];
    if (elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    float tmp1 = 0.0, tmp2 = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth +
            nbNeurons * seq + sequence * nbNeurons * elem;
        
        float deltaTmp = delta[offset];
        float xHatTmp = xHat[offset];
        float dxHat = Ɣ[depth] * deltaTmp;
        tmp1 += dxHat;
        tmp2 += dxHat * xHatTmp;
    }
    
    sum1[seq + sequence * elem] = tmp1;
    sum2[seq + sequence * elem] = tmp2;
}

kernel void backwardWeights1LayerNormSeq4Float(
    const device float4 * delta,
    const device float4 * xHat,
    const device float4 * Ɣ,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * sum1,
    device float * sum2,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    
    if (pNbNeurons && pNbBatch && pSequence &&
        delta && xHat && Ɣ && sum1 && sum2)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
    }
    else
        return ;
    
    uint elem = id[1];
    uint seq = id[0];
    if (elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    float4 tmp1 = 0.0, tmp2 = 0.0;
    for (uint depth=0; depth<nbNeurons/4; depth++)
    {
        uint offset = (depth * 4 +
            nbNeurons * seq + sequence * nbNeurons * elem) / 4;
        
        float4 deltaTmp = delta[offset];
        float4 xHatTmp = xHat[offset];
        float4 dxHat = Ɣ[depth] * deltaTmp;
        tmp1 += dxHat;
        tmp2 += dxHat * xHatTmp;
    }
    
    sum1[seq + sequence * elem] = tmp1[0] + +tmp1[1] + tmp1[2] + tmp1[3];
    sum2[seq + sequence * elem] = tmp2[0] + tmp2[1] + tmp2[2] + tmp2[3];
}

kernel void backwardWeights2LayerNormSeqFloat(
    const device float * delta,
    const device float * xHat,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pAccumulate,
    device float * dƔ,
    device float * dβ,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint accumulate;
    
    if (pNbNeurons && pNbBatch && pSequence && pAccumulate &&
        delta && xHat&& dƔ && dβ)
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
    
    float tmp1 = 0.0, tmp2 = 0.0;
    for (uint elem=0; elem<nbBatch; elem++) {
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offset = depth +
            nbNeurons * seq + sequence * nbNeurons * elem;
        
        float deltaTmp = delta[offset];
        float xHatTmp = xHat[offset];
        
        tmp1 += deltaTmp * xHatTmp;
        tmp2 += deltaTmp;
    }}
    
    if (accumulate)
    {
        dƔ[depth] += tmp1;
        dβ[depth] += tmp2;
    }
    else
    {
        dƔ[depth] = tmp1;
        dβ[depth] = tmp2;
    }
}

kernel void backwardWeights2LayerNormSeq4Float(
    const device float4 * delta,
    const device float4 * xHat,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    constant uint * pAccumulate,
    device float4 * dƔ,
    device float4 * dβ,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    uint accumulate;
    
    if (pNbNeurons && pNbBatch && pSequence && pAccumulate &&
        delta && xHat&& dƔ && dβ)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        sequence = *pSequence;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id;
    if (depth * 4 >= nbNeurons)
    {
        return ;
    }
    
    float4 tmp1 = 0.0, tmp2 = 0.0;
    for (uint elem=0; elem<nbBatch; elem++) {
    for (uint seq=0; seq<sequence; seq++)
    {
        uint offset = (depth * 4 +
            nbNeurons * seq + sequence * nbNeurons * elem) / 4;
        
        float4 deltaTmp = delta[offset];
        float4 xHatTmp = xHat[offset];
        
        tmp1 += deltaTmp * xHatTmp;
        tmp2 += deltaTmp;
    }}
    
    if (accumulate)
    {
        dƔ[depth] += tmp1;
        dβ[depth] += tmp2;
    }
    else
    {
        dƔ[depth] = tmp1;
        dβ[depth] = tmp2;
    }
}

kernel void backwardLayerNormSeqFloat(
    const device float * σ2,
    const device float * xHat,
    const device float * Ɣ,
    const device float * sum1,
    const device float * sum2,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float * delta,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    float Ɛ = 1e-5;
    
    if (pNbNeurons && pNbBatch && pSequence &&
        σ2 && xHat && Ɣ && sum1 && sum2 && delta)
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
    float tmp2 = sum1[seq + sequence * elem];
    float tmp3 = xHat[offset] * sum2[seq + sequence * elem];
    
    delta[offset] = mult * (tmp1 - tmp2 - tmp3);
}

kernel void backwardLayerNormSeq4Float(
    const device float * σ2,
    const device float4 * xHat,
    const device float4 * Ɣ,
    const device float * sum1,
    const device float * sum2,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pSequence,
    device float4 * delta,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint sequence;
    float Ɛ = 1e-5;
    
    if (pNbNeurons && pNbBatch && pSequence &&
        σ2 && xHat && Ɣ && sum1 && sum2 && delta)
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
    uint nbElems = nbNeurons;
    
    if (depth * 4 >= nbNeurons || elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offset =
        (depth * 4 + nbNeurons * seq + sequence * nbNeurons * elem) / 4;
    
    float mult =
        1.0 / ((float)nbElems * sqrt(σ2[seq + sequence * elem] + Ɛ));
    float4 dxHat = Ɣ[depth] * delta[offset];
    float4 tmp1 = nbElems * dxHat;
    float tmp2 = sum1[seq + sequence * elem];
    float4 tmp3 = xHat[offset] * sum2[seq + sequence * elem];
    
    delta[offset] = mult * (tmp1 - tmp2 - tmp3);
}
