//
// InstanceNorm.metal
// GrAIdient
//
// Created by Jean-François Reboud on 17/02/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void computeInstanceNormConvμ(
    const device half * tmps,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device half * μ,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    
    if (pNbChannels && pNbBatch && pDimensions && tmps && μ)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    if (depth >= nbChannels || elem >= nbBatch)
    {
        return ;
    }
    
    uint nbElems = width * height;
    float sum = 0.0;
    
    for (uint x=0; x<width; x++){
    for (uint y=0; y<height; y++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        uint offset = y + (offsetStart + x) * width;
            
        sum += tmps[offset];
    }}
    
    μ[depth + nbChannels * elem] = sum / nbElems;
}

kernel void computeInstanceNormConvσ2(
    const device half * tmps,
    const device half * μ,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device half * σ2,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    
    if (pNbChannels && pNbBatch && pDimensions && tmps && μ && σ2)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    if (depth >= nbChannels || elem >= nbBatch)
    {
        return ;
    }
    
    uint nbElems = width * height;
    float sum = 0.0;
    
    for (uint x=0; x<width; x++){
    for (uint y=0; y<height; y++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        uint offset = y + (offsetStart + x) * width;
            
        float tmp = tmps[offset] - μ[depth + nbChannels * elem];
        sum += tmp * tmp;
    }}
    
    σ2[depth + nbChannels * elem] = sum / nbElems;
}

kernel void forwardInstanceNormConv(
    const device half * β,
    const device half * Ɣ,
    const device half * μ,
    const device half * σ2,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device half * tmps,
    device half * xHat,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbChannels && pNbBatch && pDimensions && β && Ɣ &&
        tmps && xHat && μ && σ2)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint depth = id[0] / width;
    uint elem = id[1] / height;
    uint i = id[1] % height;
    uint j = id[0] % width;
    
    if (i * elem >= height * nbBatch ||
        j * depth >= width * nbChannels)
    {
        return ;
    }
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float tmp1 = tmps[offset] - μ[depth + nbChannels * elem];
    float tmp2 = sqrt(σ2[depth + nbChannels * elem] + Ɛ);
    float xhat = tmp1 / tmp2;
    xHat[offset] = xhat;
    tmps[offset] = Ɣ[depth] * xhat + β[depth];
}

kernel void forwardAdaIN(
    const device half * outsPrev,
    const device half * styles,
    const device half * μ,
    const device half * σ2,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device half * outs,
    device half * xHat,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbChannels && pNbBatch && pDimensions && outsPrev && styles &&
        outs && xHat && μ && σ2)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint depth = id[0] / width;
    uint elem = id[1] / height;
    uint i = id[1] % height;
    uint j = id[0] % width;
    
    if (i * elem >= height * nbBatch ||
        j * depth >= width * nbChannels)
    {
        return ;
    }
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float tmp1 = outsPrev[offset] - μ[depth + nbChannels * elem];
    float tmp2 = sqrt(σ2[depth + nbChannels * elem] + Ɛ);
    float xhat = tmp1 / tmp2;
    xHat[offset] = xhat;
    outs[offset] = styles[depth] * xhat + styles[depth + nbChannels];
}

kernel void backwardWeightsInstanceNormConv(
    const device half * delta,
    const device half * xHat,
    const device half * Ɣ,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pAccumulate,
    device half * sum1,
    device half * sum2,
    device half * dƔ,
    device half * dβ,
    uint id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    uint accumulate;
    
    if (pNbChannels && pNbBatch && pDimensions && pAccumulate &&
        delta && xHat && Ɣ &&
        sum1 && sum2 && dƔ && dβ)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id;
    if (depth >= nbChannels)
    {
        return ;
    }
    
    float tmp3 = 0.0, tmp4 = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        float tmp1 = 0.0, tmp2 = 0.0;
        for (uint x=0; x<width; x++){
        for (uint y=0; y<height; y++)
        {
            uint offsetStart = (depth + nbChannels * elem) * height;
            uint offset = y + (offsetStart + x) * width;
            
            float deltaTmp = delta[offset];
            float xHatTmp = xHat[offset];
            float dxHat = Ɣ[depth] * deltaTmp;
            tmp1 += dxHat;
            tmp2 += dxHat * xHatTmp;
            tmp3 += deltaTmp * xHatTmp;
            tmp4 += deltaTmp;
        }}
        
        sum1[depth + nbChannels * elem] = tmp1;
        sum2[depth + nbChannels * elem] = tmp2;
    }
    
    if (accumulate)
    {
        dƔ[depth] += tmp3;
        dβ[depth] += tmp4;
    }
    else
    {
        dƔ[depth] = tmp3;
        dβ[depth] = tmp4;
    }
}

kernel void backward2AdaIN(
    const device half * delta,
    const device half * xHat,
    const device half * outStyles,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pDirty,
    device half * sum1,
    device half * sum2,
    device half * deltaStyles,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    uint dirty;
    
    if (pNbChannels && pNbBatch && pDimensions && pDirty &&
        delta && xHat && outStyles &&
        sum1 && sum2 && deltaStyles)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    if (depth >= nbChannels || elem >= nbBatch)
    {
        return ;
    }
    
    float tmp1 = 0.0, tmp2 = 0.0;
    float tmp3 = 0.0, tmp4 = 0.0;
    
    for (uint x=0; x<width; x++){
    for (uint y=0; y<height; y++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        uint offset = y + (offsetStart + x) * width;
        
        float deltaTmp = delta[offset];
        float xHatTmp = xHat[offset];
        float dxHat = outStyles[depth] * deltaTmp;
        tmp1 += dxHat;
        tmp2 += dxHat * xHatTmp;
        tmp3 += deltaTmp * xHatTmp;
        tmp4 += deltaTmp;
    }}
        
    sum1[depth + nbChannels * elem] = tmp1;
    sum2[depth + nbChannels * elem] = tmp2;
    
    uint offset = (2 * nbChannels) * elem;
    if (dirty)
    {
        deltaStyles[depth + offset] = tmp3;
        deltaStyles[depth + nbChannels + offset] = tmp4;
    }
    else
    {
        deltaStyles[depth + offset] += tmp3;
        deltaStyles[depth + nbChannels + offset] += tmp4;
    }
}

kernel void backwardInstanceNormConv(
    const device half * σ2,
    const device half * xHat,
    const device half * Ɣ,
    const device half * sum1,
    const device half * sum2,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device half * delta,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbChannels && pNbBatch && pDimensions &&
        σ2 && xHat && Ɣ && sum1 && sum2 && delta)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint depth = id[0] / width;
    uint elem = id[1] / height;
    uint i = id[1] % height;
    uint j = id[0] % width;
    uint nbElems = width * height;
    
    if (i * elem >= height * nbBatch ||
        j * depth >= width * nbChannels)
    {
        return ;
    }
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float mult =
        1.0 / ((float)nbElems * sqrt(σ2[depth + nbChannels * elem] + Ɛ));
    float dxHat = Ɣ[depth] * delta[offset];
    float tmp1 = nbElems * dxHat;
    float tmp2 = sum1[depth + nbChannels * elem];
    float tmp3 = xHat[offset] * sum2[depth + nbChannels * elem];
    
    delta[offset] = mult * (tmp1 - tmp2 - tmp3);
}

kernel void backward1AdaIN(
    const device half * delta,
    const device half * σ2,
    const device half * xHat,
    const device half * styles,
    const device half * sum1,
    const device half * sum2,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    uint dirty;
    float Ɛ = 1e-5;
    
    if (pNbChannels && pNbBatch && pDimensions && pDirty &&
        delta && σ2 && xHat && styles && sum1 && sum2 && deltaPrev)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0] / width;
    uint elem = id[1] / height;
    uint i = id[1] % height;
    uint j = id[0] % width;
    uint nbElems = width * height;
    
    if (i * elem >= height * nbBatch ||
        j * depth >= width * nbChannels)
    {
        return ;
    }
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float mult =
        1.0 / ((float)nbElems * sqrt(σ2[depth + nbChannels * elem] + Ɛ));
    float dxHat = styles[depth] * delta[offset];
    float tmp1 = nbElems * dxHat;
    float tmp2 = sum1[depth + nbChannels * elem];
    float tmp3 = xHat[offset] * sum2[depth + nbChannels * elem];
    
    if (dirty)
    {
        deltaPrev[offset] = mult * (tmp1 - tmp2 - tmp3);
    }
    else
    {
        deltaPrev[offset] += mult * (tmp1 - tmp2 - tmp3);
    }
}
