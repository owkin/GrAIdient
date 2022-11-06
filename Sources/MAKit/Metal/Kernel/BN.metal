//
// BN.metal
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void computeConvμ(
    const device float * tmps,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pFirstCall,
    device float * μ,
    device float * Eμ,
    uint id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    uint firstCall;
    
    if (pNbChannels && pNbBatch && pDimensions && pFirstCall && tmps &&
        μ && Eμ)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        firstCall = *pFirstCall;
    }
    else
        return ;
    
    uint depth = id;
    if (depth >= nbChannels)
    {
        return ;
    }
    
    uint nbElems = nbBatch * width * height;
    float sum = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        for (uint x=0; x<width; x++){
        for (uint y=0; y<height; y++)
        {
            uint offsetStart = (depth + nbChannels * elem) * height;
            uint offset = y + (offsetStart + x) * width;
                
            sum += tmps[offset];
        }}
    }
    μ[depth] = sum / nbElems;
    
    if (pFirstCall)
    {
        Eμ[depth] = μ[depth];
    }
    else
    {
        Eμ[depth] = 0.9 * Eμ[depth] + 0.1 * μ[depth];
    }
}

kernel void computeConvσ2(
    const device float * tmps,
    const device float * μ,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pFirstCall,
    device float * σ2,
    device float * Eσ2,
    uint id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    uint firstCall;
    
    if (pNbChannels && pNbBatch && pDimensions && pFirstCall &&
        tmps && μ && σ2 && Eσ2)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        firstCall = *pFirstCall;
    }
    else
        return ;
    
    uint depth = id;
    if (depth >= nbChannels)
    {
        return ;
    }
    
    uint nbElems = nbBatch * width * height;
    float sum = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        for (uint x=0; x<width; x++){
        for (uint y=0; y<height; y++)
        {
            uint offsetStart = (depth + nbChannels * elem) * height;
            uint offset = y + (offsetStart + x) * width;
                
            float tmp = tmps[offset] - μ[depth];
            sum += tmp * tmp;
        }}
    }
    σ2[depth] = sum / nbElems;
    
    if (firstCall)
    {
        Eσ2[depth] = σ2[depth];
    }
    else
    {
        Eσ2[depth] = 0.9 * Eσ2[depth] + 0.1 * σ2[depth];
    }
}

kernel void forwardBNConvTraining(
    const device float * β,
    const device float * Ɣ,
    const device float * μ,
    const device float * σ2,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device float * tmps,
    device float * xHat,
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
    
    float tmp1 = tmps[offset] - μ[depth];
    float tmp2 = sqrt(σ2[depth] + Ɛ);
    float xhat = tmp1 / tmp2;
    xHat[offset] = xhat;
    tmps[offset] = Ɣ[depth] * xhat + β[depth];
}

kernel void forwardBNConvInference(
    const device float * β,
    const device float * Ɣ,
    const device float * Eμ,
    const device float * Eσ2,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pM,
    constant uint * pDimensions,
    device float * tmps,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint m;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbChannels && pNbBatch && pM && pDimensions && β && Ɣ &&
        tmps && Eμ && Eσ2)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        m = *pM;
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
    
    float Var = Eσ2[depth];
    if (m > 1)
    {
        Var *= (float)m / ((float)m - 1);
    }
    float tmp1 = tmps[offset] - Eμ[depth];
    float tmp2 = sqrt(Var + Ɛ);
    float xhat = tmp1 / tmp2;
    tmps[offset] = Ɣ[depth] * xhat + β[depth];
}

kernel void backwardWeightsBNConv(
    const device float * delta,
    const device float * xHat,
    const device float * Ɣ,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pAccumulate,
    device float * sum1,
    device float * sum2,
    device float * dƔ,
    device float * dβ,
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
    
    float tmp1 = 0.0, tmp2 = 0.0;
    float tmp3 = 0.0, tmp4 = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        for (uint x=0; x<width; x++){
        for (uint y=0; y<height; y++)
        {
            uint offsetStart = (depth + nbChannels * elem) * height;
            uint offset = y + (offsetStart + x) * width;
                
            float deltaTmp = delta[offset];
            float xHatTmp = xHat[offset];
            float dxhat = Ɣ[depth] * deltaTmp;
            tmp1 += dxhat;
            tmp2 += dxhat * xHatTmp;
            
            tmp3 += deltaTmp * xHatTmp;
            tmp4 += deltaTmp;
        }}
    }
    sum1[depth] = tmp1;
    sum2[depth] = tmp2;
    
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

kernel void backwardBNConvTraining(
    const device float * σ2,
    const device float * xHat,
    const device float * Ɣ,
    const device float * sum1,
    const device float * sum2,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device float * delta,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbChannels && pNbBatch && pDimensions && σ2 && xHat && Ɣ &&
        delta)
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
    uint nbElems = nbBatch * width * height;
    
    if (i * elem >= height * nbBatch ||
        j * depth >= width * nbChannels)
    {
        return ;
    }
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float mult = 1.0 / ((float)nbElems * sqrt(σ2[depth] + Ɛ));
    float dxhat = Ɣ[depth] * delta[offset];
    float tmp1 = nbElems * dxhat;
    float tmp2 = sum1[depth];
    float tmp3 = xHat[offset] * sum2[depth];
    
    delta[offset] = mult * (tmp1 - tmp2 - tmp3);
}

kernel void backwardBNConvInference(
    const device float * Ɣ,
    const device float * Eσ2,
    constant uint * pNbChannels,
    constant uint * pNbBatch,
    constant uint * pM,
    constant uint * pDimensions,
    device float * delta,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbBatch;
    uint m;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbChannels && pNbBatch && pM && pDimensions && Ɣ &&
        Eσ2 && delta)
    {
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        m = *pM;
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
    
    float Var = Eσ2[depth];
    if (m > 1)
    {
        Var *= (float)m / ((float)m - 1);
    }
    float tmp1 = delta[offset];
    float tmp2 = sqrt(Var + Ɛ);
    float xhat = tmp1 / tmp2;
    delta[offset] = Ɣ[depth] * xhat;
}
