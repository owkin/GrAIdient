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
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pFirstCall,
    device float * μ,
    device float * Eμ,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    uint width;
    uint height;
    uint firstCall;
    
    if (pNbNeurones && pNbBatch && pDimensions && pFirstCall && tmps &&
        μ && Eμ)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        firstCall = *pFirstCall;
    }
    else
        return ;
    
    uint neurone = id;
    if (neurone >= nbNeurones)
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
            uint offsetStart =
                (neurone + nbNeurones * elem) * height;
            uint offset = y +
                (offsetStart + x) * width;
                
            sum += tmps[offset];
        }}
    }
    μ[neurone] = sum / nbElems;
    
    if (pFirstCall)
    {
        Eμ[neurone] = μ[neurone];
    }
    else
    {
        Eμ[neurone] = 0.9 * Eμ[neurone] + 0.1 * μ[neurone];
    }
}

kernel void computeConvσ2(
    const device float * tmps,
    const device float * μ,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pFirstCall,
    device float * σ2,
    device float * Eσ2,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    uint width;
    uint height;
    uint firstCall;
    
    if (pNbNeurones && pNbBatch && pDimensions && pFirstCall &&
        tmps && μ && σ2 && Eσ2)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        firstCall = *pFirstCall;
    }
    else
        return ;
    
    uint neurone = id;
    if (neurone >= nbNeurones)
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
            uint offsetStart =
                (neurone + nbNeurones * elem) * height;
            uint offset = y +
                (offsetStart + x) * width;
                
            float tmp = tmps[offset] - μ[neurone];
            sum += tmp * tmp;
        }}
    }
    σ2[neurone] = sum / nbElems;
    
    if (firstCall)
    {
        Eσ2[neurone] = σ2[neurone];
    }
    else
    {
        Eσ2[neurone] = 0.9 * Eσ2[neurone] + 0.1 * σ2[neurone];
    }
}

kernel void forwardBNConvTraining(
    const device float * β,
    const device float * Ɣ,
    const device float * μ,
    const device float * σ2,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device float * tmps,
    device float * xHat,
    uint3 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbNeurones && pNbBatch && pDimensions && β && Ɣ &&
        tmps && xHat && μ && σ2)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint neurone = id[2] % nbNeurones;
    uint elem = id[2] / nbNeurones;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurones * nbBatch)
    {
        return ;
    }
    
    uint offsetStart =
        (neurone + nbNeurones * elem) * height;
    uint offset = j +
        (offsetStart + i) * width;
    
    float tmp1 = tmps[offset] - μ[neurone];
    float tmp2 = sqrt(σ2[neurone] + Ɛ);
    float xhat = tmp1 / tmp2;
    xHat[offset] = xhat;
    tmps[offset] = Ɣ[neurone] * xhat + β[neurone];
}

kernel void forwardBNConvInference(
    const device float * β,
    const device float * Ɣ,
    const device float * Eμ,
    const device float * Eσ2,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    constant uint * pM,
    constant uint * pDimensions,
    device float * tmps,
    uint3 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    uint m;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbNeurones && pNbBatch && pM && pDimensions && β && Ɣ &&
        tmps && Eμ && Eσ2)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
        m = *pM;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint neurone = id[2] % nbNeurones;
    uint elem = id[2] / nbNeurones;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurones * nbBatch)
    {
        return ;
    }
    
    uint offsetStart =
        (neurone + nbNeurones * elem) * height;
    uint offset = j +
        (offsetStart + i) * width;
    
    float Var = Eσ2[neurone];
    if (m > 1)
    {
        Var *= (float)m / ((float)m - 1);
    }
    float tmp1 = tmps[offset] - Eμ[neurone];
    float tmp2 = sqrt(Var + Ɛ);
    float xhat = tmp1 / tmp2;
    tmps[offset] = Ɣ[neurone] * xhat + β[neurone];
}

kernel void computeConvTmp(
    const device float * delta,
    const device float * xHat,
    const device float * Ɣ,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pAccumulate,
    device float * somme1,
    device float * somme2,
    device float * dƔ,
    device float * dβ,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    uint width;
    uint height;
    uint accumulate;
    
    if (pNbNeurones && pNbBatch && pDimensions && pAccumulate &&
        delta && xHat && Ɣ &&
        somme1 && somme2 && dƔ && dβ)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint neurone = id;
    if (neurone >= nbNeurones)
    {
        return ;
    }
    
    float sum1 = 0.0, sum2 = 0.0;
    float sum3 = 0.0, sum4 = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        for (uint x=0; x<width; x++){
        for (uint y=0; y<height; y++)
        {
            uint offsetStart =
                (neurone + nbNeurones * elem) * height;
            uint offset = y +
                (offsetStart + x) * width;
                
            float deltaTmp = delta[offset];
            float xHatTmp = xHat[offset];
            float dxhat = Ɣ[neurone] * deltaTmp;
            sum1 += dxhat;
            sum2 += dxhat * xHatTmp;
            
            sum3 += deltaTmp * xHatTmp;
            sum4 += deltaTmp;
        }}
    }
    somme1[neurone] = sum1;
    somme2[neurone] = sum2;
    
    if (accumulate)
    {
        dƔ[neurone] += sum3;
        dβ[neurone] += sum4;
    }
    else
    {
        dƔ[neurone] = sum3;
        dβ[neurone] = sum4;
    }
}

kernel void backwardBNConvTraining(
    const device float * σ2,
    const device float * xHat,
    const device float * Ɣ,
    const device float * somme1,
    const device float * somme2,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device float * delta,
    uint3 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbNeurones && pNbBatch && pDimensions && σ2 && xHat && Ɣ &&
        delta)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint neurone = id[2] % nbNeurones;
    uint elem = id[2] / nbNeurones;
    uint nbElems = nbBatch * width * height;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurones * nbBatch)
    {
        return ;
    }
    
    uint offsetStart =
        (neurone + nbNeurones * elem) * height;
    uint offset = j +
        (offsetStart + i) * width;
    
    float mult = 1.0 / ((float)nbElems * sqrt(σ2[neurone] + Ɛ));
    float dxhat = Ɣ[neurone] * delta[offset];
    float tmp1 = nbElems * dxhat;
    float tmp2 = somme1[neurone];
    float tmp3 = xHat[offset] * somme2[neurone];
    
    delta[offset] = mult * (tmp1 - tmp2 - tmp3);
}

kernel void backwardBNConvInference(
    const device float * Ɣ,
    const device float * Eσ2,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    constant uint * pM,
    constant uint * pDimensions,
    device float * delta,
    uint3 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    uint m;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbNeurones && pNbBatch && pM && pDimensions && Ɣ &&
        Eσ2 && delta)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
        m = *pM;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint neurone = id[2] % nbNeurones;
    uint elem = id[2] / nbNeurones;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurones * nbBatch)
    {
        return ;
    }
    
    uint offsetStart =
        (neurone + nbNeurones * elem) * height;
    uint offset = j +
        (offsetStart + i) * width;
    
    float Var = Eσ2[neurone];
    if (m > 1)
    {
        Var *= (float)m / ((float)m - 1);
    }
    float tmp1 = delta[offset];
    float tmp2 = sqrt(Var + Ɛ);
    float xhat = tmp1 / tmp2;
    delta[offset] = Ɣ[neurone] * xhat;
}
