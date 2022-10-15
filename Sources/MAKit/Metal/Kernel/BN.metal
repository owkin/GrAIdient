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
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pFirstCall,
    device float * μ,
    device float * Eμ,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint width;
    uint height;
    uint firstCall;
    
    if (pNbNeurons && pNbBatch && pDimensions && pFirstCall && tmps &&
        μ && Eμ)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        firstCall = *pFirstCall;
    }
    else
        return ;
    
    uint neuron = id;
    if (neuron >= nbNeurons)
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
                (neuron + nbNeurons * elem) * height;
            uint offset = y +
                (offsetStart + x) * width;
                
            sum += tmps[offset];
        }}
    }
    μ[neuron] = sum / nbElems;
    
    if (pFirstCall)
    {
        Eμ[neuron] = μ[neuron];
    }
    else
    {
        Eμ[neuron] = 0.9 * Eμ[neuron] + 0.1 * μ[neuron];
    }
}

kernel void computeConvσ2(
    const device float * tmps,
    const device float * μ,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pFirstCall,
    device float * σ2,
    device float * Eσ2,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint width;
    uint height;
    uint firstCall;
    
    if (pNbNeurons && pNbBatch && pDimensions && pFirstCall &&
        tmps && μ && σ2 && Eσ2)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        firstCall = *pFirstCall;
    }
    else
        return ;
    
    uint neuron = id;
    if (neuron >= nbNeurons)
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
                (neuron + nbNeurons * elem) * height;
            uint offset = y +
                (offsetStart + x) * width;
                
            float tmp = tmps[offset] - μ[neuron];
            sum += tmp * tmp;
        }}
    }
    σ2[neuron] = sum / nbElems;
    
    if (firstCall)
    {
        Eσ2[neuron] = σ2[neuron];
    }
    else
    {
        Eσ2[neuron] = 0.9 * Eσ2[neuron] + 0.1 * σ2[neuron];
    }
}

kernel void forwardBNConvTraining(
    const device float * β,
    const device float * Ɣ,
    const device float * μ,
    const device float * σ2,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device float * tmps,
    device float * xHat,
    uint3 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbNeurons && pNbBatch && pDimensions && β && Ɣ &&
        tmps && xHat && μ && σ2)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint neuron = id[2] % nbNeurons;
    uint elem = id[2] / nbNeurons;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurons * nbBatch)
    {
        return ;
    }
    
    uint offsetStart =
        (neuron + nbNeurons * elem) * height;
    uint offset = j +
        (offsetStart + i) * width;
    
    float tmp1 = tmps[offset] - μ[neuron];
    float tmp2 = sqrt(σ2[neuron] + Ɛ);
    float xhat = tmp1 / tmp2;
    xHat[offset] = xhat;
    tmps[offset] = Ɣ[neuron] * xhat + β[neuron];
}

kernel void forwardBNConvInference(
    const device float * β,
    const device float * Ɣ,
    const device float * Eμ,
    const device float * Eσ2,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pM,
    constant uint * pDimensions,
    device float * tmps,
    uint3 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint m;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbNeurons && pNbBatch && pM && pDimensions && β && Ɣ &&
        tmps && Eμ && Eσ2)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        m = *pM;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint neuron = id[2] % nbNeurons;
    uint elem = id[2] / nbNeurons;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurons * nbBatch)
    {
        return ;
    }
    
    uint offsetStart =
        (neuron + nbNeurons * elem) * height;
    uint offset = j +
        (offsetStart + i) * width;
    
    float Var = Eσ2[neuron];
    if (m > 1)
    {
        Var *= (float)m / ((float)m - 1);
    }
    float tmp1 = tmps[offset] - Eμ[neuron];
    float tmp2 = sqrt(Var + Ɛ);
    float xhat = tmp1 / tmp2;
    tmps[offset] = Ɣ[neuron] * xhat + β[neuron];
}

kernel void backwardWeightsBNConv(
    const device float * delta,
    const device float * xHat,
    const device float * Ɣ,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    constant uint * pAccumulate,
    device float * sum1,
    device float * sum2,
    device float * dƔ,
    device float * dβ,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint width;
    uint height;
    uint accumulate;
    
    if (pNbNeurons && pNbBatch && pDimensions && pAccumulate &&
        delta && xHat && Ɣ &&
        sum1 && sum2 && dƔ && dβ)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint neuron = id;
    if (neuron >= nbNeurons)
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
            uint offsetStart =
                (neuron + nbNeurons * elem) * height;
            uint offset = y +
                (offsetStart + x) * width;
                
            float deltaTmp = delta[offset];
            float xHatTmp = xHat[offset];
            float dxhat = Ɣ[neuron] * deltaTmp;
            tmp1 += dxhat;
            tmp2 += dxhat * xHatTmp;
            
            tmp3 += deltaTmp * xHatTmp;
            tmp4 += deltaTmp;
        }}
    }
    sum1[neuron] = tmp1;
    sum2[neuron] = tmp2;
    
    if (accumulate)
    {
        dƔ[neuron] += tmp3;
        dβ[neuron] += tmp4;
    }
    else
    {
        dƔ[neuron] = tmp3;
        dβ[neuron] = tmp4;
    }
}

kernel void backwardBNConvTraining(
    const device float * σ2,
    const device float * xHat,
    const device float * Ɣ,
    const device float * sum1,
    const device float * sum2,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pDimensions,
    device float * delta,
    uint3 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbNeurons && pNbBatch && pDimensions && σ2 && xHat && Ɣ &&
        delta)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint neuron = id[2] % nbNeurons;
    uint elem = id[2] / nbNeurons;
    uint nbElems = nbBatch * width * height;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurons * nbBatch)
    {
        return ;
    }
    
    uint offsetStart =
        (neuron + nbNeurons * elem) * height;
    uint offset = j +
        (offsetStart + i) * width;
    
    float mult = 1.0 / ((float)nbElems * sqrt(σ2[neuron] + Ɛ));
    float dxhat = Ɣ[neuron] * delta[offset];
    float tmp1 = nbElems * dxhat;
    float tmp2 = sum1[neuron];
    float tmp3 = xHat[offset] * sum2[neuron];
    
    delta[offset] = mult * (tmp1 - tmp2 - tmp3);
}

kernel void backwardBNConvInference(
    const device float * Ɣ,
    const device float * Eσ2,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pM,
    constant uint * pDimensions,
    device float * delta,
    uint3 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint m;
    uint width;
    uint height;
    float Ɛ = 1e-5;
    
    if (pNbNeurons && pNbBatch && pM && pDimensions && Ɣ &&
        Eσ2 && delta)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        m = *pM;
        width = pDimensions[0];
        height = pDimensions[1];
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint neuron = id[2] % nbNeurons;
    uint elem = id[2] / nbNeurons;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurons * nbBatch)
    {
        return ;
    }
    
    uint offsetStart =
        (neuron + nbNeurons * elem) * height;
    uint offset = j +
        (offsetStart + i) * width;
    
    float Var = Eσ2[neuron];
    if (m > 1)
    {
        Var *= (float)m / ((float)m - 1);
    }
    float tmp1 = delta[offset];
    float tmp2 = sqrt(Var + Ɛ);
    float xhat = tmp1 / tmp2;
    delta[offset] = Ɣ[neuron] * xhat;
}
