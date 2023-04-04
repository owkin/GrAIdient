//
// Activation.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void forwardReLU(
    constant uint * pNbElems,
    device float * tmps,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    tmps[id] = outs[id];
    if (tmps[id] < 0)
    {
        outs[id] = 0.0;
    }
    else
    {
        outs[id] = tmps[id];
    }
}

kernel void backwardReLU(
    const device float * tmps,
    constant uint * pNbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    if (tmps[id] < 0)
    {
        delta[id] = 0.0;
    }
}

kernel void forwardLeakyReLU(
    constant uint * pNbElems,
    device float * tmps,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float Ɛ = 0.01;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    tmps[id] = outs[id];
    if (tmps[id] < 0)
    {
        outs[id] = Ɛ * tmps[id];
    }
    else
    {
        outs[id] = tmps[id];
    }
}

kernel void backwardLeakyReLU(
    const device float * tmps,
    constant uint * pNbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float Ɛ = 0.01;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    if (tmps[id] < 0)
    {
        delta[id] = Ɛ * delta[id];
    }
}

kernel void forwardSoftReLU(
    constant uint * pNbElems,
    device float * tmps,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float Ɛ = 0.01;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    tmps[id] = outs[id];
    outs[id] = Ɛ * tmps[id] + (1 - Ɛ) * log(1 + exp(tmps[id]));
}

kernel void backwardSoftReLU(
    const device float * tmps,
    constant uint * pNbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float Ɛ = 0.01;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float derivative = Ɛ + (1 - Ɛ) / (1 + exp(-tmps[id]));
    delta[id] = delta[id] * derivative;
}

kernel void forwardSigmoid(
   constant uint * pNbElems,
   device float * tmps,
   device float * outs,
   uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    tmps[id] = outs[id];
    outs[id] = 1.0 / (1.0 + exp(-tmps[id]));
}

kernel void backwardSigmoid(
    const device float * tmps,
    constant uint * pNbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float tmp = 1.0 / (1.0 + exp(-tmps[id]));
    float derivative = tmp * (1 - tmp);
    delta[id] = delta[id] * derivative;
}

kernel void forwardGELU(
   constant uint * pNbElems,
   device float * tmps,
   device float * outs,
   uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float cst = sqrt(2.0 / 3.14159);
    float x = outs[id];
    float tmp1 = cst * (x + 0.044715 * pow(x, 3));
    float tmp2;
    if (tmp1 >= 0)
    {
        tmp2 = (1.0 - exp(-2.0 * tmp1)) / (1.0 + exp(-2.0 * tmp1));
    }
    else
    {
        tmp2 = (exp(2.0 * tmp1) - 1.0) / (exp(2.0 * tmp1) + 1.0);
    }
    tmps[id] = x;
    outs[id] = 0.5 * x * (1 + tmp2);
}

kernel void backwardGELU(
    const device float * tmps,
    constant uint * pNbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float cst = sqrt(2.0 / 3.14159);
    float x = tmps[id];
    float tmp1 = cst * (x + 0.044715 * pow(x, 3));
    float tmp2;
    if (tmp1 >= 0)
    {
        tmp2 = (1.0 - exp(-2.0 * tmp1)) / (1.0 + exp(-2.0 * tmp1));
    }
    else
    {
        tmp2 = (exp(2.0 * tmp1) - 1.0) / (exp(2.0 * tmp1) + 1.0);
    }
    float tmp3 = cst * (1 + 3 * 0.044715 * x * x) * (1 - tmp2 * tmp2);
    float derivative = 0.5 * (1 + tmp2 + x * tmp3);
    delta[id] = delta[id] * derivative;
}
