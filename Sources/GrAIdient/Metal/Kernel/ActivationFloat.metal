//
// Activation.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void forwardReLUFloat(
    constant uint & nbElems,
    device float * tmps,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
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

kernel void forwardReLUInferenceFloat(
    constant uint & nbElems,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
    if (id >= nbElems)
    {
        return ;
    }
    
    float tmp = outs[id];
    if (tmp < 0)
    {
        outs[id] = 0.0;
    }
    else
    {
        outs[id] = tmp;
    }
}

kernel void backwardReLUFloat(
    const device float * tmps,
    constant uint & nbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    if (id >= nbElems)
    {
        return ;
    }
    
    if (tmps[id] < 0)
    {
        delta[id] = 0.0;
    }
}

kernel void forwardLeakyReLUFloat(
    constant uint & nbElems,
    device float * tmps,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
    float Ɛ = 0.01;
    
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

kernel void forwardLeakyReLUInferenceFloat(
    constant uint & nbElems,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
    float Ɛ = 0.01;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float tmp = outs[id];
    if (outs[id] < 0)
    {
        outs[id] = Ɛ * tmp;
    }
    else
    {
        outs[id] = tmp;
    }
}

kernel void backwardLeakyReLUFloat(
    const device float * tmps,
    constant uint & nbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    float Ɛ = 0.01;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    if (tmps[id] < 0)
    {
        delta[id] = Ɛ * delta[id];
    }
}

kernel void forwardSoftReLUFloat(
    constant uint & nbElems,
    device float * tmps,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
    float Ɛ = 0.01;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    tmps[id] = outs[id];
    outs[id] = Ɛ * tmps[id] + (1 - Ɛ) * log(1 + exp(tmps[id]));
}

kernel void forwardSoftReLUInferenceFloat(
    constant uint & nbElems,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
    float Ɛ = 0.01;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float tmp = outs[id];
    outs[id] = Ɛ * tmp + (1 - Ɛ) * log(1 + exp(tmp));
}

kernel void backwardSoftReLUFloat(
    const device float * tmps,
    constant uint & nbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    float Ɛ = 0.01;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float derivative = Ɛ + (1 - Ɛ) / (1 + exp(-tmps[id]));
    delta[id] = delta[id] * derivative;
}

kernel void forwardSigmoidFloat(
   constant uint & nbElems,
   device float * tmps,
   device float * outs,
   uint id [[ thread_position_in_grid ]])
{
    if (id >= nbElems)
    {
        return ;
    }
    
    tmps[id] = outs[id];
    if (tmps[id] >= 0)
    {
        outs[id] = 1.0 / (1.0 + exp(-tmps[id]));
    }
    else
    {
        outs[id] = exp(tmps[id]) / (1.0 + exp(tmps[id]));
    }
}

kernel void forwardSigmoidInferenceFloat(
   constant uint & nbElems,
   device float * outs,
   uint id [[ thread_position_in_grid ]])
{
    if (id >= nbElems)
    {
        return ;
    }
    
    float tmp = outs[id];
    if (tmp >= 0)
    {
        outs[id] = 1.0 / (1.0 + exp(-tmp));
    }
    else
    {
        outs[id] = exp(tmp) / (1.0 + exp(tmp));
    }
}

kernel void backwardSigmoidFloat(
    const device float * tmps,
    constant uint & nbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    if (id >= nbElems)
    {
        return ;
    }
    
    float tmp;
    if (tmps[id] >= 0)
    {
        tmp = 1.0 / (1.0 + exp(-tmps[id]));
    }
    else
    {
        tmp = exp(tmps[id]) / (1.0 + exp(tmps[id]));
    }
    
    float derivative = tmp * (1 - tmp);
    delta[id] = delta[id] * derivative;
}

kernel void forwardGELUApproxFloat(
   constant uint & nbElems,
   device float * tmps,
   device float * outs,
   uint id [[ thread_position_in_grid ]])
{
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

kernel void forwardGELUApproxInferenceFloat(
   constant uint & nbElems,
   device float * outs,
   uint id [[ thread_position_in_grid ]])
{
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
    outs[id] = 0.5 * x * (1 + tmp2);
}

kernel void backwardGELUApproxFloat(
    const device float * tmps,
    constant uint & nbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
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

/*
 * Approximation to the error function.
 * Based on code from:
 * https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff#answer-35148199
 */
float erf(float a)
{
    float r, s, t, u;
    t = metal::abs(a);
    s = a * a;
    if (t > 0.927734375f)
    {
        // maximum error 0.99527 ulp
        r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
        u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
        r = metal::fma(r, s, u);
        r = metal::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
        r = metal::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
        r = metal::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
        r = metal::fma(r, t, -t);
        // TODO, replace with expm1 when implemented
        r = 1.0f - metal::exp(r);
        r = metal::copysign(r, a);
    }
    else
    {
        // maximum error 0.98929 ulp
        r = -5.96761703e-4f; // -0x1.38e000p-11
        r = metal::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
        r = metal::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
        r = metal::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
        r = metal::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
        r = metal::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
        r = metal::fma(r, a, a);
    }
    return r;
}

kernel void forwardGELUFloat(
   constant uint & nbElems,
   device float * tmps,
   device float * outs,
   uint id [[ thread_position_in_grid ]])
{
    if (id >= nbElems)
    {
        return ;
    }
    
    float x = outs[id];
    tmps[id] = x;
    outs[id] = 0.5 * x * (1 + erf(x / sqrt(2.0)));
}

kernel void forwardGELUInferenceFloat(
   constant uint & nbElems,
   device float * outs,
   uint id [[ thread_position_in_grid ]])
{
    if (id >= nbElems)
    {
        return ;
    }
    
    float x = outs[id];
    outs[id] = 0.5 * x * (1 + erf(x / sqrt(2.0)));
}

kernel void backwardGELUFloat(
    const device float * tmps,
    constant uint & nbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    if (id >= nbElems)
    {
        return ;
    }
    
    float x = tmps[id];
    float tmp1 = 0.5 * (1.0 + erf(x / sqrt(2.0)));
    float tmp2 = x / sqrt(2.0 * M_PI_F) * exp(-x * x / 2.0);
    float derivative = tmp1 + tmp2;
    delta[id] = delta[id] * derivative;
}
