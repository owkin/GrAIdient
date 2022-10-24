# Gradient Checking

Gradient checking is a way to estimage the gradient of weights 
thanks to the following approximation: 

$$ \delta w = \nabla_{w}Loss \approx \frac{Loss(w+h) - Loss(w-h)}{2h} $$

This estimation allows to challenge the theoretical gradients given 
by `MAKit` during the backward pass.

`MAKit` implements gradient checking tests for the different available layers 
to ensure their backward API to operate the right gradient formula 
according to their forward API. We want the following difference to be 
very small: 
 
$$ 
\frac{\left\Vert \delta w^{approx} - \delta w^{MAKit} \right\Vert_{2}}
{\left\Vert \delta w^{approx} \right\Vert_{2} + 
 \left\Vert \delta w^{MAKit} \right\Vert_{2}} = \epsilon 
$$

## Next Chapter

Previous chapter: [Optimizer](OPTIMIZER.md). \
Next chapter: [Plugin](PLUGIN.md).
