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

Note that one way to compute gradient checking could be to duplicate the 
model and operate one single weight modification per model copy and run the 
forward pass on each and every one of these copies. The problem with this 
solution is that the whole estimation logic appears at the `Model` level.

The current solution prefers to deletage the weight modification on the 
`Layer` component because the `Layer` is already 
the low level component where the signal flows `forward` and `backward`. 
Thus, the `forwardGC` only appears as a new way to make the signal flow.

## Next Chapter

Previous chapter: [Optimizer](OPTIMIZER.md). \
Next chapter: [Plugin](PLUGIN.md).
