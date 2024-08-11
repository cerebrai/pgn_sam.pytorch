""" Implementation of Penalizing Gradient Norm for Efficiently Improving
Generalization in Deep Learning
"""

from typing import Iterable
from copy import deepcopy

import torch
from torch.optim._multi_tensor import SGD

__all__ = ["PGN_SAMSGD"]


class PGN_SAMSGD(SGD):
    """ SGD wrapped with Penalized Gradient Norm Sharp-Aware Minimization

    Args:
        params: tensors to be optimized
        lr: learning rate
        momentum: momentum factor
        dampening: damping factor
        weight_decay: weight decay factor
        nesterov: enables Nesterov momentum
        rho: neighborhood size
        alpha: weighted average factor between the original gradient and the penality.

    """

    def __init__(self,
                 params: Iterable[torch.Tensor],
                 lr: float,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 rho: float = 0.05,
                 alpha: float = 0.1,
                 ):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")

        if alpha < 0:
            raise ValueError(f"Invalid weighting factor: {alpha}")

        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

        # todo: generalize this
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.param_groups[0]["rho"] = rho
        self.param_groups[0]["alpha"] = alpha

    @torch.no_grad()
    def step(self,
             closure
             ) -> torch.Tensor:
        """

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns: the loss value evaluated on the original point

        """
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:

            rho = group['rho']
            alpha = group['alpha']

            # -- Getting the original grads, i.e., g1
            original_grads, params_with_grads = self._get_gradients(group)
            device = original_grads[0].device

            # compute \hat{\epsilon}=\rho/\norm{g}\|g\|
            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in original_grads]).norm(2)
            epsilon = deepcopy(original_grads)
            torch._foreach_mul_(epsilon, rho / grad_norm)

            # virtual step toward \epsilon
            torch._foreach_add_(params_with_grads, epsilon)

            # Get the updated grads, i.e., g2
            closure()
            updated_grads, _ = self._get_gradients(group)

            # virtual step back to the original point
            torch._foreach_sub_(params_with_grads, epsilon)

            # compute the penalized gradient
            final_gradient = [alpha*grad1 + (1-alpha)*grad2 for grad1, grad2 in zip(original_grads, updated_grads)]

            # Update the gradient values for SGD with final_gradient
            with torch.no_grad():
                for param, grad in zip(params_with_grads, final_gradient):
                    param.grad = grad

        super().step()
        return loss

    def _get_gradients(self, group):
        """ Collects the gradients of the parameters.
        """

        grads = []
        params_with_grads = []

        for p in group['params']:
            if p.grad is not None:
                # without clone().detach(), p.grad will be zeroed by closure()
                grads.append(p.grad.clone().detach())
                params_with_grads.append(p)

        return grads, params_with_grads
