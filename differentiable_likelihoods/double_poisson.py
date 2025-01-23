# mypy: allow-untyped-defs
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.distributions import Poisson


__all__ = ["DoublePoisson"]


class DoublePoisson(ExponentialFamily):
    r"""
    Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by

    .. math::
      \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}

    Example::

        >>> # xdoctest: +SKIP("poisson_cpu not implemented for 'Long'")
        >>> m = Poisson(torch.tensor([4]))
        >>> m.sample()
        tensor([ 3.])

    Args:
        rate (Number, Tensor): the rate parameter
    """
    arg_constraints = {"rate": constraints.nonnegative}
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        return self.rate

    @property
    def mode(self):
        return self.rate.floor()

    @property
    def variance(self):
        return self.rate * self.disp

    def __init__(self, rate, disp, validate_args=None):
        self.rate, self.disp = broadcast_all(rate, disp)
        if isinstance(rate, Number) and isinstance(disp, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        self.poisson = Poisson(self.rate)
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Poisson, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        new.disp = self.disp.expand(batch_shape)
        super(Poisson, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # with torch.no_grad():
        #     return torch.poisson(self.rate.expand(shape), self.disp.expand(shape))
        raise NotImplementedError

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        rate, disp, value = broadcast_all(self.rate, self.disp, value)
        rate_logp = self.poisson.log_prob(value)
        value_logp = Poisson(value).log_prob(value)
        prod = rate * disp
        inv_c = 1.0 + (1 - disp) / (12 * prod) * (1 + 1/prod)
        log_c = torch.log(1.0 / inv_c)
        log_prob = log_c + 0.5*torch.log(disp) + \
              self.disp*rate_logp + (1 - disp)*value_logp
        return log_prob

    # @property
    # def _natural_params(self):
    #     return (torch.log(self.rate), self.disp)

    # def _log_normalizer(self, x):
    #     return torch.exp(x)


if __name__ == "__main__":
    import torch
    from torch.distributions import Poisson

    rate = torch.tensor([4.0])
    disp = torch.tensor([1.0])
    rate.requires_grad_(True)
    disp.requires_grad_(True)

    data = Poisson(rate).sample([10])
    # data.requires_grad_(True)

    log_prob = DoublePoisson(rate, disp).log_prob(data)

    grad = torch.autograd.grad(
        outputs=log_prob.sum(), 
        inputs=rate, is_grads_batched=False)[0]

    breakpoint()