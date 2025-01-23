import unittest
import numpy.testing as npt


class TestDoublePoisson(unittest.TestCase):
    def test_poisson_case(self):
        from differentiable_likelihoods.double_poisson import DoublePoisson
        from torch.distributions import Poisson
        import torch

        rate = torch.tensor([3.0])
        disp = torch.tensor([1.0])
        double_poisson = DoublePoisson(rate, disp)
        poisson = Poisson(rate)

        sample = poisson.sample([10])
        log_prob_double_poisson = double_poisson.log_prob(sample)
        log_prob_poisson = poisson.log_prob(sample)
        npt.assert_array_almost_equal(log_prob_double_poisson, log_prob_poisson)
        breakpoint()


if __name__ == "__main__":
    unittest.main()