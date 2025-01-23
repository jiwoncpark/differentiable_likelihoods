from torch.distributions import Poisson


if __name__ == "__main__":
    import torch
    m = Poisson(torch.tensor([4.0]))
    m.sample()
    breakpoint()

    m = Poisson(4)
    m.sample()
    breakpoint()