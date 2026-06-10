import math
import torch

class VESDE:
    """
    Variance-Exploding SDE (NCSN++/EDM-style) with log-linear noise schedule.
    Training perturbation: x_t = x_0 + sigma(t) * z,  z~N(0,I),  t in [0,1].
    VE SDE: d x = g(t) dW, where g(t) = sqrt(d/dt sigma(t)^2).
    Reverse SDE: d x = - g(t)^2 * score(x,t) dt + g(t) dar W.
    Probability flow ODE: d x = - 0.5 * g(t)^2 * score(x,t) dt.
    """
    def __init__(self, sde_config):
        sigma_min = sde_config['sigma_min']
        sigma_max = sde_config['sigma_max']
        T = 1.0
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.T = float(T)
        self._log_r = math.log(self.sigma_max / self.sigma_min + 1e-12)
        assert self._log_r > 0, "sigma_max must be > sigma_min"

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Log-linear schedule: sigma(t) = sigma_min * (sigma_max/sigma_min)^t."""
        return self.sigma_min * torch.exp(self._log_r * t)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) = sqrt( d/dt sigma^2 ) = sigma(t) * sqrt(2 * log_r)."""
        sigma_t = self.sigma(t)
        return sigma_t * math.sqrt(2.0 * self._log_r)

    def sde(self, x: torch.Tensor, t: torch.Tensor):
        drift = torch.zeros_like(x)
        g = self.diffusion(t).view(-1, *([1]*(x.dim()-1)))
        return drift, g

    def reverse(self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor):
        drift, g = self.sde(x, t)
        drift = drift - (g**2) * score
        return drift, g

    def perturb(self, x: torch.Tensor, t: torch.Tensor):
        sigma = self.sigma(t).view(-1, *([1]*(x.dim()-1)))
        noise = torch.randn_like(x)
        xt = x + sigma * noise
        return xt, noise, sigma

    def prior_sampling(self, shape, device=None):
        return torch.randn(shape, device=device) * self.sigma_max