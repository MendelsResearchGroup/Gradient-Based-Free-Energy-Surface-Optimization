"""Bounded condition map for the selected bead epsilons."""

import torch


class ReparameterizedCondition(torch.nn.Module):
    """Train x and two affine layers to produce the selected bead epsilons.

    There is no activation between the layers: this overparameterizes the
    condition without changing its physical meaning. The allowed domain is
    the box defined by the supplied lower and upper limits. A straight-through
    clamp uses the affine map's gradient, including at/outside the bounds;
    this is a surrogate gradient, not the derivative of a hard clamp.
    """

    def __init__(self, starting_eps, lower, upper, hidden_dim=32):
        super().__init__()
        self.register_buffer("lower", lower.detach().clone().to(starting_eps))
        self.register_buffer("upper", upper.detach().clone().to(starting_eps))
        self.x = torch.nn.Parameter(starting_eps.detach().clone())
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(starting_eps.numel(), hidden_dim),
            torch.nn.Linear(hidden_dim, starting_eps.numel()),
        ).to(device=starting_eps.device, dtype=starting_eps.dtype)
        with torch.no_grad():
            # Start at the supplied epsilons; nonzero weights let all parts train.
            self.layers[-1].bias.add_(starting_eps - self.layers(self.x))
        self.project_()

    def forward(self):
        raw = self.layers(self.x)
        bounded = raw.clamp(min=self.lower, max=self.upper)
        return bounded.detach() + (raw - raw.detach())

    @torch.no_grad()
    def project_(self):
        """Remove output overshoot after each update by shifting the final bias."""
        raw = self.layers(self.x)
        bounded = raw.clamp(min=self.lower, max=self.upper)
        self.layers[-1].bias.add_(bounded - raw)
