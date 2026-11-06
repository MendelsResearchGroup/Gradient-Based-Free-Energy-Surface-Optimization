import torch
from torch.utils.checkpoint import checkpoint

from model import symmetric_zero_diagonal


def symmetric_noise(batch_size, device, dtype):
    """Independent N(0,1) noise on the upper triangle, mirrored below it."""
    x = torch.randn(batch_size, 10, 10, device=device, dtype=dtype)
    x = torch.triu(x, diagonal=1)
    return x + x.transpose(-1, -2)


class DifferentiableDDIMSampler:
    """DDIM sampler retaining gradients through every step to the epsilons."""

    def __init__(
        self,
        model,
        diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        self.model = model
        self.diffusion_steps = diffusion_steps

        parameter = next(model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype

        betas = torch.linspace(
            beta_start,
            beta_end,
            diffusion_steps,
            device=self.device,
            dtype=self.dtype,
        )
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0)

    def _timesteps(self, num_steps):
        if not 1 <= num_steps <= self.diffusion_steps:
            raise ValueError(
                f"num_steps must be in [1, {self.diffusion_steps}]"
            )

        # Includes t=0 and t=T-1. With num_steps <= T, rounding is unique.
        return torch.linspace(
            0,
            self.diffusion_steps - 1,
            num_steps,
            device=self.device,
        ).round().long().flip(0)

    def sample(
        self,
        epsilons,
        batch_size=None,
        num_steps=50,
        guidance_scale=2.0,
        x_T=None,
        checkpoint_steps=False,
    ):
        """Generate symmetric 10x10 frames with DDIM.

        Args:
            epsilons: [10] or [B,10], ordered by atom ID; may require grad.
            batch_size: Needed only when a shared [10] vector is used without x_T.
                        Defaults to 1 in that case.
            num_steps: Number of DDIM denoising steps.
            guidance_scale: Classifier-free guidance scale.
            x_T: Optional [B,10,10] initial latent. May require grad.
            checkpoint_steps: Recompute denoiser activations during backward to
                              save memory, without truncating sampler gradients.

        Returns:
            [B,10,10] frames, deterministic for fixed x_T and epsilons.
        """
        epsilons = torch.as_tensor(epsilons, device=self.device, dtype=self.dtype)
        if epsilons.ndim not in (1, 2) or epsilons.shape[-1] != 10:
            raise ValueError("epsilons must have shape [10] or [B,10]")
        if epsilons.ndim == 2:
            if batch_size is not None and batch_size != epsilons.shape[0]:
                raise ValueError("batch_size does not match epsilons")
            batch_size = epsilons.shape[0]

        if x_T is not None:
            if x_T.ndim != 3 or x_T.shape[1:] != (10, 10):
                raise ValueError("x_T must have shape [B,10,10]")
            inferred_batch = x_T.shape[0]
            if batch_size is not None and batch_size != inferred_batch:
                raise ValueError("batch_size does not match x_T")
            batch_size = inferred_batch
            x = x_T.to(device=self.device, dtype=self.dtype)
            x = symmetric_zero_diagonal(x)
        else:
            if batch_size is None:
                batch_size = 1
            if batch_size < 1:
                raise ValueError("batch_size must be positive")
            x = symmetric_noise(batch_size, self.device, self.dtype)

        if epsilons.ndim == 1:
            epsilons = epsilons[None].expand(batch_size, -1)
        timesteps = self._timesteps(num_steps)

        for i, t_value in enumerate(timesteps):
            t = torch.full(
                (batch_size,),
                int(t_value.item()),
                device=self.device,
                dtype=torch.long,
            )

            if checkpoint_steps and torch.is_grad_enabled():
                predicted_noise = checkpoint(
                    self.model.predict_cfg, x, t, epsilons,
                    guidance_scale=guidance_scale, use_reentrant=False,
                )
            else:
                predicted_noise = self.model.predict_cfg(
                    x, t, epsilons, guidance_scale=guidance_scale,
                )

            alpha_bar_t = self.alpha_bar[t_value]
            predicted_x0 = (
                x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)
            predicted_x0 = symmetric_zero_diagonal(predicted_x0)

            if i == len(timesteps) - 1:
                x = predicted_x0
            else:
                prev_t = timesteps[i + 1]
                alpha_bar_prev = self.alpha_bar[prev_t]

                direction = (1.0 - alpha_bar_prev).sqrt() * predicted_noise

                x = torch.sqrt(alpha_bar_prev) * predicted_x0 + direction
                x = symmetric_zero_diagonal(x)

        return x
