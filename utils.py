import torch
import torch.nn as nn


def gradient_penalty(critic, labels, real, fake, device="cpu"):
    BATCH_SIZE, C, seq_size = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, seq_size).to(device)
    interpolated_seqs = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_seqs, labels)

    # Take the gradient of the scores with respect to the seqs
    gradient = torch.autograd.grad(
        inputs=interpolated_seqs,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])
