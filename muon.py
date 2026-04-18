import torch


@torch.no_grad()
def _newton_schulz(G, steps=5, eps=1e-7):
    # Approximates the orthogonal factor U @ V.T of the SVD G = U S V.T via
    # a quintic polynomial iteration. Operates on the smaller dim by
    # transposing tall matrices.
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz.

    Only supports 2D matrix parameters. Use AdamW for 1D parameters
    (biases, norm scales) and for embeddings / LM heads — see Keller
    Jordan's paper for the rationale.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise RuntimeError(
                        f"Muon only supports 2D parameters; got shape {tuple(p.shape)}"
                    )
                g = p.grad

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                update = _newton_schulz(update, steps=ns_steps)
                # Equalize update magnitude across different matrix shapes.
                scale = max(1.0, update.shape[0] / update.shape[1]) ** 0.5
                p.add_(update, alpha=-lr * scale)
        return loss
