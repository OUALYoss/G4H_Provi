import torch


class GCount:
    """
    Calcule les features [C0, Cδ] pour un batch de séquences d'indices.
    - C0 : comptage brut
    - Cδ : comptage décayé exponentiel, base e, demi-vie delta
    """
    def __init__(self, vocab_size, delta, device='cpu'):
        self.vocab_size = vocab_size
        self.delta = delta
        self.device = device

    def __call__(self, batch):
        """
        batch: Tensor shape (batch_size, context_size) d'indices de tokens
        Retourne: Tensor shape (batch_size, 2 * vocab_size)
        """
        batch = batch.to(self.device)
        batch_size, context_size = batch.shape

        # One-hot encoding : (batch_size, context_size, vocab_size)
        one_hot = torch.nn.functional.one_hot(batch, num_classes=self.vocab_size).float()

        # --- Comptage brut C0
        C0 = one_hot.sum(dim=1)  # (batch_size, vocab_size)

        # --- Comptage décayé Cδ
        # Δt : [context_size-1, ..., 0] (dernier événement = t=0)
        positions = torch.arange(context_size, device=self.device)
        deltas = (context_size - 1) - positions  # (context_size, )
        decay_factors = torch.exp(-deltas.float() / self.delta)  # (context_size,)
        decay_factors = decay_factors.view(1, context_size, 1)  # broadcast

        weighted_one_hot = one_hot * decay_factors  # (batch_size, context_size, vocab_size)
        Cdelta = weighted_one_hot.sum(dim=1)  # (batch_size, vocab_size)

        # --- Concatenation
        features = torch.cat([C0, Cdelta], dim=1)  # (batch_size, 2*vocab_size)
        return features
