import torch


class GCount:
    """
    Calcule les features [C0, Cδ] pour tous les batchs d'un dataset.
    Utilisation :
        gcount = GCount(vocab_size=..., delta=..., device=...)
        features = gcount.transform(dataset)
    """
    def __init__(self, vocab_size, delta, device='cuda'):
        self.vocab_size = vocab_size
        self.delta = delta
        self.device = device

    def _features_single_batch(self, batch):
        """
        Calcule [C0, Cδ] pour chaque séquence d’un batch.
        - batch : (batch_size, context_size), indices de tokens
        Retour : (batch_size, 2 * vocab_size)
        """
        batch = batch.to(self.device)
        batch_size, context_size = batch.shape

        # Encodage one-hot pour compter facilement chaque token
        one_hot = torch.nn.functional.one_hot(batch, num_classes=self.vocab_size).float()
        print(one_hot)

        # Comptage brut
        C0 = one_hot.sum(dim=1)
        print("C0 ", C0)

        # Facteur de décay exponentiel pour chaque position (dernier=0)
        deltas = torch.arange(context_size - 1, -1, -1, device=self.device)
        print("les deltas t  ∆t ", deltas)
        decay_factors = torch.exp(-deltas.float() / self.delta).view(1, context_size, 1)
        print("le decay ", decay_factors)

        # Comptage décayé
        weighted_one_hot = one_hot * decay_factors
        print(weighted_one_hot)
        Cdelta = weighted_one_hot.sum(dim=1)

        # On concatène [C0, Cδ] pour chaque séquence du batch
        features = torch.cat([C0, Cdelta], dim=1)
        return features

    def transform(self, dataset):
        """
        Traite tout le dataset (itérable sur les batchs).
        Retourne un tensor de shape (nb_sequences_total, 2 * vocab_size)
        """
        all_features = []
        for batch in dataset:
            print("batch ", batch)
            features = self._features_single_batch(batch)
            all_features.append(features.cpu())
        if all_features:
            return torch.cat(all_features, dim=0)
        else:
            return torch.empty(0, 2 * self.vocab_size)
