import torch


class SVD_PPMI_OPTI:
    def __init__(self, vocab_size, window_size=2, embedding_dim=100, device='cuda'):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.pad_idx = vocab_size+1
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.embeddings = None

    def compute_cooccurrence_matrix(self, batch_iterable):
        cooc = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.float32, device=self.device)
        offsets = [i for i in range(-self.window_size, self.window_size+1) if i != 0]

        for batch in batch_iterable:  # ***SEULE BOUCLE***
            batch = batch.to(self.device)  # (batch_size, context_size)
            print("batch ", batch)
            mask = (batch != self.pad_idx)
            for offset in offsets:
                # Décalage contextuel
                context = torch.roll(batch, shifts=offset, dims=1)
                context_mask = torch.roll(mask, shifts=offset, dims=1)
                valid = mask & context_mask
                targets = batch[valid]
                contexts = context[valid]
                # Accumulation vectorisée des cooccurrences
                cooc.index_put_(
                    (targets, contexts),
                    torch.ones_like(targets, dtype=torch.float32, device=self.device),
                    accumulate=True
                )
        return cooc

    def compute_ppmi_matrix(self, cooc, eps=1e-8):
        total_sum = cooc.sum()
        row_sums = cooc.sum(dim=1, keepdim=True)
        col_sums = cooc.sum(dim=0, keepdim=True)
        denom = row_sums @ col_sums
        ppmi = torch.log((cooc * total_sum + eps) / (denom + eps))
        ppmi = torch.where(torch.isnan(ppmi) | (ppmi < 0), torch.zeros_like(ppmi), ppmi)
        return ppmi

    def compute_svd_embeddings(self, ppmi):
        U, S, Vh = torch.linalg.svd(ppmi, full_matrices=False)
        U = U[:, :self.embedding_dim]
        S = S[:self.embedding_dim]
        embeddings = U * S.unsqueeze(0)
        return embeddings

    def fit(self, batch_iterable):
        print(">> Calcul de la matrice de cooccurrence ...")
        cooc = self.compute_cooccurrence_matrix(batch_iterable)
        print("coocureence ", cooc)
        print(">> Calcul de la matrice PPMI ...")
        ppmi = self.compute_ppmi_matrix(cooc)
        print(">> Calcul SVD pour les embeddings ...")
        self.embeddings = self.compute_svd_embeddings(ppmi)
        print(">> Embeddings shape:", self.embeddings.shape)
        return self.embeddings

    def save_embeddings(self, path):
        if self.embeddings is not None:
            torch.save(self.embeddings.cpu(), path)
            print(f">> Embeddings sauvegardés sous : {path}")
        else:
            print(">> Aucun embeddings à sauvegarder.")
