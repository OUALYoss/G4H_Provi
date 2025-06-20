import torch


class SVD_PPMI_OPTIMA:
    def __init__(self, vocab_size, window_size=2, embedding_dim=100, device='cuda'):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.embeddings = None

    def build_cooc_matrix(self, dataset):
        cooc = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.float32, device=self.device)
        total_flat_tokens = []
        cp = 0
        for batch in dataset:
            batch = batch.to(self.device)
            cp += 1
            print("batch "+str(cp), batch)
            B, L = batch.shape

            for offset in range(1, self.window_size + 1):
                left = batch[:, :-offset]
                right = batch[:, offset:]

                indices_ij = torch.stack([left.reshape(-1), right.reshape(-1)])
                cooc.index_put_(tuple(indices_ij), torch.ones(indices_ij.shape[1], device=self.device), accumulate=True)

                indices_ji = torch.stack([right.reshape(-1), left.reshape(-1)])
                cooc.index_put_(tuple(indices_ji), torch.ones(indices_ji.shape[1], device=self.device), accumulate=True)

            total_flat_tokens.append(batch.flatten())

        all_tokens = torch.cat(total_flat_tokens)
        counts = torch.bincount(all_tokens, minlength=self.vocab_size)
        cooc[range(self.vocab_size), range(self.vocab_size)] = counts.float()  # <-- correction ici

        return cooc

    def ppmi_matrix(self, cooc):
        total = cooc.sum()  #N
        sum_over_rows = cooc.sum(dim=1, keepdim=True) 
        sum_over_cols = cooc.sum(dim=0, keepdim=True)
        expected = sum_over_rows @ sum_over_cols / total.clamp_min(1e-8)
        ppmi = torch.log((cooc + 1e-8) / (expected + 1e-8))
        ppmi = torch.clamp(ppmi, min=0)# Transforme la PMI en PPMI 
        ppmi[torch.isnan(ppmi)] = 0
        return ppmi

    def svd_embeddings(self, ppmi):
        # U matrice ortho
        # S vecteur des valeurs singulières
        # Vh matrice diag
        U, S, Vh = torch.linalg.svd(ppmi, full_matrices=False) 
        emb = U[:, :self.embedding_dim] * S[:self.embedding_dim].unsqueeze(0)  # Φ[i,j]=U[i,j]⋅S[j]
        return emb

    def fit(self, dataset):
        cooc = self.build_cooc_matrix(dataset)
        print('Cooccurrence Matrix:\n', cooc.cpu().numpy())
        ppmi = self.ppmi_matrix(cooc)
        print('PPMI Matrix:\n', ppmi.cpu().numpy())
        emb = self.svd_embeddings(ppmi)
        print('Embeddings shape:', emb.shape)
        self.embeddings = emb
        return emb
