import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class SVD_PPMI_OPTI:
    def __init__(self, vocab_size, window_size=2, embedding_dim=100, pad_idx=0):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx

    def accumulate_cooc_sparse(self, dataloader):
        # Utilise un dict pour la COO (sparse) : (i,j):count
        cooc = {}
        for batch in dataloader:
            # batch : (batch_size, seq_len)
            batch_np = batch.numpy() if torch.is_tensor(batch) else batch
            mask = batch_np != self.pad_idx
            batch_size, seq_len = batch_np.shape
            for offset in range(1, self.window_size+1):
                # Décale la matrice pour context gauche et droite, vectorisé
                for dir in [-1, 1]:
                    idxs = np.arange(seq_len)
                    context_pos = idxs + offset*dir
                    valid = (context_pos >= 0) & (context_pos < seq_len)
                    targets = batch_np[:, idxs[valid]]
                    contexts = batch_np[:, context_pos[valid]]
                    mask_pair = mask[:, idxs[valid]] & mask[:, context_pos[valid]]
                    # On évite les boucles : on empile tout
                    for t, c, m in zip(targets.flatten(), contexts.flatten(), mask_pair.flatten()):
                        if m and t != self.pad_idx and c != self.pad_idx:
                            key = (t, c)
                            cooc[key] = cooc.get(key, 0) + 1
        return cooc

    def cooc_to_sparse_matrix(self, cooc):
        rows, cols, data = zip(*[(i, j, v) for (i, j), v in cooc.items()])
        M = sparse.coo_matrix((data, (rows, cols)), shape=(self.vocab_size, self.vocab_size))
        return M

    def compute_ppmi_matrix(self, cooc_matrix, eps=1e-8):
        total = cooc_matrix.sum()
        sum_over_rows = np.array(cooc_matrix.sum(axis=1)).flatten()
        sum_over_cols = np.array(cooc_matrix.sum(axis=0)).flatten()
        rows, cols = cooc_matrix.nonzero()
        vals = cooc_matrix.data
        pmi = np.log((vals * total) / (sum_over_rows[rows] * sum_over_cols[cols]) + eps)
        ppmi = np.maximum(pmi, 0)
        return sparse.coo_matrix((ppmi, (rows, cols)), shape=cooc_matrix.shape)

    def compute_svd(self, ppmi_matrix):
        svd = TruncatedSVD(n_components=self.embedding_dim)
        embeddings = svd.fit_transform(ppmi_matrix)
        return embeddings

    def fit(self, dataloader):
        print(">> Accumulation vectorisée des cooccurrences ...")
        cooc = self.accumulate_cooc_sparse(dataloader)
        print(">> Conversion en sparse matrix ...")
        cooc_matrix = self.cooc_to_sparse_matrix(cooc)
        print(">> Calcul PPMI ...")
        ppmi_matrix = self.compute_ppmi_matrix(cooc_matrix)
        print(">> SVD/Embeddings ...")
        embeddings = self.compute_svd(ppmi_matrix)
        print(">> Embeddings shape:", embeddings.shape)
        return embeddings