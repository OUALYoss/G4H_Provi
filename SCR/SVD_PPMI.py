import torch


class SVDPPMI:
    def __init__(self, vocab_size, window_size=2, embedding_dim=100, device='cuda'):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.co_occurrence_matrix = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=self.device)

    def compute_co_occurrence_matrix(self, dataset):
        for batch in dataset:
            batch = batch.to(self.device)
            mask = (batch != 0)
            indices = torch.arange(batch.size(1)).unsqueeze(0).expand_as(batch).to(self.device)
            context_windows = indices.unsqueeze(2) - indices.unsqueeze(1)
            context_mask = (context_windows.abs() <= self.window_size) & (context_windows != 0)

            target_words = batch.unsqueeze(2).expand(-1, -1, batch.size(1))
            context_words = batch.unsqueeze(1).expand(-1, batch.size(1), -1)

            target_words = target_words[mask.unsqueeze(2) & context_mask]
            context_words = context_words[mask.unsqueeze(1) & context_mask]

            self.co_occurrence_matrix.index_put_(
                (target_words, context_words),
                torch.tensor(1.0, dtype=torch.float32, device=self.device),
                accumulate=True
            )

    def compute_ppmi_matrix(self):
        epsilon = 1e-8  # petit nombre pour éviter les 0
        total_sum = self.co_occurrence_matrix.sum()
        row_sums = self.co_occurrence_matrix.sum(dim=1, keepdim=True)
        col_sums = self.co_occurrence_matrix.sum(dim=0, keepdim=True)

        num = self.co_occurrence_matrix * total_sum
        denom = row_sums * col_sums
        # Ajoute l'epsilon dans num et denom
        ppmi_matrix = torch.log((num + epsilon) / (denom + epsilon))
        ppmi_matrix[ppmi_matrix < 0] = 0
        # Nettoie ce qui reste au cas où (optionnel mais sûr) :
        ppmi_matrix[torch.isnan(ppmi_matrix)] = 0
        ppmi_matrix[torch.isinf(ppmi_matrix)] = 0
        return ppmi_matrix

    def compute_svd(self, ppmi_matrix):
        U, S, V = torch.svd(ppmi_matrix)
        U = U[:, :self.embedding_dim]
        S = S[:self.embedding_dim]
        word_embeddings = U @ torch.diag(S)
        return word_embeddings

    def fit(self, dataset):
        self.compute_co_occurrence_matrix(dataset)
        ppmi_matrix = self.compute_ppmi_matrix()
        word_embeddings = self.compute_svd(ppmi_matrix)
        return word_embeddings

# Example usage:
# dataset = [...]  # Your dataset that yields (batch_size, sequence_length) tensors
# vocab_size = ...  # The size of your vocabulary
# svd_ppmi = SVDPPMI(vocab_size)
# embeddings = svd_ppmi.fit(dataset)


def run_svd_ppmi(dataset=None, vocab_size=None):
        print(">>> Exécution de SVD_PPMI sur dataset réel")

        if dataset is None or vocab_size is None:
            print(">>> Dataset ou vocab_size non fourni")
            return

        model = SVDPPMI(vocab_size=vocab_size, window_size=2, embedding_dim=50, device='cpu')
        embeddings = model.fit(dataset)

        print(">>> Embeddings générés :")
        print(embeddings)