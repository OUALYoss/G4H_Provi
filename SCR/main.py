import torch
from SVD_PPMI_Optim import SVD_PPMI_OPTI


class Dummydataset:
    def __init__(self, nb_batches, batch_size, context_size, vocab_size):
        self.nb_batches = nb_batches
        self.batch_size = batch_size
        self.context_size = context_size
        self.vocab_size = vocab_size

    def __len__(self):
        return self.nb_batches

    def __iter__(self):
        for _ in range(self.nb_batches):
            yield torch.randint(0, self.vocab_size, (self.batch_size, self.context_size))



"""def main():
    #Paramètres
    batch_size = 512
    context_size = 94
    vocab_size = 733

    #batch = torch.randint(0, vocab_size, (batch_size, context_size))
    nb_batches = 10000
    dataset = Dummydataset(nb_batches, batch_size, context_size, vocab_size)

    run_svd_ppmi(dataset, vocab_size=vocab_size)"""


def main():
    vocab_size = 733
    window_size = 2
    embedding_dim = 100
    pad_idx = 0
    batch_size = 512
    context_size = 94
    nb_batches = 10000
    device = 'cuda'  # ou 'cpu' selon ta machine

    # Ex : création de 10 000 batches aléatoires pour le test
    # Remplace cette ligne par tes vrais batches si tu les as déjà
    batches = [torch.randint(0, vocab_size, (batch_size, context_size)) for _ in range(nb_batches)]
    # Pour un vrai pipeline, tu peux faire: batches = ton_iterateur_ou_liste

    model = SVD_PPMI_OPTI(
        vocab_size=vocab_size,
        window_size=window_size,
        embedding_dim=embedding_dim,
        pad_idx=pad_idx,
        device=device
    )

    embeddings = model.fit(batches)
    print(embeddings)

if __name__ == "__main__":
    main()