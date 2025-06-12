import torch
from SVD_PPMI_Optim import SVD_PPMI_OPTI
from SVD_PPMI import run_svd_ppmi
import time


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


def main():
    start = time.time()
    #Paramètres
    batch_size = 512
    context_size = 94
    vocab_size = 733

    #batch = torch.randint(0, vocab_size, (batch_size, context_size))
    nb_batches = 10000
    dataset = Dummydataset(nb_batches, batch_size, context_size, vocab_size)

    run_svd_ppmi(dataset, vocab_size=vocab_size)
    end = time.time()  # Arrête le chronomètre
    print(f"Durée d'exécution : {end - start:.4f} secondes")


"""def main():
    start = time.time()
    vocab_size = 733
    window_size = 2
    embedding_dim = 100
    pad_idx = 0
    batch_size = 512
    context_size = 94
    nb_batches = 10000
    device = 'cuda'  # gpu  NVIDIA Tesla T4

    # batche
    batches = [torch.randint(0, vocab_size, (batch_size, context_size)) for _ in range(nb_batches)]
    dataset = Dummydataset(nb_batches, batch_size, context_size, vocab_size)
    # pipeline
    model = SVD_PPMI_OPTI(
        vocab_size=vocab_size,
        window_size=window_size,
        embedding_dim=embedding_dim,
        pad_idx=pad_idx,
        device=device
    )

    embeddings = model.fit(dataset)
    print(embeddings)
    end = time.time()  # Arrête le chronomètre
    print(f"Durée d'exécution : {end - start:.4f} secondes")"""


if __name__ == "__main__":
    main()