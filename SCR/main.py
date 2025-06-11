import torch
from SVD_PPMI import run_svd_ppmi


def main():
    #Paramètres
    batch_size = 512
    context_size = 94
    vocab_size = 733

    #batch = torch.randint(0, vocab_size, (batch_size, context_size))
    nb_batches = 1000
    dataset = [
      torch.randint(0, vocab_size, (batch_size, context_size))
      for _ in range(nb_batches)]

    run_svd_ppmi(dataset, vocab_size=vocab_size)


if __name__ == "__main__":
    main()