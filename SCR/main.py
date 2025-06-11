import torch
from SVD_PPMI import run_svd_ppmi


def main():

    print("Hello la team")

    batch_size = 512
    context_size = 94
    vocab_size = 733

    # Exemple de dataset avec des indices aléatoires entre 1 et 732
    batch = torch.randint(1, vocab_size, (batch_size, context_size))

    dataset = [batch]  # Envelopper dans une liste comme batch unique (ou plusieurs)

    run_svd_ppmi(dataset, vocab_size=vocab_size)


if __name__ == "__main__":
    main()