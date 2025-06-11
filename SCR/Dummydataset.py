import torch


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