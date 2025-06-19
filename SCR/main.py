import torch
from SVD_PPMI import SVDPPMI
from SVD_PPMI_Nadi import SVD_PPMI_OPTIMA
from Gcount import GCount
from SVD_PPMI_Optim import SVD_PPMI_OPTI
import time
from Kernelized_Unit_Ball_Word_Embedding import *


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


def aggregate_with_embeddings(C0, Cdelta, phi):
    """
    Agrège les features C0 et Cδ par multiplication avec les embeddings phi.
    C0, Cdelta : tensors shape (n, vocab_size)
    phi : tensor shape (vocab_size, embedding_dim)
    Retour : tensor shape (n, 2 * embedding_dim)
    """
    agg_C0 = C0 @ phi          # (n, embedding_dim)
    agg_Cdelta = Cdelta @ phi  # (n, embedding_dim)
    return torch.cat([agg_C0, agg_Cdelta], dim=1)




"""def main():
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
    print(f"Durée d'exécution : {end - start:.4f} secondes")"""


def main():
    start = time.time()
    vocab_size = 4
    window_size = 1
    embedding_dim = 5
    batch_size = 3
    context_size = 5
    nb_batches = 1
    device = 'cuda'  # gpu  NVIDIA Tesla T4

    # batche
    dataset = Dummydataset(nb_batches, batch_size, context_size, vocab_size)
    start = time.time()
    # pipeline
    model = SVD_PPMI_OPTIMA(
        vocab_size=vocab_size,
        window_size=window_size,
        embedding_dim=embedding_dim,
        device=device
    )

    embeddings_svd_opti = model.fit(dataset)
    end = time.time()  # Arrête le chronomètre
    print(f"Durée d'exécution de embeddings_svd_opti : {end - start:.4f} secondes")
    start = time.time()

    print("-------------------------------------------------------------------------")
    print("Gcount delay")

    delta = 2 #par exemple
    gcount = GCount(vocab_size, delta, device)
    features = gcount.transform(dataset)
    print("Decayedcounting ", gcount.transform(dataset))
    C0 = features[:, :vocab_size]
    Cdelta = features[:, vocab_size:]
    features_gemb_local = aggregate_with_embeddings(C0, Cdelta, embeddings_svd_opti)
    print("Shape features_gemb_local:", features_gemb_local.shape)






    #print("Building cooccurrence matrix...")
    """cooc_matrix = build_cooc_matrix(dataset, vocab_size, window_size, pad_idx, device)

    # -- 2. α-PPMI --
    print("Computing α-PPMI (aPPMI) matrix...")
    alpha = 0.75
    appmi_matrix = appmi_sparse(cooc_matrix, alpha=alpha)



    #KUBWE
    # -- 3. KUBE vectorisé --
    print("Training KUBE embeddings...")
    embeddings_kube = kube_optimize_vec(
        appmi_matrix,
        embedding_dim=embedding_dim,
        kernel_degree=2,
        num_iter=50,
        lr=0.1,
        unit_ball=True,
        verbose=True,
        device=device
     )
    end = time.time()  # Arrête le chronomètre
    print(f"Durée d'exécution de embeddings_kube : {end - start:.4f} secondes")"""
    """
    model1 = SVDPPMI(
        vocab_size=vocab_size,
        window_size=window_size,
        embedding_dim=embedding_dim,
        device=device
    )
    embeddings_svdppmi = model1.fit(dataset)
    end = time.time() 
    print(f"Durée d'exécution de embeddings_svdppmi : {end - start:.4f} secondes")
    
    compare_embeddings_quality(embeddings_svd_opti, embeddings_svdppmi)
    name1 = "embeddings_svd_opti"
    name2 = "embeddings_kube"
    name3 = "embeddings_svdppmi"
    compare_embeddings_quality(embeddings_svd_opti, embeddings_kube, name1=name1, name2=name2)
    compare_embeddings_quality(embeddings_svdppmi, embeddings_kube, name1=name3, name2=name2)"""

    

if __name__ == "__main__":
    main()