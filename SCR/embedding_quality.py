import torch
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def embedding_quality_report(embeddings, method_name="Méthode"):
    with torch.no_grad():
        # 1. Stats similarité cosinus sur paires aléatoires
        vocab_size = embeddings.size(0)
        nb_samples = min(1000, vocab_size**2)
        idx1 = torch.randint(0, vocab_size, (nb_samples,))
        idx2 = torch.randint(0, vocab_size, (nb_samples,))
        sims = cosine_similarity(embeddings[idx1], embeddings[idx2], dim=1)
        sim_mean, sim_std = sims.mean().item(), sims.std().item()

        # 2. Normes
        norms = torch.norm(embeddings, dim=1)
        norm_min, norm_max, norm_mean = norms.min().item(), norms.max().item(), norms.mean().item()

        # 3. Clustering KMeans & Silhouette
        n_clusters = min(10, vocab_size) if vocab_size > 10 else vocab_size
        emb_np = embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=3)
        labels = kmeans.fit_predict(emb_np)
        sil_score = silhouette_score(emb_np, labels)

        # 4. Affichage PCA
        pca = PCA(n_components=2)
        emb2d = pca.fit_transform(emb_np)
        plt.figure(figsize=(5,4))
        plt.scatter(emb2d[:,0], emb2d[:,1], c=labels, s=7, cmap='tab10', alpha=0.7)
        plt.title(f"Projection PCA des embeddings ({method_name})")
        plt.tight_layout()
        plt.show()

        print(f"\n===== Rapport qualité pour {method_name} =====")
        print(f"Similarité cosinus aléatoire: moyenne={sim_mean:.4f}, std={sim_std:.4f}")
        print(f"Normes des vecteurs: min={norm_min:.4f}, max={norm_max:.4f}, moyenne={norm_mean:.4f}")
        print(f"Silhouette score clustering (k={n_clusters}): {sil_score:.4f}")


def compare_embeddings_quality(embeddings1, embeddings2, name1="SVD_PPMI_OPTI", name2="SVDPPMI"):
    print("==== COMPARAISON QUALITÉ DES EMBEDDINGS ====")
    embedding_quality_report(embeddings1, method_name=name1)
    embedding_quality_report(embeddings2, method_name=name2)
    print("==== FIN DE LA COMPARAISON ====")

# Exemple d’utilisation :
# compare_embeddings_quality(embeddings_svd_opti, embeddings_svdppmi)
