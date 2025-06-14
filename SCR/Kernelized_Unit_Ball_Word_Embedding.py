import torch
from tqdm import trange

def normalize_rows(Y):
    return Y / (Y.norm(dim=1, keepdim=True) + 1e-8)

def polynomial_kernel_mat(Y, degree=3):
    # (Y @ Y^T + 1) / 2, puis puissance degree
    S = (Y @ Y.T + 1.0) / 2.0
    return S.pow(degree)

def kube_optimize_vec(
    cooc_matrix,        # torch.sparse_coo_tensor [N, N]
    embedding_dim=100,
    kernel_degree=2,
    unit_ball=True,
    num_iter=50,
    lr=0.1,
    verbose=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    N = cooc_matrix.size(0)
    Y = torch.randn(N, embedding_dim, device=device)
    Y = normalize_rows(Y)
    cooc_matrix = cooc_matrix.coalesce()
    indices = cooc_matrix.indices()
    values = cooc_matrix.values()

    for it in trange(num_iter, desc="KUBE vectorized"):
        Y_old = Y.clone()

        # 1. Attractive force (matmul sparse)
        #   Pour chaque mot i, att[i] = somme_j cooc[i, j] * Y[j]
        att = torch.zeros_like(Y)
        # indices: [2, nnz], values: [nnz]
        att.index_add_(
            0,
            indices[0],
            values.unsqueeze(1) * Y[indices[1]]
        )
        att = normalize_rows(att)

        # 2. Repulsive force (full kernel sim, sans diag/voisins directs)
        S = polynomial_kernel_mat(Y, degree=kernel_degree)   # [N, N]
        # On masque : S[i,i] = 0 et S[i,voisins]=0
        S[torch.arange(N), torch.arange(N)] = 0.0
        S[indices[0], indices[1]] = 0.0

        rep = S @ Y
        rep = normalize_rows(rep)

        # 3. Gradient & Update
        grad = att - rep
        grad = normalize_rows(grad)
        Y = Y + lr * grad
        if unit_ball:
            Y = normalize_rows(Y)

        if verbose and (it % 10 == 0 or it == num_iter-1):
            avg_change = (Y - Y_old).norm().item() / N
            print(f"Iter {it+1}/{num_iter}, avg_update_norm={avg_change:.6f}")

    return Y


def appmi_sparse(cooc_matrix, alpha=0.75, eps=1e-8):
    # cooc_matrix : torch.sparse_coo_tensor [N, N]
    cooc_matrix = cooc_matrix.coalesce()
    rows, cols = cooc_matrix.indices()
    vals = cooc_matrix.values()

    total_sum = vals.sum()
    row_sums = torch.zeros(cooc_matrix.size(0), device=vals.device).index_add_(0, rows, vals)
    col_sums = torch.zeros(cooc_matrix.size(1), device=vals.device).index_add_(0, cols, vals)
    # alpha-PPMI pour chaque entry
    appmi_vals = torch.log((vals * total_sum + eps) / (row_sums[rows] * col_sums[cols].pow(alpha) + eps))
    appmi_vals = torch.where(appmi_vals > 0, appmi_vals, torch.zeros_like(appmi_vals))
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), appmi_vals, cooc_matrix.shape
    ).coalesce()


def build_cooc_matrix(dataset, vocab_size, window_size=2, pad_idx=0, device='cuda'):
    cooc_counts = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=device)
    offsets = [i for i in range(-window_size, window_size+1) if i != 0]
    for batch in dataset:
        batch = batch.to(device)
        mask = (batch != pad_idx)
        for offset in offsets:
            context = torch.roll(batch, shifts=offset, dims=1)
            context_mask = torch.roll(mask, shifts=offset, dims=1)
            valid = mask & context_mask
            targets = batch[valid]
            contexts = context[valid]
            cooc_counts.index_put_(
                (targets, contexts),
                torch.ones_like(targets, dtype=torch.float32, device=device),
                accumulate=True
            )
    rows, cols = cooc_counts.nonzero(as_tuple=True)
    values = cooc_counts[rows, cols]
    cooc_matrix = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), values, (vocab_size, vocab_size)
    ).coalesce()
    return cooc_matrix

# -- UTILISATION EXEMPLE --
# cooc_matrix : torch.sparse_coo_tensor [N, N], comme ci-dessus

# embeddings = kube_optimize_vec(cooc_matrix, embedding_dim=100, kernel_degree=2, num_iter=50, lr=0.1)
