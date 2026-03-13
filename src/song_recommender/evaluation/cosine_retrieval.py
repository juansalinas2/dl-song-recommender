import numpy as np

def topk_cosine(emb: np.ndarray, query: np.ndarray, k: int, exclude_index = None):
    if not np.isfinite(emb).all():
        raise ValueError('Embedding matrix contains NaN or inf values. Rebuild embeddings or restart the kernel.')
    if not np.isfinite(query).all():
        raise ValueError('Query vector contains NaN or inf values. Rebuild embeddings or restart the kernel.')
    # Use float64 einsum here to avoid spurious float32 matmul warnings from some NumPy/OpenBLAS builds.
    scores = np.einsum('ij,j->i', emb.astype(np.float64), query.astype(np.float64), optimize=True)

    if not np.isfinite(scores).all():
        raise ValueError('Cosine scores contain NaN or inf values after matmul.')

    # remove self match when checking queries on training set
    if exclude_index is not None:
        scores[exclude_index] = -np.inf   

    k = int(min(max(k, 1), len(scores)))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    return idx, scores[idx]