"""
Etapa 2 - Prova de conceito: treinamento e avaliação de modelos iniciais

Protocolo:
  - holdout por usuário: 20% das avaliações de cada usuário vão para teste (seed 42)
  - Avaliação 1 (predição de nota, só itens do teste): RMSE e MAE
  - Avaliação 2 (qualidade da lista): Precision@10, Recall@10, NDCG@10
      relevante = item do teste com nota >= 4; ranking completo sobre todos os
      itens não vistos no treino (sem amostragem de negativos).
Modelos: popularidade, item-kNN (cosseno), SVD (Funk/Surprise), ALS implícito.
"""
import numpy as np, pandas as pd, scipy.sparse as sp, time, json, os
from sklearn.preprocessing import normalize

SEED, K, TEST_FRAC, REL_THR = 42, 10, 0.2, 4
rng = np.random.default_rng(SEED)
t0 = time.time()

# ---------------- dados e split ----------------
r = pd.read_csv("../data/ratings.csv")
r["u"] = r.user_id - 1            # ids já são contíguos a partir de 1
r["i"] = r.book_id - 1
nU, nI = r.u.max() + 1, r.i.max() + 1

r = r.sample(frac=1, random_state=SEED)
r["rank"] = r.groupby("u").cumcount()
r["n_user"] = r.groupby("u")["u"].transform("size")
test_mask = r["rank"] < np.floor(r.n_user * TEST_FRAC)
train, test = r[~test_mask].copy(), r[test_mask].copy()
print(f"treino {len(train):,} | teste {len(test):,} | usuários {nU:,} | itens {nI:,}")

R = sp.csr_matrix((train.rating.values.astype(np.float32), (train.u.values, train.i.values)), shape=(nU, nI))
mu = train.rating.mean()
item_mean = np.asarray(R.sum(0)).ravel() / np.maximum(np.asarray((R > 0).sum(0)).ravel(), 1)
item_cnt = np.asarray((R > 0).sum(0)).ravel()

# relevantes no teste (dict usuário -> conjunto de itens com nota >= 4)
test_rel = test[test.rating >= REL_THR].groupby("u")["i"].apply(set).to_dict()

# ---------------- métricas ----------------
def rating_metrics(pred):
    err = pred - test.rating.values
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))

def ranking_metrics(score_fn, name, batch=2000):
    """score_fn(user_idx_array) -> matriz (len, nI) de scores. Itens de treino são mascarados."""
    users = np.array(sorted(test_rel.keys()))
    P, Rc, N = [], [], []
    disc = 1 / np.log2(np.arange(2, K + 2))
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        S = score_fn(ub).astype(np.float32)
        S[R[ub].nonzero()] = -np.inf          # nunca recomendar o já lido
        top = np.argpartition(-S, K, axis=1)[:, :K]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1), 1)
        for row, u in enumerate(ub):
            rel = test_rel[u]
            hits = np.array([1.0 if it in rel else 0.0 for it in top[row]])
            P.append(hits.sum() / K)
            Rc.append(hits.sum() / len(rel))
            dcg = (hits * disc).sum()
            idcg = disc[:min(len(rel), K)].sum()
            N.append(dcg / idcg)
    res = {"precision@10": float(np.mean(P)), "recall@10": float(np.mean(Rc)), "ndcg@10": float(np.mean(N))}
    print(f"  {name}: {res}")
    return res

results = {}

# ---------------- 1. Baselines ----------------
print("\n[1] Baselines")
rm = rating_metrics(np.full(len(test), mu));          results["Média global"] = {"rmse": rm[0], "mae": rm[1]}
rm = rating_metrics(item_mean[test.i.values]);         results["Média do item"] = {"rmse": rm[0], "mae": rm[1]}
pop_scores = np.tile(item_cnt.astype(np.float32), (1, 1))
results["Popularidade (nº de avaliações)"] = ranking_metrics(lambda ub: np.repeat(pop_scores, len(ub), 0), "popularidade")
results["Popularidade (nº de avaliações)"].update({"rmse": results["Média do item"]["rmse"], "mae": results["Média do item"]["mae"]})

# ---------------- 2. Item-kNN (cosseno sobre notas centradas no item) ----------------
print("\n[2] Item-kNN")
t = time.time()
Rc_ = R.copy().astype(np.float32)
Rc_.data = Rc_.data - item_mean[Rc_.indices]               # centraliza por item (R é csr -> indices = itens)
Xn = normalize(Rc_.T.tocsr(), axis=1)                      # itens x usuários, normalizados
S_full = (Xn @ Xn.T).toarray().astype(np.float32)          # similaridade item-item 10k x 10k
np.fill_diagonal(S_full, 0)
KN = 50
idx = np.argpartition(-S_full, KN, axis=1)[:, KN:]         # zera tudo fora dos 50 vizinhos
S_knn = S_full.copy(); np.put_along_axis(S_knn, idx, 0, axis=1)
S_knn[S_knn < 0] = 0
del S_full, idx

# predição de nota: média do item + média ponderada dos desvios dos vizinhos avaliados pelo usuário
Ruc = R.copy().astype(np.float32); Ruc.data = Ruc.data - item_mean[Ruc.indices]
pred = np.empty(len(test), dtype=np.float32)
Rb = (R > 0).astype(np.float32)
for s in range(0, len(test), 20000):
    tu, ti = test.u.values[s:s+20000], test.i.values[s:s+20000]
    Su = S_knn[ti]                                         # (b, nI) similaridades do item alvo com todos
    numer = np.asarray(Ruc[tu].multiply(Su).sum(1)).ravel()
    denom = np.asarray(Rb[tu].multiply(Su).sum(1)).ravel()
    pred[s:s+20000] = item_mean[ti] + np.where(denom > 0, numer / np.maximum(denom, 1e-9), 0)
pred = np.clip(pred, 1, 5)
rm = rating_metrics(pred)
knn_rank = ranking_metrics(lambda ub: np.asarray(Rc_[ub].dot(S_knn)) , "item-kNN")
results["Item-kNN (cosseno, k=50)"] = {"rmse": rm[0], "mae": rm[1], **knn_rank}
print(f"  tempo item-kNN: {time.time()-t:.0f}s")
del S_knn

# ---------------- 3. SVD (Funk-SVD, Surprise) ----------------
print("\n[3] SVD / Surprise")
from surprise import SVD, Dataset, Reader
t = time.time()
ds = Dataset.load_from_df(train[["u", "i", "rating"]], Reader(rating_scale=(1, 5))).build_full_trainset()
svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=SEED)
svd.fit(ds)
# mapeia ids internos -> matrizes alinhadas aos índices externos
P = np.zeros((nU, 50), np.float32); Q = np.zeros((nI, 50), np.float32)
bu = np.zeros(nU, np.float32); bi = np.zeros(nI, np.float32)
for inner in range(ds.n_users): P[ds.to_raw_uid(inner)] = svd.pu[inner]; bu[ds.to_raw_uid(inner)] = svd.bu[inner]
for inner in range(ds.n_items): Q[ds.to_raw_iid(inner)] = svd.qi[inner]; bi[ds.to_raw_iid(inner)] = svd.bi[inner]
pred = np.clip(svd.trainset.global_mean + bu[test.u.values] + bi[test.i.values] + (P[test.u.values] * Q[test.i.values]).sum(1), 1, 5)
rm = rating_metrics(pred)
svd_rank = ranking_metrics(lambda ub: svd.trainset.global_mean + bu[ub][:, None] + bi[None, :] + P[ub] @ Q.T, "SVD")
results["SVD (Funk, 50 fatores)"] = {"rmse": rm[0], "mae": rm[1], **svd_rank}
print(f"  tempo SVD: {time.time()-t:.0f}s")


# ---------------- 3b. PureSVD (Cremonesi et al., 2010) ----------------
print("\n[3b] PureSVD")
from scipy.sparse.linalg import svds
t = time.time()
Uu, sg, Vt = svds(Rimp_bin := (R > 0).astype(np.float32).tocsr(), k=50, random_state=SEED)
Uu = Uu * sg
pure_rank = ranking_metrics(lambda ub: Uu[ub] @ Vt, "PureSVD")
results["PureSVD (50 fatores, binário)"] = {"rmse": None, "mae": None, **pure_rank}
print(f"  tempo PureSVD: {time.time()-t:.0f}s")

# ---------------- 4. ALS implícito (implicit) ----------------
print("\n[4] ALS implícito")
import implicit, threadpoolctl
threadpoolctl.threadpool_limits(1, "blas")
t = time.time()
Rimp = (R > 0).astype(np.float32).tocsr()                  # feedback implícito: avaliou = interagiu
als = implicit.als.AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, random_state=SEED, use_gpu=False)
als.fit(Rimp * 20)                                          # alpha = 20 (confiança)
U, V = np.asarray(als.user_factors), np.asarray(als.item_factors)
als_rank = ranking_metrics(lambda ub: U[ub] @ V.T, "ALS")
results["ALS implícito (64 fatores)"] = {"rmse": None, "mae": None, **als_rank}
print(f"  tempo ALS: {time.time()-t:.0f}s")

json.dump(results, open("../docs/poc_results.json", "w"), indent=2, ensure_ascii=False)
df = pd.DataFrame(results).T[["rmse", "mae", "precision@10", "recall@10", "ndcg@10"]]
print("\n", df.round(4).to_string())
df.to_csv("../docs/poc_results.csv")

# ---------------- exemplo qualitativo ----------------
b = pd.read_csv("../data/books.csv")[["book_id", "title", "authors"]]
u = int(rng.choice(list(test_rel.keys())))
lidos = train[train.u == u].sort_values("rating", ascending=False).head(5).i.values + 1
S = svd.trainset.global_mean + bu[u] + bi + P[u] @ Q.T; S[R[u].nonzero()[1]] = -np.inf
rec = np.argsort(-S)[:10] + 1
S2 = U[u] @ V.T; S2[R[u].nonzero()[1]] = -np.inf
rec2 = np.argsort(-S2)[:10] + 1
rel_titles = b.set_index("book_id").loc[[i + 1 for i in test_rel[u]], "title"].tolist()
ex = {"usuario": u + 1,
      "mais_bem_avaliados_no_treino": b.set_index("book_id").loc[lidos, "title"].tolist(),
      "relevantes_no_teste": rel_titles,
      "top10_svd_funk": b.set_index("book_id").loc[rec, "title"].tolist(),
      "top10_als": b.set_index("book_id").loc[rec2, "title"].tolist()}
json.dump(ex, open("../docs/exemplo_recomendacao.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(ex, indent=2, ensure_ascii=False))
print(f"\ntempo total {time.time()-t0:.0f}s")
