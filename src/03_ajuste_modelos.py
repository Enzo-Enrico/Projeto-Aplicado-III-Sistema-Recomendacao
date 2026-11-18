"""
Etapa 3 - Ajuste do pipeline e reavaliação

Protocolo (por usuário, semente 42):
  - 20% das avaliações de cada usuário -> TESTE (idêntico à Etapa 2)
  - dos 80% restantes, 15% -> VALIDAÇÃO (usada só para escolher hiperparâmetros)
  - o restante -> TREINO
  Após a escolha, cada modelo é re-treinado em TREINO+VALIDAÇÃO e avaliado uma única vez no TESTE.

Modelos ajustados: Item-kNN (k, shrinkage), PureSVD (fatores), ALS (fatores, reg, alpha),
                   ALS + to_read (feedback implícito adicional).
Métricas: Precision@10, Recall@10, NDCG@10 (ranking exaustivo), cobertura de catálogo e
          popularidade média dos itens recomendados; RMSE/MAE só para o Item-kNN.
Saídas: ../docs/ajuste_validacao.csv, ../docs/resultados_finais.csv, ../docs/sensibilidade_k.csv
"""
import numpy as np, pandas as pd, scipy.sparse as sp, time, json, itertools
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import implicit, threadpoolctl
threadpoolctl.threadpool_limits(1, "blas")

SEED, K, REL_THR = 42, 10, 4
rng = np.random.default_rng(SEED)
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

# ---------------- dados e splits ----------------
r = pd.read_csv("../data/ratings.csv")
r["u"], r["i"] = r.user_id - 1, r.book_id - 1
nU, nI = r.u.max() + 1, r.i.max() + 1
r = r.sample(frac=1, random_state=SEED)
r["rank"] = r.groupby("u").cumcount()
r["n"] = r.groupby("u")["u"].transform("size")
test = r[r["rank"] < np.floor(r.n * 0.2)].copy()
rest = r[r["rank"] >= np.floor(r.n * 0.2)].copy()
rest["rank2"] = rest.groupby("u").cumcount()
rest["n2"] = rest.groupby("u")["u"].transform("size")
val = rest[rest.rank2 < np.floor(rest.n2 * 0.15)].copy()
train = rest[rest.rank2 >= np.floor(rest.n2 * 0.15)].copy()
log(f"treino {len(train):,} | validação {len(val):,} | teste {len(test):,}")

to_read = pd.read_csv("../data/to_read.csv")
to_read["u"], to_read["i"] = to_read.user_id - 1, to_read.book_id - 1
to_read = to_read[(to_read.u < nU) & (to_read.i < nI)]

def build(df):
    R = sp.csr_matrix((df.rating.values.astype(np.float32), (df.u.values, df.i.values)), shape=(nU, nI))
    return R
R_tr, R_trval = build(train), build(pd.concat([train, val]))
item_pop_full = np.asarray((R_trval > 0).sum(0)).ravel()

def rel_dict(df): return df[df.rating >= REL_THR].groupby("u")["i"].apply(set).to_dict()
rel_val, rel_test = rel_dict(val), rel_dict(test)

# ---------------- métricas ----------------
def evaluate(score_fn, R_seen, rel, k=K, batch=2000, extra=False):
    users = np.array(sorted(rel.keys()))
    disc = 1 / np.log2(np.arange(2, k + 2))
    P, Rc, N = [], [], []
    rec_counts = np.zeros(nI, np.int64)
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        S = score_fn(ub).astype(np.float32)
        S[R_seen[ub].nonzero()] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1), 1)
        for row, u in enumerate(ub):
            rl = rel[u]
            hits = np.fromiter((1.0 if it in rl else 0.0 for it in top[row]), float, k)
            P.append(hits.sum() / k); Rc.append(hits.sum() / len(rl))
            N.append((hits * disc).sum() / disc[:min(len(rl), k)].sum())
        if extra: np.add.at(rec_counts, top.ravel(), 1)
    out = {"precision@%d" % k: np.mean(P), "recall@%d" % k: np.mean(Rc), "ndcg@%d" % k: np.mean(N)}
    if extra:
        out["cobertura_catalogo"] = (rec_counts > 0).mean()
        w = rec_counts / rec_counts.sum()
        out["popularidade_media_rec"] = float((w * item_pop_full).sum())   # nº médio de avaliações dos itens recomendados
        # Gini da distribuição de recomendações entre itens (0 = uniforme)
        x = np.sort(rec_counts); n = len(x)
        out["gini_recomendacoes"] = float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
    return out

# ---------------- modelos ----------------
def item_knn(R, k_nn, shrink):
    item_mean = np.asarray(R.sum(0)).ravel() / np.maximum(np.asarray((R > 0).sum(0)).ravel(), 1)
    Rc = R.copy().astype(np.float32); Rc.data -= item_mean[Rc.indices]
    X = Rc.T.tocsr()                                  # itens x usuários
    num = (X @ X.T).toarray().astype(np.float32)
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    co = ((X != 0).astype(np.float32) @ (X != 0).astype(np.float32).T).toarray()   # nº de usuários em comum
    S = num / (np.outer(nrm, nrm) + 1e-9) * (co / (co + shrink))                  # cosseno com encolhimento
    np.fill_diagonal(S, 0); S[S < 0] = 0
    idx = np.argpartition(-S, k_nn, axis=1)[:, k_nn:]
    np.put_along_axis(S, idx, 0, axis=1)
    return Rc, S, item_mean

def pure_svd(R, f):
    B = (R > 0).astype(np.float32).tocsr()
    U, s, Vt = svds(B, k=f, random_state=SEED)
    return U * s, Vt

def als(R, f, reg, alpha, iters=15, extra=None):
    B = (R > 0).astype(np.float32).tocsr()
    if extra is not None: B = (B + extra).tocsr(); B.data[:] = 1
    m = implicit.als.AlternatingLeastSquares(factors=f, regularization=reg, iterations=iters, random_state=SEED, use_gpu=False)
    m.fit(B * alpha, show_progress=False)
    return np.asarray(m.user_factors), np.asarray(m.item_factors)

TR_bin = sp.csr_matrix((np.ones(len(to_read), np.float32), (to_read.u.values, to_read.i.values)), shape=(nU, nI))

# ---------------- 1. ajuste na validação ----------------
rows = []
log("== Item-kNN ==")
for k_nn, shrink in itertools.product([20, 50, 100, 200], [0, 10, 50]):
    Rc, S, _ = item_knn(R_tr, k_nn, shrink)
    m = evaluate(lambda ub: np.asarray(Rc[ub].dot(S)), R_tr, rel_val)
    rows.append({"modelo": "Item-kNN", "params": f"k={k_nn}, shrink={shrink}", **m}); log(rows[-1])
    del S

log("== PureSVD ==")
for f in [20, 50, 100, 200, 300]:
    U, Vt = pure_svd(R_tr, f)
    m = evaluate(lambda ub: U[ub] @ Vt, R_tr, rel_val)
    rows.append({"modelo": "PureSVD", "params": f"fatores={f}", **m}); log(rows[-1])

log("== ALS ==")
for f, reg, alpha in itertools.product([64, 128, 256], [0.01, 0.1], [10, 40]):
    U, V = als(R_tr, f, reg, alpha)
    m = evaluate(lambda ub: U[ub] @ V.T, R_tr, rel_val)
    rows.append({"modelo": "ALS", "params": f"fatores={f}, reg={reg}, alpha={alpha}", **m}); log(rows[-1])

val_df = pd.DataFrame(rows)
val_df.to_csv("../docs/ajuste_validacao.csv", index=False)
best = {m: val_df[val_df.modelo == m].sort_values("ndcg@10").iloc[-1] for m in ["Item-kNN", "PureSVD", "ALS"]}
for m, b in best.items(): log("melhor", m, b.params, round(b["ndcg@10"], 4))

# ALS + to_read: com os melhores hiperparâmetros do ALS
bp = dict(zip(["f", "reg", "alpha"], [float(x.split("=")[1]) for x in best["ALS"].params.split(", ")]))
U, V = als(R_tr, int(bp["f"]), bp["reg"], bp["alpha"], extra=TR_bin)
m = evaluate(lambda ub: U[ub] @ V.T, R_tr, rel_val)
rows.append({"modelo": "ALS + to_read", "params": best["ALS"].params, **m}); log(rows[-1])
val_df = pd.DataFrame(rows); val_df.to_csv("../docs/ajuste_validacao.csv", index=False)

# ---------------- 2. avaliação final no teste (treino+validação) ----------------
log("== Teste final ==")
final = {}
item_cnt = np.asarray((R_trval > 0).sum(0)).ravel().astype(np.float32)
final["Popularidade"] = evaluate(lambda ub: np.repeat(item_cnt[None, :], len(ub), 0), R_trval, rel_test, extra=True)

kk, sh = [int(x.split("=")[1]) for x in best["Item-kNN"].params.split(", ")]
Rc, S, item_mean = item_knn(R_trval, kk, sh)
final["Item-kNN"] = evaluate(lambda ub: np.asarray(Rc[ub].dot(S)), R_trval, rel_test, extra=True)
# RMSE/MAE do kNN
Rb = (R_trval > 0).astype(np.float32)
pred = np.empty(len(test), np.float32)
for s in range(0, len(test), 20000):
    tu, ti = test.u.values[s:s+20000], test.i.values[s:s+20000]
    Su = S[ti]
    numer = np.asarray(Rc[tu].multiply(Su).sum(1)).ravel(); denom = np.asarray(Rb[tu].multiply(Su).sum(1)).ravel()
    pred[s:s+20000] = item_mean[ti] + np.where(denom > 0, numer / np.maximum(denom, 1e-9), 0)
err = np.clip(pred, 1, 5) - test.rating.values
final["Item-kNN"].update({"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))})
del S

f = int(best["PureSVD"].params.split("=")[1])
U, Vt = pure_svd(R_trval, f)
final["PureSVD"] = evaluate(lambda ub: U[ub] @ Vt, R_trval, rel_test, extra=True)
# sensibilidade a k para o PureSVD
sens = []
for kk_ in [5, 10, 20, 50]:
    sens.append({"k": kk_, **evaluate(lambda ub: U[ub] @ Vt, R_trval, rel_test, k=kk_)})
pd.DataFrame(sens).to_csv("../docs/sensibilidade_k.csv", index=False)

U, V = als(R_trval, int(bp["f"]), bp["reg"], bp["alpha"])
final["ALS"] = evaluate(lambda ub: U[ub] @ V.T, R_trval, rel_test, extra=True)
U2, V2 = als(R_trval, int(bp["f"]), bp["reg"], bp["alpha"], extra=TR_bin)
final["ALS + to_read"] = evaluate(lambda ub: U2[ub] @ V2.T, R_trval, rel_test, extra=True)

fdf = pd.DataFrame(final).T
fdf.insert(0, "params", ["-", best["Item-kNN"].params, best["PureSVD"].params, best["ALS"].params, best["ALS"].params])
fdf.to_csv("../docs/resultados_finais.csv")
print("\n", fdf.round(4).to_string())

# ---------------- 3. salvar modelo final (PureSVD) para o protótipo ----------------
np.savez_compressed("../docs/modelo_puresvd.npz", U=U.astype(np.float32), Vt=Vt.astype(np.float32), f=f)
json.dump({"item_knn": best["Item-kNN"].params, "puresvd": best["PureSVD"].params, "als": best["ALS"].params},
          open("../docs/melhores_hiperparametros.json", "w"), indent=2)
log("fim")
