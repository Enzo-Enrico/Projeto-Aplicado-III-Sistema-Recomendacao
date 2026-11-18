"""
Etapa 3 (continuação) - Extensão da grade do ALS na borda ótima e modelo final.
Mesmo protocolo de 03_ajuste_modelos.py. Salva ../docs/modelo_als.npz para o protótipo
e ../docs/exemplos_finais.json com exemplos qualitativos.
"""
import numpy as np, pandas as pd, scipy.sparse as sp, json, itertools, time
import implicit, threadpoolctl
threadpoolctl.threadpool_limits(1, "blas")
SEED, K, REL_THR = 42, 10, 4
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

r = pd.read_csv("../data/ratings.csv"); r["u"], r["i"] = r.user_id - 1, r.book_id - 1
nU, nI = r.u.max() + 1, r.i.max() + 1
r = r.sample(frac=1, random_state=SEED); r["rank"] = r.groupby("u").cumcount(); r["n"] = r.groupby("u")["u"].transform("size")
test = r[r["rank"] < np.floor(r.n * 0.2)]; rest = r[r["rank"] >= np.floor(r.n * 0.2)].copy()
rest["rank2"] = rest.groupby("u").cumcount(); rest["n2"] = rest.groupby("u")["u"].transform("size")
val = rest[rest.rank2 < np.floor(rest.n2 * 0.15)]; train = rest[rest.rank2 >= np.floor(rest.n2 * 0.15)]
B = lambda df: sp.csr_matrix((np.ones(len(df), np.float32), (df.u.values, df.i.values)), shape=(nU, nI))
B_tr, B_trval = B(train), B(pd.concat([train, val]))
rel = lambda df: df[df.rating >= REL_THR].groupby("u")["i"].apply(set).to_dict()
rel_val, rel_test = rel(val), rel(test)
item_pop = np.asarray(B_trval.sum(0)).ravel()

def evaluate(score_fn, R_seen, rel, k=K, batch=2000):
    users = np.array(sorted(rel.keys())); disc = 1 / np.log2(np.arange(2, k + 2)); P, Rc, N = [], [], []
    rec_counts = np.zeros(nI, np.int64)
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]; S = score_fn(ub).astype(np.float32); S[R_seen[ub].nonzero()] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]; top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1), 1)
        for row, u in enumerate(ub):
            rl = rel[u]; hits = np.fromiter((1.0 if it in rl else 0.0 for it in top[row]), float, k)
            P.append(hits.sum() / k); Rc.append(hits.sum() / len(rl)); N.append((hits * disc).sum() / disc[:min(len(rl), k)].sum())
        np.add.at(rec_counts, top.ravel(), 1)
    x = np.sort(rec_counts); n = len(x); w = rec_counts / rec_counts.sum()
    return {"precision@10": np.mean(P), "recall@10": np.mean(Rc), "ndcg@10": np.mean(N), "cobertura_catalogo": (rec_counts > 0).mean(),
            "popularidade_media_rec": float((w * item_pop).sum()), "gini_recomendacoes": float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))}

def als(Bm, f, reg, alpha, iters=15):
    m = implicit.als.AlternatingLeastSquares(factors=f, regularization=reg, iterations=iters, random_state=SEED, use_gpu=False)
    m.fit(Bm * alpha, show_progress=False); return np.asarray(m.user_factors), np.asarray(m.item_factors)

rows = []
for f, alpha in itertools.product([256, 384], [3, 5, 10]):
    U, V = als(B_tr, f, 0.01, alpha); m = evaluate(lambda ub: U[ub] @ V.T, B_tr, rel_val)
    rows.append({"modelo": "ALS", "params": f"fatores={f}, reg=0.01, alpha={alpha}", **m}); log(rows[-1]["params"], round(m["ndcg@10"], 4))
df = pd.DataFrame(rows); df.to_csv("../docs/ajuste_validacao_als_ext.csv", index=False)
best = df.sort_values("ndcg@10").iloc[-1]; log("melhor:", best.params)
f, alpha = int(best.params.split(",")[0].split("=")[1]), float(best.params.split("alpha=")[1])

# modelo final: treino+validação -> teste
U, V = als(B_trval, f, 0.01, alpha)
final = evaluate(lambda ub: U[ub] @ V.T, B_trval, rel_test); final["params"] = best.params
json.dump(final, open("../docs/resultado_final_als.json", "w"), indent=2, default=float); log("teste:", final)
sens = [{"k": kk, **{a: b for a, b in evaluate(lambda ub: U[ub] @ V.T, B_trval, rel_test, k=kk).items() if "@" in a}} for kk in [5, 10, 20, 50]]
pd.DataFrame(sens).to_csv("../docs/sensibilidade_k_als.csv", index=False)

# modelo de produção (todas as avaliações) para o protótipo
Uall, Vall = als(B(r), f, 0.01, alpha)
np.savez_compressed("../docs/modelo_als.npz", U=Uall.astype(np.float32), V=Vall.astype(np.float32), reg=0.01, alpha=alpha)

# exemplos qualitativos (3 usuários) com o modelo teste
books = pd.read_csv("../data/books.csv").set_index("book_id")["title"]
rng = np.random.default_rng(7); ex = []
for u in rng.choice(sorted(rel_test.keys()), 3, replace=False):
    u = int(u); seen = B_trval[u].nonzero()[1]; S = U[u] @ V.T; S[seen] = -np.inf
    top = np.argsort(-S)[:10]
    liked = pd.concat([train, val]); liked = liked[(liked.u == u)].sort_values("rating", ascending=False).head(5).i.values
    ex.append({"usuario": u + 1, "gostou": [books[i + 1] for i in liked], "relevantes_teste": [books[i + 1] for i in rel_test[u]][:8],
               "top10": [books[i + 1] for i in top], "acertos": [books[i + 1] for i in top if i in rel_test[u]]})
json.dump(ex, open("../docs/exemplos_finais.json", "w"), indent=2, ensure_ascii=False)
log("fim")
