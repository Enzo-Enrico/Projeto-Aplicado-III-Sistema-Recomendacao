"""
Etapa 2 - Análise exploratória do goodbooks-10k
Gera estatísticas (eda_stats.txt) e figuras em figures/
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, os, json

os.makedirs("../figures", exist_ok=True)
r = pd.read_csv("../data/ratings.csv")
b = pd.read_csv("../data/books.csv")
tr = pd.read_csv("../data/to_read.csv")
tags = pd.read_csv("../data/tags.csv")
bt = pd.read_csv("../data/book_tags.csv")

out = {}
out["n_ratings"] = len(r)
out["n_users"] = r.user_id.nunique()
out["n_books"] = r.book_id.nunique()
out["duplicates_user_book"] = int(r.duplicated(["user_id", "book_id"]).sum())
out["missing_ratings"] = int(r.isna().sum().sum())
out["sparsity_pct"] = round(100 * (1 - len(r) / (out["n_users"] * out["n_books"])), 3)
out["rating_mean"] = round(r.rating.mean(), 3)
out["rating_dist_pct"] = (r.rating.value_counts(normalize=True).sort_index() * 100).round(2).to_dict()

upu = r.groupby("user_id").size()
rpb = r.groupby("book_id").size()
out["ratings_per_user"] = {"min": int(upu.min()), "median": float(upu.median()), "mean": round(upu.mean(), 1), "max": int(upu.max()),
                           "p25": float(upu.quantile(.25)), "p75": float(upu.quantile(.75))}
out["ratings_per_book"] = {"min": int(rpb.min()), "median": float(rpb.median()), "mean": round(rpb.mean(), 1), "max": int(rpb.max())}
top20 = rpb.sort_values(ascending=False).head(int(0.2 * len(rpb)))
out["share_ratings_top20pct_books"] = round(100 * top20.sum() / len(r), 1)

out["to_read_rows"] = len(tr)
out["books_missing"] = b.isna().sum()[b.isna().sum() > 0].to_dict()
out["lang_top"] = b.language_code.value_counts().head(6).to_dict()
out["pub_year_median"] = float(b.original_publication_year.median())
out["n_tags"] = len(tags)
out["book_tags_rows"] = len(bt)

# livros mais avaliados
top_books = rpb.sort_values(ascending=False).head(10).reset_index()
top_books.columns = ["book_id", "n"]
top_books = top_books.merge(b[["book_id", "title", "authors", "average_rating"]], on="book_id")
out["top10_books"] = top_books[["title", "authors", "n", "average_rating"]].to_dict("records")

# tags mais usadas (depois de filtro grosseiro)
tt = bt.groupby("tag_id")["count"].sum().reset_index().merge(tags, on="tag_id").sort_values("count", ascending=False)
out["top_tags"] = tt.head(15)[["tag_name", "count"]].to_dict("records")

with open("../docs/eda_stats.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(out, indent=2, ensure_ascii=False, default=str))

# ---------- figuras ----------
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
C = "#1f4e79"

fig, ax = plt.subplots(figsize=(6, 3.4))
vc = r.rating.value_counts().sort_index()
ax.bar(vc.index, vc.values / 1e6, color=C)
ax.set_xlabel("Nota (1–5)"); ax.set_ylabel("Avaliações (milhões)")
ax.set_title("Figura 1 – Distribuição das notas")
for i, v in zip(vc.index, vc.values): ax.text(i, v / 1e6 + 0.03, f"{100*v/len(r):.1f}%", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("../figures/fig1_dist_notas.png"); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
axes[0].hist(upu, bins=60, color=C); axes[0].set_yscale("log")
axes[0].set_xlabel("Avaliações por usuário"); axes[0].set_ylabel("Usuários (log)"); axes[0].set_title("(a) Usuários")
axes[1].hist(rpb, bins=60, color=C); axes[1].set_yscale("log")
axes[1].set_xlabel("Avaliações por livro"); axes[1].set_ylabel("Livros (log)"); axes[1].set_title("(b) Livros")
fig.suptitle("Figura 2 – Distribuição de avaliações por usuário e por livro")
plt.tight_layout(); plt.savefig("../figures/fig2_long_tail.png"); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.4))
s = rpb.sort_values(ascending=False).cumsum() / len(r)
ax.plot(np.arange(1, len(s) + 1) / len(s) * 100, s.values * 100, color=C)
ax.axvline(20, ls="--", c="gray"); ax.axhline(out["share_ratings_top20pct_books"], ls="--", c="gray")
ax.set_xlabel("% dos livros (ordenados por popularidade)"); ax.set_ylabel("% acumulado das avaliações")
ax.set_title("Figura 3 – Concentração de avaliações (cauda longa)")
plt.tight_layout(); plt.savefig("../figures/fig3_cauda_longa.png"); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.4))
b.original_publication_year.dropna().clip(1900, 2017).hist(bins=50, color=C, ax=ax)
ax.set_xlabel("Ano de publicação original (truncado em 1900)"); ax.set_ylabel("Livros")
ax.set_title("Figura 4 – Ano de publicação dos livros")
plt.tight_layout(); plt.savefig("../figures/fig4_ano_pub.png"); plt.close()
print("figuras ok")
