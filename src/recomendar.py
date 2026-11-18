"""
Protótipo de recomendação de livros - Projeto Aplicado III (Mackenzie)
Modelo: ALS para feedback implícito (Hu, Koren e Volinsky, 2008), 256 fatores, reg 0,01, alpha 5,
treinado sobre as 5,98 milhões de avaliações do goodbooks-10k.

Uso:
  python recomendar.py --usuario 4765                          # top-N para um usuário do dataset
  python recomendar.py --livros "Harry Potter" "Hobbit" "Dune" # top-N para um leitor NOVO, a partir do que já leu
  python recomendar.py --treinar                               # re-treina e salva ../docs/modelo_als.npz
Opções: --n 10 (tamanho da lista), --csv saida.csv (salva a lista)

Leitor novo (caso de uma biblioteca): o vetor do leitor é obtido por "folding-in", resolvendo o passo
de usuário do ALS com os fatores dos itens fixos, a partir dos livros informados. Não é preciso re-treinar.
"""
import argparse, os, sys, numpy as np, pandas as pd, scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, DOCS = os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "docs")
MODEL = os.path.join(DOCS, "modelo_als.npz")
F, REG, ALPHA = 256, 0.01, 5.0

def treinar():
    import implicit
    r = pd.read_csv(os.path.join(DATA, "ratings.csv"))
    B = sp.csr_matrix((np.ones(len(r), np.float32), (r.user_id.values - 1, r.book_id.values - 1)))
    m = implicit.als.AlternatingLeastSquares(factors=F, regularization=REG, iterations=15, random_state=42, use_gpu=False)
    m.fit(B * ALPHA)
    np.savez_compressed(MODEL, U=np.asarray(m.user_factors, np.float32), V=np.asarray(m.item_factors, np.float32), reg=REG, alpha=ALPHA)

def carregar():
    if not os.path.exists(MODEL):
        print("Modelo não encontrado; treinando (~2 min)..."); treinar()
    m = np.load(MODEL)
    books = pd.read_csv(os.path.join(DATA, "books.csv"))[["book_id", "title", "authors", "original_publication_year", "average_rating"]]
    return m["U"], m["V"], float(m["reg"]), float(m["alpha"]), books

def folding_in(V, ids, reg, alpha):
    """Resolve  u = (Vᵀ C V + reg·I)⁻¹ Vᵀ C p  para um leitor novo com itens `ids` (Hu et al., 2008)."""
    Vi = V[ids]; c = 1 + alpha                      # confiança dos itens observados
    A = V.T @ V + (c - 1) * (Vi.T @ Vi) + reg * np.eye(V.shape[1], dtype=np.float32)
    b = c * Vi.sum(0)
    return np.linalg.solve(A, b)

def top_n(scores, excluir, n, books):
    scores = scores.copy(); scores[list(excluir)] = -np.inf
    idx = np.argsort(-scores)[:n]
    out = books.set_index("book_id").loc[idx + 1].reset_index()
    out.insert(0, "posicao", range(1, n + 1)); out["score"] = scores[idx].round(3)
    return out[["posicao", "title", "authors", "original_publication_year", "average_rating", "score"]]

def buscar_livros(titulos, books):
    ids = []
    for t in titulos:
        hit = books[books.title.str.contains(t, case=False, regex=False)]
        if hit.empty: print(f"  aviso: '{t}' não encontrado no catálogo"); continue
        row = hit.sort_values("average_rating", ascending=False).iloc[0]
        print(f"  '{t}' -> {row.title}"); ids.append(int(row.book_id) - 1)
    return ids

def main():
    ap = argparse.ArgumentParser(description="Recomendador de livros (ALS) - Projeto Aplicado III")
    ap.add_argument("--usuario", type=int); ap.add_argument("--livros", nargs="+"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--treinar", action="store_true"); ap.add_argument("--csv")
    a = ap.parse_args()
    if a.treinar: treinar(); print("modelo salvo em", MODEL); return
    U, V, reg, alpha, books = carregar()
    pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 55)
    if a.usuario:
        r = pd.read_csv(os.path.join(DATA, "ratings.csv")); lidos = r[r.user_id == a.usuario]
        if lidos.empty: sys.exit("usuário não encontrado")
        print(f"\nUsuário {a.usuario}: {len(lidos)} livros avaliados. Mais bem avaliados:")
        print(books.set_index("book_id").loc[lidos.sort_values("rating", ascending=False).book_id.head(5), "title"].to_string(header=False))
        rec = top_n(U[a.usuario - 1] @ V.T, lidos.book_id.values - 1, a.n, books)
    elif a.livros:
        print("Livros informados:"); ids = buscar_livros(a.livros, books)
        if not ids: sys.exit("nenhum livro reconhecido")
        rec = top_n(folding_in(V, ids, reg, alpha) @ V.T, ids, a.n, books)
    else:
        ap.print_help(); return
    print(f"\nTop-{a.n} recomendações:\n" + rec.to_string(index=False))
    if a.csv: rec.to_csv(a.csv, index=False); print("salvo em", a.csv)

if __name__ == "__main__":
    main()
