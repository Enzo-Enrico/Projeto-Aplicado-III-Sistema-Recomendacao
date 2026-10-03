"""Baixa o goodbooks-10k (Zajac, 2017) para a pasta data/."""
import urllib.request, os
BASE = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
here = os.path.dirname(os.path.abspath(__file__))
for f in ["ratings", "books", "to_read", "tags", "book_tags"]:
    dst = os.path.join(here, f"{f}.csv")
    if not os.path.exists(dst):
        print("baixando", f); urllib.request.urlretrieve(BASE + f + ".csv", dst)
print("ok")
