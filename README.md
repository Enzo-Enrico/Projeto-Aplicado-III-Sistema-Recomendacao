# Projeto Aplicado III – Sistema de Recomendação de Livros

**Universidade Presbiteriana Mackenzie – Curso de Ciência de Dados**
**Aluno:** Enzo Enrico Braga Cavalcante (RA 10744145)
**Local/Ano:** Guarulhos – 2026

## Sobre o projeto

Desenvolvimento e avaliação de um **sistema de recomendação de livros baseado em filtragem colaborativa**, capaz de gerar sugestões personalizadas a partir do histórico de avaliações dos usuários. O projeto tem caráter extensionista: o código será disponibilizado de forma aberta e documentada para que possa ser adaptado por bibliotecas públicas, escolares e comunitárias como ferramenta de apoio ao incentivo à leitura.

O trabalho dá continuidade ao [Projeto Aplicado II](https://github.com/Enzo-Enrico/Projeto-Aplicado-II-Analise-Sentimento), no qual foi construído um pipeline de NLP para análise de sentimento em avaliações de filmes e séries.

### Alinhamento com os ODS (Agenda 2030 – ONU)

- **ODS 4 – Educação de qualidade:** incentivo ao hábito de leitura por meio de recomendações personalizadas.
- **ODS 10 – Redução das desigualdades:** levar a equipamentos públicos uma tecnologia hoje restrita a grandes plataformas comerciais.
- **ODS 9 – Indústria, inovação e infraestrutura:** artefato aberto e reprodutível para instituições públicas.

## Dataset

[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) (Zajac, 2017) – avaliações explícitas de usuários do Goodreads sobre os 10.000 livros mais avaliados da plataforma.

| Arquivo | Dimensão aprox. | Principais colunas |
|---|---|---|
| `ratings.csv` | 5.976.479 × 3 | `user_id`, `book_id`, `rating` (1–5) |
| `books.csv` | 10.000 × 23 | `book_id`, `authors`, `original_publication_year`, `title`, `language_code`, `average_rating`, `ratings_count`… |
| `to_read.csv` | 912.705 × 2 | `user_id`, `book_id` (sinal implícito "quero ler") |
| `tags.csv` | 34.252 × 2 | `tag_id`, `tag_name` |
| `book_tags.csv` | 999.912 × 3 | `goodreads_book_id`, `tag_id`, `count` |

Cerca de 6 milhões de avaliações de 53.424 usuários sobre 10.000 livros; matriz usuário-item com esparsidade superior a 98%.

## Objetivos

**Geral:** desenvolver e avaliar um sistema de recomendação de livros baseado em filtragem colaborativa, com vistas à aplicação em bibliotecas públicas e escolares.

**Específicos:**
1. Aquisição, análise exploratória e tratamento do dataset.
2. Baselines não personalizados (popularidade e média de avaliações).
3. Filtragem colaborativa por vizinhança (k-NN usuário-usuário e item-item) e por fatoração de matrizes (SVD e variantes).
4. Modelo híbrido incorporando metadados e tags dos livros.
5. Avaliação com RMSE, MAE, Precision@k, Recall@k e NDCG@k.
6. Protótipo funcional de recomendação.
7. Documentação completa e reprodutível.

## Metodologia (visão geral)

```
Dados brutos → EDA → Tratamento → Matriz usuário-item → Modelos (baseline / k-NN / SVD / híbrido) → Avaliação → Protótipo
```

**Ferramentas:** Python, Pandas, NumPy, SciPy, Scikit-learn, Surprise / Implicit, Matplotlib, Seaborn, Jupyter.

## Estrutura do repositório

```
├── data/          # scripts de download e amostras dos dados brutos e tratados
├── notebooks/     # análise exploratória e experimentos de modelagem
├── src/           # módulos Python reutilizáveis (pré-processamento, modelos, avaliação)
├── docs/          # relatórios das etapas, apresentação e guia de implantação
└── README.md
```

## Como executar

```bash
git clone https://github.com/Enzo-Enrico/Projeto-Aplicado-III-Sistema-Recomendacao.git
cd Projeto-Aplicado-III-Sistema-Recomendacao
pip install -r requirements.txt
python data/download.py        # baixa o goodbooks-10k
jupyter notebook notebooks/
```

*(Instruções serão detalhadas conforme o código for desenvolvido.)*

## Cronograma

| Etapa | Atividades | Período | Status |
|---|---|---|---|
| 1 | Tema, grupo, repositório, Capa, Sumário e Introdução | Set/2026 | ✅ Concluída |
| 2 | Aquisição, EDA, tratamento; Referencial Teórico e Metodologia | Set–Out/2026 | ⏳ |
| 3 | Baselines, filtragem colaborativa, modelo híbrido; Resultados preliminares | Out–Nov/2026 | ⏳ |
| 4 | Refinamento, protótipo, guia de implantação, Resumo, Conclusão, relatório e apresentação final | Nov–Dez/2026 | ⏳ |

## Entregas

- [Etapa 1 – Capa, Sumário e Introdução](docs/Projeto_Aplicado_III_Etapa1.pdf)

## Referências

- KOREN, Y.; BELL, R.; VOLINSKY, C. Matrix factorization techniques for recommender systems. *Computer*, v. 42, n. 8, 2009.
- RICCI, F.; ROKACH, L.; SHAPIRA, B. Introduction to recommender systems handbook. In: *Recommender Systems Handbook*. Springer, 2011.
- INSTITUTO PRÓ-LIVRO. *Retratos da Leitura no Brasil*. 5. ed. 2020.
- ONU. *Transformando nosso mundo: a Agenda 2030 para o Desenvolvimento Sustentável*. 2015.
- ZAJAC, Z. *goodbooks-10k: a new dataset for book recommendations*. 2017.
