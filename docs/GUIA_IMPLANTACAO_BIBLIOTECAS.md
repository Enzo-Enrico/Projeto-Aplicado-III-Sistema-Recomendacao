# Guia de implantação – Recomendador de livros para bibliotecas

**Para quem é este guia:** bibliotecários, professores e gestores de bibliotecas públicas, comunitárias ou escolares que queiram oferecer sugestões de leitura personalizadas aos seus frequentadores, sem custo de licença.

**O que o sistema faz:** a partir do que uma pessoa já leu (ou já emprestou), ele sugere uma lista de livros que leitores com gostos parecidos também apreciaram. Não usa dados pessoais — só a lista de títulos.

---

## 1. O que você precisa

- Um computador comum (Windows, Mac ou Linux) com pelo menos 4 GB de memória.
- Python 3.10 ou superior instalado ([python.org](https://www.python.org/downloads/)).
- Cerca de 10 minutos para a instalação e 3 minutos para o primeiro treinamento.

## 2. Instalação (uma única vez)

Abra o terminal (Prompt de Comando no Windows) e digite, linha por linha:

```bash
git clone https://github.com/Enzo-Enrico/Projeto-Aplicado-III-Sistema-Recomendacao.git
cd Projeto-Aplicado-III-Sistema-Recomendacao
pip install -r requirements.txt
python data/download.py
cd src
python recomendar.py --treinar
```

A última linha treina o modelo com 6 milhões de avaliações públicas de leitores e salva o resultado. Só precisa ser feita uma vez.

## 3. Uso no dia a dia

### Sugestões para um leitor a partir do que ele já leu

```bash
python recomendar.py --livros "Harry Potter" "O Hobbit" "Percy Jackson" --n 10
```

O sistema procura os títulos no catálogo, confirma qual edição encontrou e devolve 10 sugestões com autor, ano e nota média. Para salvar a lista em uma planilha:

```bash
python recomendar.py --livros "Sapiens" "Freakonomics" --n 10 --csv sugestoes_joao.csv
```

Dicas:
- Use 3 a 10 títulos que a pessoa realmente gostou — quanto mais, melhor a sugestão.
- Os títulos são buscados em inglês (idioma do catálogo de referência). "O Hobbit" funciona porque o título contém "Hobbit"; para outros, use o título original.
- Se um título não for encontrado, o sistema avisa e segue com os demais.

### Ideias de uso na biblioteca

- **Balcão de atendimento:** o bibliotecário pergunta 3 livros favoritos e imprime a lista.
- **Clube de leitura / sala de aula:** cada aluno informa seus livros preferidos; o professor usa as listas para montar a próxima rodada de leituras.
- **Marcadores de página:** gerar listas para os 20 livros mais emprestados e imprimir "quem leu este, também gostou de…".

## 4. Adaptando ao acervo da sua biblioteca

O modelo foi treinado com avaliações públicas do Goodreads, portanto sugere livros que podem não estar no seu acervo. Dois caminhos:

1. **Filtro simples (recomendado para começar):** peça uma lista maior (`--n 50`) e salve em CSV; cruze com a planilha do acervo e mantenha só os títulos disponíveis.
2. **Treinar com seus próprios dados (requer apoio técnico):** se a biblioteca tem histórico de empréstimos (quem pegou o quê), basta gerar um arquivo `ratings.csv` com as colunas `user_id, book_id, rating` (use `rating = 5` para cada empréstimo, se não houver nota) e um `books.csv` com `book_id, title, authors`. Coloque-os na pasta `data/` e rode `python recomendar.py --treinar`. Os códigos de usuário podem ser anônimos (números).

## 5. Privacidade

O sistema não precisa de nome, documento ou contato: apenas códigos numéricos de leitor e os títulos. Recomenda-se anonimizar os identificadores antes de usar históricos de empréstimo, em conformidade com a LGPD.

## 6. Limitações conhecidas

- O catálogo de referência é majoritariamente em inglês e contém algumas edições duplicadas (box sets); às vezes o sistema sugere outra edição de um livro já lido.
- Leitores com menos de 3 livros informados recebem sugestões mais genéricas (próximas dos livros mais populares).
- As sugestões refletem o gosto da comunidade do Goodreads, não um juízo de qualidade literária. Use-as como ponto de partida para a conversa com o leitor, não como substituto da mediação do bibliotecário.

## 7. Suporte e contato

Dúvidas, sugestões e relatos de uso podem ser abertos como *issues* no repositório do projeto no GitHub. O código é aberto (uso livre para fins educacionais e sem fins lucrativos).

*Projeto Aplicado III – Ciência de Dados, Universidade Presbiteriana Mackenzie – Enzo Enrico Braga Cavalcante, Guarulhos, 2026.*
