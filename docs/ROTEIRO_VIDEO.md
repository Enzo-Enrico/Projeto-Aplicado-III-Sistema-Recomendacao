# Roteiro do vídeo de apresentação – Projeto Aplicado III

## Apresentação do projeto (5 minutos, YouTube)

Formato sugerido: gravação de tela dos slides (`docs/Apresentacao_PA3.pdf`) com narração, mais 40 segundos de demonstração do terminal no final. Ensaie uma vez cronometrando; o texto abaixo tem cerca de 720 palavras, o que dá 5 minutos em ritmo normal.

---

**[0:00–0:30] Slide 1 – Abertura**

Olá, meu nome é Enzo Enrico Braga Cavalcante, aluno de Ciência de Dados do Mackenzie, e este é o meu Projeto Aplicado III: um sistema de recomendação de livros baseado em filtragem colaborativa, pensado como ferramenta aberta para bibliotecas públicas e escolares incentivarem a leitura. Nos próximos cinco minutos vou apresentar o problema, os dados, o método, os resultados e o que aprendi.

**[0:30–1:10] Slide 2 – Problema e objetivo**

A pesquisa Retratos da Leitura no Brasil mostra que metade dos brasileiros não é leitora, e que a principal barreira não é falta de acesso ao livro, mas dificuldade de encontrar algo interessante. Isso é um problema de descoberta — exatamente o que sistemas de recomendação resolvem. Grandes plataformas têm isso; bibliotecas públicas, não. O objetivo foi desenvolver e avaliar um recomendador de livros e disponibilizá-lo como código aberto, com guia de implantação, alinhado aos ODS 4, 9 e 10 da ONU.

**[1:10–1:50] Slide 3 – Dados**

Usei o goodbooks-10k: 6 milhões de avaliações de 53 mil leitores sobre 10 mil livros do Goodreads. A matriz usuário-livro tem 98,9% de células vazias, e 20% dos livros concentram dois terços das avaliações — uma cauda longa típica. Isso já indicava dois cuidados: popularidade seria um baseline difícil de bater, e eu precisaria medir se o sistema só recomenda best-sellers.

**[1:50–2:40] Slide 4 – Metodologia**

O pipeline: análise exploratória, matriz esparsa, divisão por usuário em treino, validação e teste, e comparação de cinco técnicas — popularidade, vizinhança item-item, SVD com vieses, PureSVD e ALS para feedback implícito. O ponto mais importante do método foi separar dois tipos de avaliação: erro de predição de nota, com RMSE e MAE, e qualidade da lista, com Precision, Recall e NDCG em 10, calculados sobre todos os 10 mil livros, sem amostragem. Como o produto final é uma lista, o NDCG@10 foi o critério de decisão. Hiperparâmetros foram escolhidos só na validação, e o teste foi usado uma única vez.

**[2:40–3:40] Slide 5 – Resultados**

Primeiro achado: o modelo com menor erro de predição, o SVD, gera a pior lista de todas — pior que recomendar best-sellers. Ele coloca no topo livros de nicho com nota alta e três avaliações. Se eu tivesse escolhido pelo RMSE, escolheria o pior modelo. Segundo achado: o ALS implícito, depois do ajuste de hiperparâmetros, ficou com NDCG@10 de 0,318 e Precision@10 de 0,278 — ou seja, quase 3 em cada 10 sugestões são livros que o leitor de fato avaliaria com nota 4 ou 5. Isso é 38% melhor que a configuração padrão e 3,6 vezes o baseline. Terceiro: o k-NN é menos preciso, mas cobre 73% do catálogo, contra 39% do ALS — para uma biblioteca, que quer ampliar repertório, isso importa. E um resultado negativo: usar as marcações "quero ler" como se fossem leituras piorou o modelo.

**[3:40–4:20] Slide 6 – Demonstração**

*(alternar para o terminal)* Aqui está o protótipo. Digito três livros que um leitor gostou — Harry Potter, O Hobbit e Percy Jackson — e o sistema devolve Nárnia, Jogos Vorazes, Senhor dos Anéis. Agora um leitor de não ficção: Sapiens, Rápido e Devagar, Freakonomics — e ele sugere Outliers, O Ponto da Virada, Previsivelmente Irracional. Isso funciona para leitores novos, sem re-treinar o modelo, por folding-in. Roda em qualquer computador em menos de 3 minutos.

**[4:20–5:00] Slide 7 – Conclusão e próximos passos**

Concluindo: a métrica tem que refletir a tarefa — ranking, não nota; baselines bem ajustados são fortes, como a literatura recente alerta; e acurácia não basta: cobertura e viés de popularidade precisam entrar na conta. Como trabalhos futuros: combinar a precisão do ALS com a cobertura do k-NN, deduplicar edições, adaptar o catálogo para português e fazer um piloto com leitores reais. O código, o modelo e o guia de implantação estão no GitHub — link na descrição. Obrigado!

---
