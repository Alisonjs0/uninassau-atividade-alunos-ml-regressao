# Etapa 1: Análise Exploratória de Dados (EDA)

## 🎯 Objetivos

Nesta etapa, seu objetivo é realizar uma investigação completa no dataset do projeto. Você irá explorar os dados para entender suas características, identificar problemas de qualidade e descobrir os primeiros insights que guiarão as próximas etapas do projeto.

**Lembre-se:** O foco aqui é **analisar**, e não modificar os dados.

---

## 📝 Tarefas Principais

Seu trabalho será documentado em um notebook Jupyter (`notebooks/01_EDA.ipynb`). Organize seu notebook seguindo as seções abaixo. Para cada tarefa, escreva o código necessário e, em seguida, use células de Markdown para documentar suas observações e conclusões.

### 1. Carregamento e Visão Geral dos Dados

Comece carregando o dataset e fazendo uma inspeção rápida para entender sua estrutura. É o primeiro contato com os dados, essencial para garantir que eles foram carregados corretamente e para ter uma ideia geral do que você tem em mãos.

**O que fazer:**
- **Carregue o arquivo CSV:** Use a função `pd.read_csv('caminho/para/seu/arquivo.csv')` do Pandas para carregar os dados.
- **Inspecione o início e o fim:** Use os métodos `.head()` e `.tail()` no seu DataFrame para visualizar as primeiras e últimas linhas. Isso ajuda a identificar se o arquivo foi lido corretamente e se há algum padrão óbvio ou problema no final do arquivo.
- **Verifique a estrutura:**
    - Use `.shape` para ver o número de linhas e colunas. Exemplo: `(1000, 15)` significa 1000 linhas e 15 colunas.
    - Use `.info()` para obter um resumo técnico, incluindo o tipo de dado de cada coluna (`int64`, `float64`, `object`) e a contagem de valores não nulos. É ótimo para uma primeira detecção de valores faltantes.
- **Calcule estatísticas descritivas:** Use `.describe()` para gerar estatísticas como média, mediana, desvio padrão, mínimo e máximo para todas as colunas numéricas. Isso dá uma noção da escala e distribuição de cada variável.

**O que documentar:**
- Uma breve descrição do dataset.
- O número de linhas e colunas.
- Uma lista das variáveis numéricas e categóricas.
- A identificação da sua variável alvo (a que você quer prever).

---

### 2. Análise de Qualidade dos Dados

Investigue problemas comuns que podem afetar a qualidade do seu modelo no futuro. Dados "sujos" (com valores faltantes ou outliers) podem distorcer análises e piorar o desempenho de modelos de machine learning.

**O que fazer:**
- **Calcule valores faltantes:** Use `.isnull().sum()` no seu DataFrame para contar quantos valores `NaN` (Not a Number) existem em cada coluna. Para ver em porcentagem, você pode dividir o resultado pelo total de linhas: `(df.isnull().sum() / len(df)) * 100`.
- **Crie um gráfico de barras:** Use Matplotlib ou Seaborn para criar um gráfico de barras com as porcentagens de valores faltantes. Isso torna a visualização do problema muito mais clara e impactante.
- **Gere boxplots para identificar outliers:** Para cada variável numérica, crie um boxplot. Esta é uma das formas mais eficazes de visualizar a dispersão dos dados e identificar valores que fogem muito do padrão.

**Como ler um Boxplot:**
Um boxplot resume a distribuição de uma variável numérica e é ótimo para identificar outliers.
- A **linha no meio da caixa** é a **mediana** (o valor central, ou Quartil 2 - Q2). 50% dos dados estão abaixo deste valor.
- A **parte inferior da caixa** é o **Primeiro Quartil (Q1)**. 25% dos dados estão abaixo deste valor.
- A **parte superior da caixa** é o **Terceiro Quartil (Q3)**. 75% dos dados estão abaixo deste valor.
- A **altura da caixa** representa o **Intervalo Interquartil (IQR = Q3 - Q1)**, que contém 50% dos dados centrais.
- As **"whiskers" (linhas que se estendem da caixa)** mostram a amplitude dos dados, geralmente até 1.5 vezes o IQR a partir de Q1 e Q3.
- **Pontos individuais fora dos whiskers** são considerados **outliers** — valores atipicamente altos ou baixos em comparação com o resto dos dados.

**O que documentar:**
- Liste as colunas que contêm valores faltantes e a porcentagem de cada uma.
- Liste as colunas que parecem ter outliers, com base nos boxplots.
- Formule uma hipótese inicial sobre por que os dados estão faltando (ex: erro de coleta, não aplicável, etc.).

---

### 3. Análise Univariada

Analise cada variável individualmente para entender sua distribuição e características. Isso ajuda a compreender o comportamento de cada feature antes de começar a cruzá-las.

**O que fazer:**
- **Para variáveis numéricas:** Crie um **histograma** para cada uma. O histograma agrupa os números em intervalos e mostra a frequência de cada intervalo. Isso ajuda a ver onde os valores se concentram e se a distribuição é simétrica, assimétrica, bimodal, etc. Use `df['coluna'].hist()` ou `sns.histplot()`. Adicionar um gráfico de densidade (`kde=True` no Seaborn) suaviza o histograma e ajuda a ver a forma da distribuição.
- **Para variáveis categóricas:** Crie um **gráfico de barras** para cada uma. Use `df['coluna'].value_counts().plot(kind='bar')` ou `sns.countplot()`. Isso mostrará quantas vezes cada categoria aparece, ajudando a identificar desbalanceamentos (quando uma categoria é muito mais frequente que as outras).

**O que documentar:**
- Descreva a forma da distribuição da sua variável alvo. Ela é simétrica? Assimétrica?
- Anote qualquer observação interessante sobre as distribuições das outras variáveis (ex: "a maioria dos alunos não tem tutoria", "a faixa de preço dos imóveis se concentra abaixo de X").

---

### 4. Análise Bivariada

Investigue a relação entre pares de variáveis para encontrar padrões e correlações. É aqui que você começa a descobrir quais features podem ser importantes para prever sua variável alvo.

**O que fazer:**
- **Calcule a matriz de correlação:** Use o método `.corr()` no seu DataFrame para calcular a correlação de Pearson entre todas as variáveis numéricas. O resultado varia de -1 (correlação negativa perfeita) a +1 (correlação positiva perfeita). Um valor próximo de 0 indica ausência de correlação linear.
- **Visualize com um heatmap:** Um heatmap (`sns.heatmap()`) transforma a matriz de correlação em um mapa de cores, tornando muito mais fácil identificar visualmente as variáveis que são fortemente correlacionadas (cores fortes, positivas ou negativas).
- **Crie scatter plots:** Para as variáveis numéricas que mostraram maior correlação com sua variável alvo, crie um gráfico de dispersão (`sns.scatterplot()`). Coloque a variável alvo no eixo Y e a outra variável no eixo X. Isso ajuda a confirmar visualmente a relação (ex: uma nuvem de pontos que sobe ou desce).
- **Crie boxplots comparativos:** Para variáveis categóricas, use `sns.boxplot()` para comparar a distribuição da variável alvo entre as diferentes categorias. Por exemplo, um boxplot de `final_grade` por `tutoring` mostrará lado a lado a distribuição de notas para alunos que têm e não têm tutoria.

**O que documentar:**
- Quais variáveis numéricas têm a correlação mais forte (positiva ou negativa) com a variável alvo?
- Existe alguma relação interessante ou inesperada que você observou nos gráficos?
- Qual variável categórica parece ter o maior impacto na variável alvo?

---

## 📊 Entregável

- **Notebook Jupyter (`notebooks/01_EDA.ipynb`):** Um notebook bem organizado contendo todo o código, visualizações e documentação da sua análise.

## ✅ Checklist de Sucesso

- [ ] Seu notebook está organizado com títulos para cada uma das 4 tarefas.
- [ ] Todas as tarefas foram executadas e documentadas.
- [ ] Os gráficos estão claros, com títulos e rótulos nos eixos.
- [ ] O notebook executa do início ao fim sem erros.
- [ ] As principais conclusões sobre os dados estão resumidas ao final do notebook.

**Bom trabalho!** 🔍
