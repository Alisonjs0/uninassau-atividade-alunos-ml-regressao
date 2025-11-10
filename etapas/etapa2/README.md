# Etapa 2: Pré-Processamento de Dados

## 🎯 Objetivos

Nesta etapa, seu objetivo é limpar e transformar os dados brutos que você analisou na Etapa 1, preparando-os para serem usados em um modelo de Machine Learning. Você irá aplicar técnicas para tratar problemas de qualidade e para converter os dados em um formato numérico e padronizado.

---

## 📝 Tarefas Principais

Seu trabalho será documentado em um notebook Jupyter (`notebooks/02_Preprocessamento.ipynb`). Organize seu notebook seguindo as seções abaixo. Para cada tarefa, escreva o código necessário e, em seguida, use células de Markdown para documentar suas decisões.

### 1. Tratamento de Valores Faltantes

**Por que é importante?** A maioria dos algoritmos de Machine Learning não consegue lidar com valores ausentes (`NaN`). Deixá-los no dataset resultará em erros.

**O que fazer:**
- **Para colunas numéricas:** Preencha os valores faltantes usando uma estratégia de imputação.
  - **Como fazer (dicas):** Use o `SimpleImputer` do Scikit-learn ou o método `.fillna()` do Pandas.
  - **Estratégia `median` (mediana):** Mais segura e recomendada se a coluna tiver outliers ou uma distribuição assimétrica.
  - **Estratégia `mean` (média):** Funciona bem para colunas com distribuição simétrica (semelhante a um sino).
- **Para colunas categóricas:** Preencha os valores faltantes com a categoria mais comum.
  - **Como fazer (dicas):** Use a estratégia `most_frequent` (moda) do `SimpleImputer` ou o método `.fillna(df['coluna'].mode()[0])`.

⚠️ **Cuidado com Data Leakage:** Para evitar vazar informação dos dados de teste para o treino, o ideal é que qualquer cálculo (como média ou mediana) seja feito **apenas** com os dados de treino e depois aplicado aos dados de teste. Em um primeiro momento, você pode fazer no dataset todo para simplificar, mas tenha essa boa prática em mente para a Etapa 3.

---

### 2. Tratamento de Outliers

**Por que é importante?** Valores extremos (outliers) podem distorcer a escala das features e influenciar negativamente o treinamento de alguns modelos, especialmente os lineares.

**O que fazer:**
- Com base na sua análise da Etapa 1, decida como tratar os outliers em colunas numéricas importantes.
- **Como fazer (dicas):**
  - **Remoção:** Se você tem certeza de que o outlier é um erro de medição ou digitação e eles são poucos, você pode removê-los. Use com cuidado para não perder dados valiosos. Ex: `df = df[df['coluna'] < valor_maximo]`.
  - **Capping (Limitar):** Uma abordagem mais segura é "aparar" os outliers, substituindo-os por um valor máximo ou mínimo aceitável (por exemplo, o limite do "whisker" do boxplot, que é `Q3 + 1.5 * IQR`).
  - **Manter:** Se os outliers são valores raros, mas legítimos (ex: uma venda de valor muito alto em um e-commerce), pode ser melhor mantê-los. Transformações (como a de log) podem ajudar a reduzir seu impacto.

---

### 3. Encoding de Variáveis Categóricas

**Por que é importante?** Modelos de ML trabalham com números, não com texto. Precisamos converter colunas categóricas (como "gênero" ou "cidade") em um formato numérico.

**O que fazer:**
- Converta todas as colunas de texto (tipo `object`) em representações numéricas.
- **Como fazer (dicas):**
  - **One-Hot Encoding (para variáveis nominais):** Use esta técnica para colunas onde as categorias **não têm uma ordem** natural (ex: `cidade`, `cor_favorita`). Ela cria novas colunas binárias (0 ou 1) para cada categoria.
    - **Ferramenta:** `pd.get_dummies(df, columns=['coluna_a', 'coluna_b'], drop_first=True)`.
    - O `drop_first=True` é importante para remover uma das categorias, evitando redundância de informação (multicolinearidade).
  - **Label Encoding / Ordinal Encoding (para variáveis ordinais):** Use para colunas onde as categorias **têm uma ordem** clara (ex: `ruim`, `médio`, `bom`). Ela atribui um número inteiro a cada categoria (ex: 0, 1, 2), preservando a ordem.
    - **Ferramenta:** `OrdinalEncoder` do Scikit-learn ou o método `.map({'ruim': 0, 'médio': 1, 'bom': 2})` do Pandas.

---

### 4. Normalização de Variáveis Numéricas

**Por que é importante?** Features com escalas muito diferentes (ex: `idade` variando de 18 a 70 e `salário` variando de 1.000 a 100.000) podem fazer com que o modelo dê mais importância à feature com a escala maior. A normalização coloca todas na mesma escala.

**O que fazer:**
- Aplique uma técnica de scaling a todas as suas colunas numéricas (após tratar outliers e faltantes).
- **Como fazer (dicas):**
  - **StandardScaler:** Transforma os dados para que tenham média 0 e desvio padrão 1. É a técnica mais comum e funciona bem para a maioria dos algoritmos.
  - **MinMaxScaler:** Transforma os dados para que fiquem em um intervalo específico, geralmente entre 0 e 1.
- **Salve o scaler:** Após treinar o scaler (`scaler.fit(dados_de_treino)`), é **crucial** salvá-lo em um arquivo.
  - **Ferramenta:** `import joblib; joblib.dump(scaler, 'models/scaler.pkl')`.
  - Isso garante que você poderá aplicar **exatamente a mesma transformação** nos dados de teste e em novos dados no futuro.

---

### 5. Feature Engineering (Opcional)

**Por que é importante?** Às vezes, as colunas originais não contêm toda a informação. Criar novas features pode ajudar o modelo a encontrar padrões que não eram óbvios antes.

**O que fazer:**
- Crie 1 ou 2 novas colunas a partir das existentes que você acredita que possam ser úteis.
- **Como fazer (dicas de ideias):**
  - **Criar uma razão:** Se você tem `distancia_km` e `tempo_minutos`, pode criar `velocidade_media = distancia_km / (tempo_minutos / 60)`.
  - **Combinar features:** Se você tem `numero_de_filhos` e `estado_civil`, pode criar uma feature binária `tem_familia_grande`.
  - **Extrair de datas:** Se tiver uma coluna de data, pode extrair o dia da semana, o mês ou se é um fim de semana.

---

## 📊 Entregáveis

1.  **Notebook (`notebooks/02_Preprocessamento.ipynb`):** Contendo todo o código e as justificativas em Markdown para suas decisões.
2.  **Dataset Limpo (`data/students_clean.csv`):** O DataFrame final, após todas as transformações, salvo em um novo arquivo CSV.
3.  **Scaler Salvo (`models/scaler.pkl`):** O objeto do scaler treinado e salvo com `joblib`.

---

## 🎤 Apresentação (5 minutos)

Prepare uma apresentação curta e objetiva (4-5 slides) para resumir seu trabalho.

- **Slide 1: Resumo dos Problemas Corrigidos:** Quantos valores faltantes foram tratados? Quantos outliers foram identificados e o que você fez com eles?
- **Slide 2: Principais Transformações:** Mostre um exemplo de encoding (One-Hot ou Label) e como ficou o resultado. Mostre um gráfico de uma variável antes e depois da normalização.
- **Slide 3: Feature Engineering (se aplicável):** Apresente a(s) nova(s) feature(s) que você criou e por que acredita que ela(s) pode(m) ser útil(is).
- **Slide 4: Resultado Final:** Mostre as dimensões do dataset antes e depois (`X` linhas, `Y` colunas -> `X` linhas, `Z` colunas) e declare que os dados estão prontos para a modelagem.

---

## ✅ Checklist de Sucesso

- [ ] Seu notebook está organizado com títulos para cada tarefa de pré-processamento.
- [ ] Todas as decisões (ex: por que usou mediana, por que removeu outliers) estão justificadas em Markdown.
- [ ] O dataset limpo foi salvo corretamente em `data/students_clean.csv`.
- [ ] O scaler foi salvo em `models/scaler.pkl`.
- [ ] O notebook executa do início ao fim sem erros.

**Bom trabalho!** 🚀