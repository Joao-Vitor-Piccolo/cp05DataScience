# Checkpoint 05 — Qualidade de vinho tinto (regressão)

Projeto da disciplina **Data Science & Statistical Computing** (FIAP, 2026): comparação de Random Forest, XGBoost e LightGBM para prever a nota sensorial (`quality`) a partir do perfil químico do vinho tinto.

## Identificação do grupo

Nomes e RMs também estão em [`RMS.md`](RMS.md) e no notebook.

| Nome | RM |
|---|---|
| João Vitor Piccolo | 565127 |
| Gabrielle Calazans | 564460 |
| Jéssica Domingues | 562973 |
| Kauã Carvalho | 566371 |
| Leonardo Pereira | 561349 |

## Problema

- **Pergunta:** dado o perfil químico de um vinho tinto, qual nota sensorial esperar?
- **Tipo:** regressão. A previsão \(\hat{y}\) é a nota na escala 0–10 (nesta base, 3–8).
- **Base:** [`winequality-red.csv`](winequality-red.csv) — UCI Wine Quality (Cortez et al., 2009), 1.599 linhas × 12 colunas. Após remover 240 duplicatas: 1.359 amostras.
- **Métrica principal:** RMSE. Auxiliares: MAE e \(R^2\).

## Resultado

Desenvolvimento só em \(\mathcal{D}_{\mathrm{treino}}\) + validação cruzada. O teste foi usado uma única vez.

| Configuração | RMSE CV | RMSE teste |
|---|---:|---:|
| **Random Forest + Optuna (modelo final)** | **0,636** | **0,635** |
| Random Forest + Grid Search | 0,638 | — |
| LightGBM + Optuna | 0,640 | — |
| XGBoost + Optuna | 0,643 | — |

Hiperparâmetros finais: `n_estimators=200`, `max_depth=23`, `min_samples_leaf=6`, `max_features≈0,72`, `random_state=42`.

MAE no teste: 0,489. \(R^2\) no teste: 0,399.

## Como reproduzir o treinamento

```bash
python -m pip install -r requirements.txt
```

Abra [`Checkpoint05_RF_XGBoost_LightGBM.ipynb`](Checkpoint05_RF_XGBoost_LightGBM.ipynb) e execute **Run All**. O Exercício 7 grava `models/wine_quality_pipeline.joblib`.

O conjunto de teste não entra em seleção de variáveis, comparação de modelos, Grid Search, Optuna nem na escolha do modelo final.

## Como rodar a aplicação

```bash
streamlit run app.py
```

A interface carrega o **mesmo** `joblib` avaliado no notebook e inclui o teste de paridade (caso de nota 5 com \(\hat{y} \approx 4{,}999839\)).

## Estrutura do repositório

```
.
├── Checkpoint05_RF_XGBoost_LightGBM.ipynb
├── winequality-red.csv
├── app.py
├── requirements.txt
├── README.md
├── RMS.md
└── models/
    └── wine_quality_pipeline.joblib
```

## Links

- GitHub: https://github.com/Joao-Vitor-Piccolo/Cp05DataScience
- Streamlit: execute localmente com `streamlit run app.py` (URL Cloud a inserir após o deploy)

## Fonte dos dados

P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 2009. Repositório UCI Machine Learning: Wine Quality (red).
