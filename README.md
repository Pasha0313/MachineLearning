# Machine Learning

Machine-learning fundamentals (worked examples in Python and R) plus applied
projects in finance, business analytics, sports analytics and LLMs.

Every folder follows the same convention: a two-digit prefix gives the order
(`01-`, `02-`, ...), and names are lowercase-kebab-case.

## Repository layout

```
01-ml-fundamentals/          Core algorithms, one folder each (python/ + r/)
02-projects/                 Applied, end-to-end projects grouped by domain
```

## 01 · ML fundamentals

Each algorithm folder contains `python/` (script + notebook + data) and, where
available, `r/`. Files ending in `_practice.py` are my own reworked versions
of the reference script.

| # | Topic | Algorithms |
|---|---|---|
| 01 | [Regression](01-ml-fundamentals/01-regression) | Simple linear · Multiple linear · Polynomial · SVR · Decision tree · Random forest |
| 02 | [Classification](01-ml-fundamentals/02-classification) | Logistic regression · K-NN · SVM · Kernel SVM · Naive Bayes · Decision tree · Random forest |
| 03 | [Clustering](01-ml-fundamentals/03-clustering) | K-means · Hierarchical |
| 04 | [Association rule learning](01-ml-fundamentals/04-association-rule-learning) | Apriori · Eclat |
| 05 | [Reinforcement learning](01-ml-fundamentals/05-reinforcement-learning) | Upper confidence bound · Thompson sampling |
| 06 | [Natural language processing](01-ml-fundamentals/06-natural-language-processing) | Bag-of-words sentiment (restaurant reviews) |
| 07 | [Deep learning](01-ml-fundamentals/07-deep-learning) | Artificial neural network · Convolutional neural network (cats vs dogs, dataset included) |
| 08 | [Dimensionality reduction](01-ml-fundamentals/08-dimensionality-reduction) | PCA · LDA · Kernel PCA |
| 09 | [Model selection & boosting](01-ml-fundamentals/09-model-selection-and-boosting) | k-fold CV & grid search · XGBoost |

Scripts read their data by relative path, so run them from inside their own
`python/` or `r/` folder.

## 02 · Projects

| # | Domain | Project | Summary |
|---|---|---|---|
| 01.01 | Finance | [Algorithmic trading with ML](02-projects/01-finance/01-algorithmic-trading-ml) | Notebook series: preprocessing, classification/regression models, backtesting, parameter optimisation, overfitting and walk-forward testing |
| 01.02 | Finance | [FX time-series forecasting](02-projects/01-finance/02-fx-time-series-forecasting) | FX time-series modelling and risk-aware forecasting |
| 02.01 | Business analytics | [Customer churn prediction](02-projects/02-business-analytics/01-customer-churn-prediction) | EDA and churn classification on a customer dataset |
| 02.02 | Business analytics | [Citi Bike demand forecasting](02-projects/02-business-analytics/02-citibike-demand-forecasting) | Cleaning, aggregation, visualisation and forecasting of 2023 Citi Bike trip data |
| 03.01 | Sports analytics | [Horse racing prediction](02-projects/03-sports-analytics/01-horse-racing-prediction) | Race-outcome modelling with TensorFlow and tree ensembles |
| 04.01 | LLM & GenAI | [Crypto-news LLM fine-tuning](02-projects/04-llm-and-genai/01-crypto-news-llm-finetuning) | Scrape crypto news, preprocess, fine-tune an LLM, evaluate and chat |
| 04.02 | LLM & GenAI | [HS-code classification with LLMs](02-projects/04-llm-and-genai/02-hs-code-classification-llm) | Four-phase pipeline: prompt prep, LLM inference, post-processing, Streamlit review |
| 04.03 | LLM & GenAI | [Foyer Doc-QA](02-projects/04-llm-and-genai/03-foyer-doc-qa) | PDF chatbot with hybrid retrieval and page-level citations (Streamlit app) |

Each project keeps its own entry point (`Main.py` / `main.py` / `app.py` or
numbered notebooks). Some older scripts still contain local absolute data
paths. Point those at your own data before running.
