# Data Documentation

## Source
UCI Machine Learning Repository: Online News Popularity Dataset
- **URL**: https://archive.ics.uci.edu/dataset/332/online+news+popularity
- **Citation**: K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision 
  Support System for Predicting the Popularity of Online News. Proceedings of the 17th 
  EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.

## Retrieval
Data is fetched programmatically using the `ucimlrepo` package:
```python
from ucimlrepo import fetch_ucirepo
online_news_popularity = fetch_ucirepo(id=332)
```

## Dataset Description
- **Instances**: 39,644 articles
- **Features**: 58 (numerical and binary categorical)
- **Target**: Number of shares
- **Period**: Articles published by Mashable over 2 years

## Preprocessing Steps
See `src/data/preprocess.py` for:
1. Feature-target separation
2. Log transformation of shares
3. Standard scaling of numerical features
4. Topic category extraction
5. Train-test split (80-20, random_state=42)

## Processed Data
Processed datasets are saved in `data/processed/`:
- `X_train.csv`
- `y_train.csv`
- `X_test.csv`
- `y_test.csv`

To regenerate, run:
```bash
python scripts/preprocess_data.py
```
