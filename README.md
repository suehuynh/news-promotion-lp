# Balance Shares and Topic Diversity in News Homepage Selection Using Integer Linear Programming

# Abstract
Online news platforms aim to maximize user engagement while maintaining topical diversity to ensure balanced content exposure. A purely prediction-driven ranking strategy—selecting the top K articles by predicted popularity—may concentrate content within a narrow set of high-performing topics, thereby reducing diversity.

This project develops a predictive–prescriptive analytics pipeline that integrates machine learning and integer linear programming (ILP) to study the trade-off between predicted popularity and topic diversity. A supervised learning model predicts article shares, and an optimization framework selects homepage articles subject to diversity constraints. Trade-offs are analyzed via Pareto frontier and sensitivity experiments.

# Project Details

## Objective
This study addresses the following research question:
> RQ: How much predicted engagement will be sacrificed to increase topical diversity in homepage article selection?

## Dataset
UCI Online News Popularity
https://archive.ics.uci.edu/dataset/332/online+news+popularity

## Project Outline
The project mirrors a real-world decision-support workflow in digital media platforms, following a predictive–prescriptive pipeline.
1. Builds a predictive model for article popularity.
2. Formulates an integer linear program to select K articles.
3. Imposes diversity constraints across topical categories.
4. Quantifies the trade-off between popularity maximization and diversity.
5. Evaluates robustness across simulations and parameter variations.
6. The project mirrors a real-world decision-support workflow in digital media platforms.

# Methodology
## Preprocessing
- Standard feature scaling on predictors and log-transformation on target variables
- Selection of content features for model simplicity and reduce collinearity
- Train–test splitting
- Reproducible modeling via controlled random seeds

## Predictive Modeling
An XGBoost regression model is trained to predict article shares. Performance metrics:
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
Predicted shares serve as inputs to the optimization stage.

## Integer Linear Programming (ILP) Formulation

## Sensitivity Analysis and Pareto Frontier
To evaluate the trade-off between diversity and predicted engagement:

1. The minimum diversity requirement D is varied.
2. For each D, the ILP is solved.
3. The total predicted shares are recorded.
4. Experiments are repeated across multiple random seeds.
5. The Pareto frontier is constructed by plotting:
- X-axis: number of distinct topics
- Y-axis: total predicted shares

This reveals the marginal opportunity cost of increasing diversity.

# Results
Key findings:
- Moderate diversity constraints incur minimal engagement loss.
- Strict diversity requirements introduce measurable opportunity costs.
- The ILP framework outperforms naive top-K selection under diversity constraints.
- Solution composition varies across random seeds but remains structurally consistent.

These results suggest that diversity can be increased strategically without substantial loss in predicted engagement.

# Discussions
This study demonstrates the value of integrating machine learning with optimization:
- Prediction alone provides ranking.
- Optimization enables policy-aware decision making.
- Sensitivity analysis quantifies trade-offs.
- Pareto analysis provides interpretable business insights.

The framework generalizes to:
- Content recommendation systems
- Advertisement allocation
- Fairness-aware ranking systems
- Resource allocation problems with diversity constraints

# Limitations
- Dataset lacks temporal and personalization features.
- Predictive model performance is moderate (limited R²).
- Optimization relies on predicted shares rather than realized engagement.
- Topic classification is represented via binary indicators rather than hierarchical taxonomy.

# Future Work
Future extensions may include:
1. Robust optimization under predictive uncertainty.
2. Multi-objective optimization with fairness penalties.
3. Dynamic (time-aware) article selection models.
4. Personalized homepage optimization.
5. Bayesian modeling of prediction uncertainty.
