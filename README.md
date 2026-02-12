# Balance Popularity with Diversity in News Feed Selection Using Integer Linear Programming

# 1. Abstract
Online news platforms aim to maximize user engagement while maintaining topical diversity to ensure balanced content exposure. A purely prediction-driven ranking strategy—selecting the top K articles by predicted popularity—may concentrate content within a narrow set of high-performing topics, thereby reducing diversity.

This project develops a predictive–prescriptive analytics pipeline that integrates machine learning and integer linear programming (ILP) to study the trade-off between predicted popularity and topic diversity. A supervised learning model predicts article shares, and an optimization framework selects homepage articles subject to diversity constraints. Trade-offs are analyzed via Pareto frontier and sensitivity experiments.

# 2. Project Details

## 2.1 Objective
This study addresses the following research question:
> RQ: How much predicted engagement will be sacrificed to increase topical diversity in homepage article selection?

## 2.2 Dataset
UCI Online News Popularity
https://archive.ics.uci.edu/dataset/332/online+news+popularity

## 2.3 Project Outline
The project mirrors a real-world decision-support workflow in digital media platforms. I replicate a predictive–prescriptive pipeline as follow:
1. Build a predictive model for article popularity.
2. Formulate an integer linear program to select K articles following the diversity constraints across topical categories.
4. Quantify the trade-off between popularity maximization and diversity.
5. Evaluate prediction and ILP model robustness across simulations and parameter variations.

# 3. Methodology
## 3.1 Preprocessing
- Standard feature scaling on predictors and log-transformation on target variables
- Selection of content features for model simplicity and reduce collinearity
- Train–test splitting
- Reproducible modeling via controlled random seeds

## 3.2 Predictive Modeling
An XGBoost regression model is trained to predict article shares. Performance metrics:
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
Predicted shares serve as inputs to the optimization stage.

## 3.3 Integer Linear Programming (ILP) Formulation
### 3.3.1 Problem Formulation
After predicting article popularity using XGBoost, the homepage selection problem is formulated as a binary Integer Linear Programming (ILP) model.

Let:

- $N =$ total number of candidate articles
- $\hat{s_i}$ = predicted shares for article $i$ 
- $x_i \in \{0,1}\ = 1$ if article $i$ is selected.
- $K =$ number of articles to display on the homepage
- $a_{ic} = {0,1}$ indicate whether article $i$ belongs to category $c$ 

Objective: maximize $$\max_{x} \sum_{i=1}^{N} s_i x_i$$

subject to:
- homepage capacity: $\sum_{i=1}^{N} x_i = K$

### 3.3.2 Phase I: Fixed-Diversity Policy Model
To enforce editorial diversity, we first introduce a strict policy requiring that each category appear at least once.

Let:
- $a_{ic} = {0,1}$ indicate whether article $i$ belongs to category $c$ 

The program additionally is subject to
- topic representative: $\sum_{i=1}^{N} a_{ic} x_i \ge 1, \quad \forall c \in \{1, \dots, C\}$

This ensures that every category is represented at least once in the homepage selection. We solve this model across multiple random seeds (via re-trained XGBoost models) to assess:

- Stability of selected sets
- Opportunity cost relative to naive top-10 selection

The opportunity cost is defined as: 
$\Delta =$ $\sum_{i \in \{Top-10}}{\hat{s_i}} - \sum_{i \in \{ILP}}{\hat{s_i}}$ 

### 3.3.3 Phase II: Parametric Diversity Model (Extended Solver)
The strict full-coverage policy may be overly restrictive in practice. To systematically study the trade-off between popularity and diversity, we generalize the model.

We introduce binary category activation variables:
- $y_c = {0,1} = 1$ indicating whether category $c$ is represented.

Now the full model become:

Decision Variables:
- $x_i \in \{0,1}\ = 1$ if article $i$ is selected.
- $y_c \in \{0,1}\ = 1$ indicating whether category $c$ is represented.

**Objective:** $$\max_{x} \sum_{i=1}^{N} s_i x_i$$

subject to:
- $\sum_{i=1}^{N} x_i = K$
- $\sum_{i=1}^{N} a_{ic} x_i \ge y_c, \quad \forall c$
- $\sum_{c=C} y_{c} \ge D, \quad \forall c$

### 3.3.4 Trade-off and Pareto Frontier
By varying $D$, we construct a Pareto frontier between:
- Total predicted shares
- Number of distinct categories

Each value of $D$ corresponds to a policy scenario:
- $D=0$: Pure popularity maximization
- $D=7$: Full categorical coverage
- Intermediate $D$: Balanced strategies

This enables quantitative analysis of the marginal cost of diversity:

Marginal Cost $(D)=f(D−1)−f(D)$

where $f(D)$ denotes the optimal predicted shares under minimum diversity level $D$.

### 3.3.5 Managerial Interpretation

The parametric ILP framework allows decision-makers to:
- Explicitly control diversity policy
- Quantify the engagement cost of enforcing diversity
- Select a policy level aligned with editorial or brand strategy

Rather than treating diversity as a binary constraint, the model enables systematic exploration of the diversity–popularity trade-off.

# Results
Key findings:
- Solution composition varies across random seeds but remains structurally consistent.
- Moderate diversity (around 3-4 topics in the homepage) constraints incur minimal engagement loss while promoting topic diversity.

These results suggest that diversity can be increased strategically without substantial loss in predicted engagement.

# Discussions
This study demonstrates the value of integrating machine learning with optimization: enabling policy-aware decision making. 

The framework can be generalized to:
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

