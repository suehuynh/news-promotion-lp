import pulp

def news_solver(
        popularity,
        topics,
        tech_indicator,
        other_indicator,
        keywords,
        images,
        hrefs,
        avg_keywords,
        avg_images,
        avg_hrefs,
        lambda_tech=0.0,
        K=10,
        other_lower_bound=2,
        solver_name="PULP_CBC_CMD",
        verbose=False
):
    """
    Build ILP model for news article selection.

    Parameters
    ----------
    popularity : array-like, shape (N,)
        (Predicted) raw shares
    topics : array-like, shape (N,)
        Topic label for each article
    keywords : array-like, shape (N,)
        Number of keyowrds per article title
    images : array-like, shape (N,)
        Number of images per article
    hrefs : array-like, shape (N,)
        Number of hyperlinks per article
    avg_keywords: float
    avg_images : float
    avg_hrefs : float
    tech_indicator : array-like, shape (N,)
        1 if tech article, 0 otherwise
    lambda_tech : float
        Weight for tech articles
    K : int
        Number of articles to select
    
    Return
    ----------
    selected_indices: list
        List of selected optimal indices
    status: str
        Solver status
    """

    popularity = popularity
    topics = topics
    images = images
    hrefs = hrefs
    keywords = keywords
    tech_indicator = tech_indicator
    N = len(popularity)
    unique_topics = list(set(topics))
    # Model
    model = pulp.LpProblem("News Selection", pulp.LpMaximize)
    
    # Decision Variables
    x = pulp.LpVariable.dicts("select", range(N), cat="Binary")
    y = pulp.LpVariable.dicts("topic", unique_topics, cat="Binary")

    # Objective
    model += pulp.lpSum(
        (popularity[i] + lambda_tech * tech_indicator[i]) * x[i]
        for i in range(N)
    )

    # Capacity constraints
    model += pulp.lpSum(x[i] for i in range(N)) == K

    # Topic activation constraints
    for i in range(N):
        model += x[i] <= y[topics[i]]

    # Topic diversity constraints
    model += pulp.lpSum(y[t] for t in unique_topics) >= 3

    # Keyword constraints
    model += pulp.lpSum((keywords[i] - avg_keywords) * x[i] for i in range(N)) >= 0

    # Images constraints
    model += pulp.lpSum((images[i] - avg_images) * x[i] for i in range(N)) >= 0

    # Hrefs constraints
    model += pulp.lpSum((hrefs[i] - avg_hrefs) * x[i] for i in range(N)) >= 0

    # Other Indicator constraints
    model += pulp.lpSum(other_indicator[i] * x[i] for i in range(N)) >= other_lower_bound

    # Choose solver
    if solver_name == "PULP_CBC_CMD":
        solver = pulp.PULP_CBC_CMD(msg=verbose)
    else:
        raise ValueError(f"Unsupported Solver:{solver_name}")
    
    # Solve
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        print(f"Warning: Solver status = {status}")
    
    # Extract solution
    selected_indices = [
        i for i, var in x.items() if var.value() == 1
    ]
    return selected_indices, status