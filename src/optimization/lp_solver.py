import pulp

def news_solver(
        shares,
        lifestyle_indicator,
        entertainment_indicator,
        bus_indicator,
        socmed_indicator,
        tech_indicator,
        world_indicator,
        other_indicator,
        K=10,
        lower_bound=0,
        solver_name="PULP_CBC_CMD",
        verbose=False
):
    """
    Build ILP model for news article selection that maximize shares and maintain topic diversity.

    Parameters
    ----------
    popularity : array-like, shape (N,)
        (Predicted) raw shares
    *_indicator : array-like, shape (N,)
        1 if article is classified as each in the category list, 0 otherwise
    K : int
        Number of articles to select
    
    Return
    ----------
    selected_indices: list
        List of selected optimal indices
    status: str
        Solver status
    """
    N = len(shares)
    
    # Model
    model = pulp.LpProblem("News Selection", pulp.LpMaximize)
    
    # Decision Variables
    x = pulp.LpVariable.dicts("select", range(N), cat="Binary")

    # Objective
    model += pulp.lpSum(
        (shares[i] * x[i] for i in range(N)))

    # Capacity constraints
    model += pulp.lpSum(x[i] for i in range(N)) == K

    # Topic constraints 
    model += pulp.lpSum(lifestyle_indicator[i] * x[i] for i in range(N)) >= lower_bound
    model += pulp.lpSum(entertainment_indicator[i] * x[i] for i in range(N)) >= lower_bound
    model += pulp.lpSum(bus_indicator[i] * x[i] for i in range(N)) >= lower_bound
    model += pulp.lpSum(socmed_indicator[i] * x[i] for i in range(N)) >= lower_bound
    model += pulp.lpSum(tech_indicator[i] * x[i] for i in range(N)) >= lower_bound
    model += pulp.lpSum(world_indicator[i] * x[i] for i in range(N)) >= lower_bound
    model += pulp.lpSum(other_indicator[i] * x[i] for i in range(N)) >= lower_bound

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
def extended_news_solver(
        shares,
        lifestyle_indicator,
        entertainment_indicator,
        bus_indicator,
        socmed_indicator,
        tech_indicator,
        world_indicator,
        other_indicator,
        K=10,
        diversity_lower_bound=1,
        solver_name="PULP_CBC_CMD",
        verbose=False
):
    """
    Build ILP model for news article selection that maximize shares and maintain topic diversity.

    Parameters
    ----------
    popularity : array-like, shape (N,)
        (Predicted) raw shares
    *_indicator : array-like, shape (N,)
        1 if article is classified as each in the category list, 0 otherwise
    K : int
        Number of articles to select
    
    Return
    ----------
    selected_indices: list
        List of selected optimal indices
    status: str
        Solver status
    """
    N = len(shares)
    categories = ["life","ent","bus","soc","tech","world","other"]
    
    # Model
    model = pulp.LpProblem("News Selection", pulp.LpMaximize)
    
    # Decision Variables
    x = pulp.LpVariable.dicts("select", range(N), cat="Binary")
    y = pulp.LpVariable.dicts("cat_used", categories, cat="Binary")

    # Objective
    model += pulp.lpSum(
        (shares[i] * x[i] for i in range(N)))

    # Capacity constraints
    model += pulp.lpSum(x[i] for i in range(N)) == K

    # Topic constraints 
    model += pulp.lpSum(lifestyle_indicator[i] * x[i] for i in range(N)) >= y["life"]
    model += pulp.lpSum(entertainment_indicator[i] * x[i] for i in range(N)) >= y["ent"]
    model += pulp.lpSum(bus_indicator[i] * x[i] for i in range(N)) >= y["bus"]
    model += pulp.lpSum(socmed_indicator[i] * x[i] for i in range(N)) >= y["soc"]
    model += pulp.lpSum(tech_indicator[i] * x[i] for i in range(N)) >= y["tech"]
    model += pulp.lpSum(world_indicator[i] * x[i] for i in range(N)) >= y["world"]
    model += pulp.lpSum(other_indicator[i] * x[i] for i in range(N)) >= y["other"]

    # Topic diversity constraints 
    model += pulp.lpSum(y[c] for c in categories) >= diversity_lower_bound
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
