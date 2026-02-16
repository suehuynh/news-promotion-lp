import pandas as pd
import matplotlib.pyplot as plt


def frontier_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sensitivity results to construct Pareto frontier data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sensitivity experiment results.

    Returns
    -------
    df_frontier : pd.DataFrame
        Aggregated statistics by minimum topic constraint.
    """

    df_frontier = (
        df
        .groupby("min_topics")
        .agg(
            mean_lp_shares=("lp_shares", "mean"),
            std_lp_shares=("lp_shares", "std"),
            mean_tradeoff=("shares_trade_off", "mean"),
            mean_realized_diversity=("realized_diversity", "mean")
        )
        .reset_index()
        .sort_values("min_topics")
    )

    return df_frontier


def frontier_viz(df_frontier: pd.DataFrame):
    """
    Plot Pareto frontier: Engagement vs Diversity.
    """

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        df_frontier["mean_realized_diversity"],
        df_frontier["mean_lp_shares"],
        yerr=df_frontier["std_lp_shares"],
        marker="o",
        capsize=4
    )

    plt.xlabel("Number of Topics Covered")
    plt.ylabel("Predicted Shares")
    plt.title("Pareto Frontier: Engagement vs Topic Diversity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def marginal_viz(df_frontier: pd.DataFrame):
    """
    Plot marginal opportunity cost of diversity constraints.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        df_frontier["min_topics"],
        df_frontier["mean_tradeoff"],
        marker="o"
    )

    plt.xlabel("Minimum Topics Required")
    plt.ylabel("Shares Lost vs Top-10 Baseline")
    plt.title("Marginal Cost of Diversity Constraints")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
