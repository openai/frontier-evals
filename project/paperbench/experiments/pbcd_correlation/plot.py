import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_and_prepare_data(filepath):
    """
    Load and prepare the data for analysis.

    Args:
        filepath (str): Path to the CSV file containing results

    Returns:
        tuple: (code_only_average, normal_average) - Average scores for both conditions
    """
    # Load and sort data
    data = pd.read_csv(filepath)
    data = data.sort_values(["agent_id", "paper_id", "run_number"])

    # Split data by condition
    normal = data[data["agent_id"] == "aisi-basic-agent-iterative-o1"]
    code_only = data[data["agent_id"] == "aisi-basic-agent-iterative-o1-pb-code-only"]

    # Calculate averages by paper
    normal_average = normal.groupby(["paper_id"]).score.agg("mean")
    code_only_average = code_only.groupby(["paper_id"]).score.agg("mean")

    return code_only_average, normal_average


def perform_regression(code_only_scores, normal_scores):
    """
    Perform linear regression analysis on the data.

    Args:
        code_only_scores (np.ndarray): Scores for code-only condition
        normal_scores (np.ndarray): Scores for normal condition

    Returns:
        tuple: (model, r_squared, slope, intercept) - Regression results
    """
    X = code_only_scores.reshape(-1, 1)
    y = normal_scores

    model = LinearRegression()
    model.fit(X, y)

    r_squared = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_

    return model, r_squared, slope, intercept


def plot_regression(code_only_scores, normal_scores, model):
    """
    Create and save a scatter plot with regression line.

    Args:
        code_only_scores (np.ndarray): Scores for code-only condition
        normal_scores (np.ndarray): Scores for normal condition
        model (LinearRegression): Fitted regression model
    """
    x_range = np.linspace(0.2, 0.7, 50)
    y_fit = model.predict(x_range.reshape(-1, 1))

    plt.rcParams.update({"font.size": 7})

    fig, ax = plt.subplots(figsize=(6.75133 / 1.5, 2.75))
    ax.scatter(code_only_scores, normal_scores, s=10)
    ax.plot(x_range, y_fit, "r-")

    # Add line of best fit equation and R² value
    r_squared = model.score(code_only_scores.reshape(-1, 1), normal_scores)
    equation = f"y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}"
    r2_text = f"R² = {r_squared:.3f}"
    ax.text(
        0.05,
        0.95,
        f"{equation}\n{r2_text}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("PaperBench Code-Dev Performance")
    ax.set_ylabel("PaperBench Performance")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "experiments/pbcd_correlation/correlation_plot.pdf",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.01,
    )
    # plt.show()
    plt.close()


def main():
    """Main execution function."""
    # Load and prepare data
    results_path = "experiments/pbcd_correlation/results.csv"
    code_only_average, normal_average = load_and_prepare_data(results_path)

    # Convert to numpy arrays
    code_only_scores = code_only_average.to_numpy()
    normal_scores = normal_average.to_numpy()

    # Perform regression analysis
    model, r_squared, slope, intercept = perform_regression(code_only_scores, normal_scores)

    # Print results
    print("Regression Results:")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")

    # Create and save plot
    plot_regression(code_only_scores, normal_scores, model)
    print("\nPlot saved as 'experiments/pbcd_correlation/correlation_plot.pdf'")


if __name__ == "__main__":
    main()
