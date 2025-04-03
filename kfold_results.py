"""
Analyze the results of the k-fold cross-validation run
"""
import pandas as pd


def find_best(train_stats_file: str) -> None:
    """
    Finds the best parameters based on r2 score and MAE from the given file and prints it.
    :param train_stats_file: The file to analyze.
    :returns: None.
    """
    df = pd.read_csv(train_stats_file)
    df.columns = df.columns.str.strip()
    df = df.apply(pd.to_numeric)
    df = df.groupby(["hidden_size", "num_layers"]).mean().reset_index()
    df = df.drop("fold", axis=1).reset_index()
    max_r2 = df.loc[df["r2"].idxmax()]
    min_mae = df.loc[df["mae"].idxmin()]
    print(f"Max R²:\n"
          f"{max_r2}\n\n")
    print(f"Min MAE:\n"
          f"{min_mae}")


if __name__ == '__main__':
    print("No weather data:")
    find_best("training_stats_no_weather.csv")
    print("-" * 30)
    print("Weather data:")
    find_best("training_stats_weather.csv")
