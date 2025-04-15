"""
Analyze the results of the k-fold cross-validation run
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_heatmap(df: pd.DataFrame, metric: str, title: str, filename: str) -> None:
    """
    Plots a heatmap for the given dataframe.
    :param df: The DataFrame to plot.
    :param metric: The metric, either mae or r2, to plot.
    :param title: The name of the training run.
    :param filename: The name of the file to save the plot to.
    :returns: None.
    """
    pivot = df.pivot(index='hidden_size', columns='num_layers', values=metric)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"{title} - {metric}")
    plt.xlabel("num_layers")
    plt.ylabel("hidden_size")
    plt.tight_layout()
    plt.savefig(filename)


def plot_results() -> None:
    """
    Plots the results of all the training runs.
    :returns: None.
    """
    # Load the results
    no_weather = pd.read_csv("./training_stats_no_weather.csv")
    weather = pd.read_csv("./training_stats_weather.csv")
    smooth_weather = pd.read_csv("./training_stats_smooth_weather.csv")
    grade = pd.read_csv("./training_stats_smooth_weather_grade.csv")
    fuel_only = pd.read_csv("./training_stats_smooth_weather_grade_fuel_only.csv")
    no_bearing = pd.read_csv("./training_stats_smooth_weather_grade_fuel_only_no_bearing.csv")

    # Strip column names and ensure numeric data
    for df in [no_weather, weather, smooth_weather, grade, fuel_only, no_bearing]:
        df.columns = df.columns.str.strip()
        df[df.columns] = df.apply(pd.to_numeric, errors='coerce')

    def process(x: pd.DataFrame) -> pd.DataFrame:
        """
        Group by hidden_size and num_layers, averages metrics.
        :param x: The DataFrame to clean up.
        :return: Grouped DataFrame.
        """
        grouped = x.groupby(["hidden_size", "num_layers"]).mean().reset_index()
        return grouped

    no_weather = process(no_weather)
    weather = process(weather)
    smooth_weather = process(smooth_weather)
    grade = process(grade)
    fuel_only = process(fuel_only)
    no_bearing = process(no_bearing)

    # Generate plots
    plot_heatmap(
        no_weather,
        'r2',
        "No Weather Data",
        "figures/no_weather_r2.png"
    )
    plot_heatmap(
        weather,
        'r2',
        "Weather Data",
        "figures/weather_r2.png"
    )
    plot_heatmap(
        smooth_weather,
        'r2',
        "Smoothed Weather Data",
        "figures/smoothed_weather_r2.png"
    )
    plot_heatmap(
        grade,
        'r2',
        "Smoothed Weather Data + Grade",
        "figures/smoothed_weather_grade_r2.png"
    )
    plot_heatmap(
        fuel_only,
        'r2',
        "Smoothed Weather Data + Grade; Fuel Only",
        "figures/smoothed_weather_grade_fuel_only_r2.png"
    )
    plot_heatmap(
        no_bearing,
        'r2',
        "Smoothed weather data + Grade; Fuel only; No Bearing",
        "figures/no_bearing_r2.png"
    )

    plot_heatmap(
        no_weather,
        'mae',
        "No Weather Data",
        "figures/no_weather_mae.png"
    )
    plot_heatmap(
        weather,
        'mae',
        "Weather Data",
        "figures/weather_mae.png"
    )
    plot_heatmap(
        smooth_weather,
        'mae',
        "Smoothed Weather Data",
        "figures/smoothed_weather_mae.png"
    )
    plot_heatmap(
        grade,
        'mae',
        "Smoothed Weather Data + Grade",
        "figures/smoothed_weather_grade_mae.png"
    )
    plot_heatmap(
        fuel_only,
        'mae',
        "Smoothed Weather Data + Grade; Fuel Only",
        "figures/smoothed_weather_grade_fuel_only_mae.png"
    )
    plot_heatmap(
        no_bearing,
        'mae',
        "Smoothed weather data + Grade; Fuel only; No Bearing",
        "figures/no_bearing_mae.png"
    )


if __name__ == '__main__':
    print("No weather data:")
    find_best("training_stats_no_weather.csv")
    print("-" * 30)
    print("Weather data:")
    find_best("training_stats_weather.csv")
    print("-" * 30)
    print("Smoothed weather data:")
    find_best("training_stats_smooth_weather.csv")
    print("-" * 30)
    print("Smoothed weather data + Grade:")
    find_best("training_stats_smooth_weather_grade.csv")
    print("-" * 30)
    print("Smoothed weather data + Grade; Fuel only:")
    find_best("training_stats_smooth_weather_grade_fuel_only.csv")
    print("-" * 30)
    print("Smoothed weather data + Grade; Fuel only; No Bearing:")
    find_best("training_stats_smooth_weather_grade_fuel_only_no_bearing.csv")

    # Plot the results of the k-fold cross-validation
    plot_results()
