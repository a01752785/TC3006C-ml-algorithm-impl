import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from k_means import KMeans, PandasDataFrameAdapter

if __name__ == "__main__":
    random.seed(31)

    iris: pd.DataFrame = pd.read_csv("iris.data", header=None)
    col_names = [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
    ]
    iris.columns = col_names

    # Perform K-Means on two variables to visualize functionality
    columns_to_keep: list[str] = ["petal_width", "petal_length"]
    columns_to_drop: list[str] = [
        col for col in iris.columns if col not in columns_to_keep
    ]
    iris = iris.drop(columns=columns_to_drop)
    iris_2 = iris.copy(deep=True)

    # Scikit-learn implementation
    sklearn_k_means = cluster.KMeans(n_clusters=3)
    iris["cluster"] = sklearn_k_means.fit_predict(iris)
    iris["source"] = "sklearn"

    # Own implementation
    adapter = PandasDataFrameAdapter(iris_2)
    k_means = KMeans(data=adapter.points(), k=3)
    k_means.train(epochs=10)
    iris_2["cluster"] = k_means.list_of_cluster_nums()
    iris_2["source"] = "own impl"

    # Data visualization
    iris_combined: pd.DataFrame = pd.concat([iris, iris_2])
    iris_combined["cluster"] = iris["cluster"].astype("category")
    sns.relplot(x="petal_length", y="petal_width",
                hue="cluster", data=iris_combined, col="source")
    plt.show()
    plt.plot(k_means.sse_epoch_list())
    plt.title("Error over time. Source: Own impl")
    plt.xlabel("Epochs")
    plt.ylabel("Sum of squared error")
    plt.show()
