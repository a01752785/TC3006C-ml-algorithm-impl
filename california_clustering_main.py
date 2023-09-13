import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from k_means import Cluster, KMeans, PandasDataFrameAdapter

if __name__ == "__main__":
    california: pd.DataFrame = pd.read_csv("california_housing.csv")

    # Cluster using longitude, latitude and housing_median_age
    columns_to_keep: list[str] = ["longitude", "latitude", "median_income"]
    columns_to_drop: list[str] = [
        col for col in california.columns if col not in columns_to_keep
    ]
    california = california.drop(columns=columns_to_drop)

    # Scale data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(california)
    california = pd.DataFrame(normalized_data, columns=columns_to_keep)

    california_2 = california.copy(deep=True)

    # Scikit-learn implementation
    sklearn_k_means = cluster.KMeans(n_clusters=3)
    california["cluster"] = sklearn_k_means.fit_predict(california)
    california["source"] = "sklearn"

    # Own implementation
    adapter = PandasDataFrameAdapter(california_2)
    k_means = KMeans(data=adapter.points(), k=3)
    k_means.train(epochs=10)
    for cluster in k_means.clusters():
        print(cluster.center())
    california_2["cluster"] = k_means.list_of_cluster_nums()
    california_2["source"] = "own impl"

    # Data visualization
    california_combined: pd.DataFrame = pd.concat([california, california_2])
    california_combined["cluster"] = california_combined["cluster"].astype(
        "category"
    )
    sns.relplot(x="longitude", y="latitude",
                hue="cluster", data=california_combined, col="source")
    plt.show()
    plt.plot(k_means.sse_epoch_list())
    plt.title("Error over time. Source: Own impl")
    plt.xlabel("Epochs")
    plt.ylabel("Sum of squared error")
    plt.show()
