import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import cluster


if __name__ == "__main__":
    california = pd.read_csv("california_housing.csv")
    print(california.head())

    # Cluster using longitude, latitude and housing_median_age
    columns_to_keep = ["longitude", "latitude", "median_income"]
    columns_to_drop = [
        col for col in california.columns if col not in columns_to_keep
    ]
    california = california.drop(columns=columns_to_drop)
    print(california.head())
    k_means = cluster.KMeans(n_clusters=6)
    california["cluster"] = k_means.fit_predict(california)
    california["cluster"] = california["cluster"].astype("category")
    print(california.head())
    sns.relplot(x="longitude", y="latitude", hue="cluster", data=california)
    plt.show()
