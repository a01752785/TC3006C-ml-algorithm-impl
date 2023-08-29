import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from sklearn import cluster
from sklearn.model_selection import train_test_split

from k_means import KMeans, PandasDataFrameAdapter

if __name__ == "__main__":
    random.seed(50)

    iris_ori: pd.DataFrame = pd.read_csv("iris.data", header=None)
    col_names = [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
    ]
    iris_ori.columns = col_names

    # Perform K-Means on two variables to visualize functionality
    columns_to_keep: list[str] = ["petal_width", "petal_length"]
    columns_to_drop: list[str] = [
        col for col in iris_ori.columns if col not in columns_to_keep
    ]
    features = iris_ori.drop(columns=columns_to_drop)
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        iris_ori["class"],
                                                        test_size=0.2,
                                                        random_state=42)
    iris = x_train.copy(deep=True)
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

    # Validation

    # Map cluster name to cluster id
    # The id with the most frequency is selected
    cluster_id_by_class_hist = {"Iris-setosa": [0, 0, 0],
                                "Iris-versicolor": [0, 0, 0],
                                "Iris-virginica": [0, 0, 0]}
    for index in x_train.index.values:
        class_name = iris_ori.loc[index]["class"]
        cluster_id_by_class_hist[class_name][iris_2.loc[index]["cluster"]] += 1

    cluster_id_by_class = {"Iris-setosa": 0,
                           "Iris-versicolor": 0,
                           "Iris-virginica": 0}
    class_by_cluster_id = ["", "", ""]
    for key in cluster_id_by_class_hist:
        max_id = -1
        max = 0
        for i in range(0, 3):
            if cluster_id_by_class_hist[key][i] > max:
                max = cluster_id_by_class_hist[key][i]
                max_id = i
        cluster_id_by_class[key] = max_id
        class_by_cluster_id[max_id] = key
    print(cluster_id_by_class)

    test_adapter = PandasDataFrameAdapter(features)
    assigned_correctly = 0
    for index in x_test.index.values:
        pred_cluster_id = k_means.predict(test_adapter.point_by_index(index))
        actual_cluster_id = cluster_id_by_class[iris_ori.iloc[index]["class"]]
        print(index)
        print(f"Prediction: {class_by_cluster_id[pred_cluster_id]}")
        print(f"Actual: {class_by_cluster_id[actual_cluster_id]}")
        if (pred_cluster_id == actual_cluster_id):
            assigned_correctly += 1
    print(f"Correct prediction ratio: {assigned_correctly / len(x_test)}")

    # Data visualization
    iris_combined: pd.DataFrame = pd.concat([iris, iris_2])
    iris_combined["cluster"] = iris_combined["cluster"].astype("category")
    sns.relplot(x="petal_length", y="petal_width",
                hue="cluster", data=iris_combined, col="source")
    plt.show()
    plt.plot(k_means.sse_epoch_list())
    plt.title("Error over time. Source: Own impl")
    plt.xlabel("Epochs")
    plt.ylabel("Sum of squared error")
    plt.show()
