from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
import pandas as pd
import random


class NDimensionalPoint:
    def __init__(self, data: Sequence[float]) -> None:
        self.dimensions_: Sequence[float] = data

    def __str__(self) -> str:
        return str(self.dimensions_)

    def to_list(self) -> list[float]:
        result = []
        for dimension in self.dimensions_:
            result.append(dimension)
        return result

    def num_dimensions(self) -> int:
        return len(self.dimensions_)

    def dimension_value(self, dimension_id: int) -> float:
        return self.dimensions_[dimension_id]

    def distance(self, other_point: NDimensionalPoint) -> float:
        cur_sum: float = 0
        for i in range(self.num_dimensions()):
            cur_sum += (self.dimension_value(i) -
                        other_point.dimension_value(i)) ** 2
        return sqrt(cur_sum)


class PandasDataFrameAdapter:
    def __init__(self, df: pd.DataFrame) -> None:
        self.column_names_: list[str] = []
        self.points_: list[NDimensionalPoint] = []
        for column in df.columns:
            self.column_names_.append(column)

        for row in range(len(df)):
            self.points_.append(NDimensionalPoint(df.loc[row]))

    def points(self) -> list[NDimensionalPoint]:
        return self.points_


class Cluster:
    def __init__(self, points: list[NDimensionalPoint],
                 center: NDimensionalPoint) -> None:
        self.points_: list[NDimensionalPoint] = points
        self.center_: NDimensionalPoint = center

    def __str__(self) -> str:
        return (
            "Center: "
            + str(self.center_)
            + "\n"
            + "Points: "
            + str([str(point) for point in self.points_])
            + "\n"
        )

    def recompute_center(self) -> None:
        # Not possible to recompute a center of a 0-point cluster
        if len(self.points_) == 0:
            return None

        acum_sum_per_dimension: list[float] = [
            0.0] * self.points_[0].num_dimensions()
        for i in range(len(self.points_)):
            for dimension_id in range(self.points_[i].num_dimensions()):
                acum_sum_per_dimension[dimension_id] += self.points_[
                    i].dimension_value(dimension_id)
        mean_per_dimension = list(
            map(
                lambda x: x / len(self.points_),
                acum_sum_per_dimension
            )
        )
        self.center_ = NDimensionalPoint(mean_per_dimension)

    def center(self) -> NDimensionalPoint:
        return self.center_

    def add_point(self, point: NDimensionalPoint) -> None:
        self.points_.append(point)

    def clear_points(self) -> None:
        self.points_.clear()

    def sse(self) -> float:
        cur_sse: float = 0
        for point in self.points_:
            cur_sse += self.center_.distance(point) ** 2
        return cur_sse


class KMeans:
    def __init__(self, data: Sequence[NDimensionalPoint], k: int) -> None:
        self.data_: Sequence[NDimensionalPoint] = data
        self.clusters_: Sequence[Cluster] = []
        self.cluster_num_for_point_: list[int] = [0] * len(data)
        # Init clusters
        for _ in range(k):
            self.clusters_.append(Cluster([], random.choice(self.data_)))

    def train(self, epochs: int) -> None:
        self.sse_: list[float] = []
        for _ in range(epochs):
            self.assign_points_to_clusters()
            self.recompute_means()
            self.recompute_sse()

    def assign_points_to_clusters(self) -> None:
        for i in range(len(self.clusters_)):
            self.clusters_[i].clear_points()
        for point_id, point in enumerate(self.data_):
            min_distance = self.clusters_[0].center().distance(point)
            min_distance_cluster_id = 0
            for i in range(len(self.clusters_)):
                if (self.clusters_[i].center().distance(point) < min_distance):
                    min_distance_cluster_id = i
            self.clusters_[min_distance_cluster_id].add_point(point)
            self.cluster_num_for_point_[point_id] = min_distance_cluster_id

    def recompute_sse(self) -> None:
        sse: float = 0.0
        for cluster in self.clusters_:
            sse += cluster.sse()
        self.sse_.append(sse)

    def recompute_means(self) -> None:
        for i in range(len(self.clusters_)):
            self.clusters_[i].recompute_center()

    def clusters(self) -> Sequence[Cluster]:
        return self.clusters_

    def list_of_cluster_nums(self) -> Sequence[int]:
        return self.cluster_num_for_point_

    def sse_epoch_list(self) -> list[float]:
        return self.sse_


def main() -> None:
    data = [NDimensionalPoint([1.01, 1.01]),
            NDimensionalPoint([1.05, 1.05]),
            NDimensionalPoint([1.11, 1.01]),
            NDimensionalPoint([1.21, 1.05]),
            NDimensionalPoint([4, 4]),
            NDimensionalPoint([5, 5])]
    k_means = KMeans(data=data, k=3)
    k_means.train(epochs=2)
    clusters = k_means.clusters()

    for cluster in clusters:
        print(cluster)


if __name__ == "__main__":
    main()
