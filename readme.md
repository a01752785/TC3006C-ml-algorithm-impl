# Implementation of the K Means Algorithm

This repository contains an implementation of the K Means Algorithm in the ```k_means.py``` file. This file consists of the class ```KMeans()```, which is the interface for the implementation.

We have tested the implementation with two different datasets, the [Iris Species Dataset](https://www.kaggle.com/datasets/uciml/iris) and the [California Housing Prices Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

This is an unsupervised machine learning algorithm, meaning that we don't use the labeled data.

For the first dataset, we test the functionality of the class by re-labeling the training data as the dataset contains that feature. We divide the dataset into training and test to see how well the model performs in predictions. Those are printed into the console, as well as the accuracy of the predictions.

For the second dataset, the features we are clustering on are unlabeled; we are testing how well the algorithm performs on assigning labels to the data.

For both datasets, two plots are created:

1. Plot comparing the output from the algorithm implemented in scikit-learn with the output from this implementation.

2. Plot showing the Sum of Squared Error, showing how the model is learning from the data over each iteration.

## Running the binaries

In order to execute both datasets' analysis, it is needed the installation of the libraries that the code uses.

Execute the following commands:
```
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
```

Then, execute the files for each dataset with the following commands:

```
python iris_clustering_main.py
python california_clustering_main.py
```

## Submission info

* Author: David Damian Galan

* ID: A01757285

* Submission date: Aug 28, 2023