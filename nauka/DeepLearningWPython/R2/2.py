### 1

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np

### 2

import matplotlib.pyplot as plt

### 3

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

### 1 - Pobranie danych

def load_housing():
    tar = Path("datasets/housing.tgz")
    if not tar.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tar)
        with tarfile.open(tar) as h_tar:
            h_tar.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing()

### 2 - Wizualizacja danych

# print(housing.head())
# print(housing.info())
# print(housing.describe())
# housing.hist(bins=50, figsize=(16,12))
# plt.show()

### 3 - Podział na dane testowe i robocze, modyfikacja danych

# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

### 2

housing["income_cat"] = pd.cut(housing["median_income"], bins= [0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
# housing["income_cat"].value_counts().sort_index().plot.bar(rot=8, grid=True)
# plt.xlabel("Income category")
# plt.ylabel("Number of districts")
# plt.show()

### 3

# splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
# strat_splits = []
# for train_index, test_index in splitter.split(housing, housing["income_cat"]):
#     strat_train_set_n = housing.iloc[train_index]
#     strat_test_set_n = housing.iloc[test_index]
#     strat_splits.append([strat_train_set_n, strat_test_set_n])

# strat_train_set, strat_test_set = strat_splits[0]

strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
# print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

### 2

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2, s= housing["population"]/100, label="population", c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False)
plt.show()