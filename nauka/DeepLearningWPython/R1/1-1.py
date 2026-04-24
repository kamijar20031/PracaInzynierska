import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

x_new = [[37_655.2]]
print(f"Szacowanie satysfakcji życiowej dla Cypryjczyków (GDP = {x_new} USD/osobe):")

model = KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)
print(f"Model K najbliższych sąsiadów: {model.predict(x_new)}")

model = LinearRegression()
model.fit(x,y)
print(f"Model regresji liniowej: {model.predict(x_new)}")

lifesat.plot(kind="scatter", grid=True, x='GDP per capita (USD)', y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()
