# packages import
from sklearn import datasets
import pandas as pd
from sklearn import linear_model
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

## for part1
boston = datasets.load_boston()
boston_y=boston.target
boston=pd.DataFrame(boston.data,columns=boston.feature_names)

## for part2
Wine = datasets.load_wine()
Wine = pd.DataFrame(Wine.data, columns = Wine.feature_names)
Iris = datasets.load_iris()
Iris = pd.DataFrame(Iris.data, columns = Iris.feature_names)

# Part 1
model = linear_model.LinearRegression()
model.fit(boston, boston_y)
model.intercept_
model.coef_
coef_table=pd.DataFrame(model.coef_,index=boston.columns)
print(np.abs(coef_table).sort_values(by=0,ascending=False))
# the best impactable factor is NOX and the least is AGE

# Part 2
def find_k(k,x):
    kmeans_model1 = KMeans(n_clusters=k, random_state=1).fit(x)
    return(kmeans_model1.inertia_)


iner_err1=list(map(lambda k: find_k(k,Wine),range(2,10)))
iner_err2=list(map(lambda k: find_k(k,Iris),range(2,10)))

k=[x for x in range(2,10)]
plt.plot(k,iner_err1)
plt.title("Scree like plot for Wine")

plt.plot(k,iner_err2)
plt.title("Scree like plot for Iris")
