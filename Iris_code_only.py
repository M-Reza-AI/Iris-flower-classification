from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np



api = KaggleApi()
api.authenticate()

# I didn't do the downloading part again because I already have the dataset in my directory
Iris_dataset = pd.read_csv('iris.csv')

y = Iris_dataset['species']
x = Iris_dataset.drop('species', axis=1)

x_train , x_val , train_y , val_y = train_test_split(x, y, test_size=0.2, random_state=0)

x_train = StandardScaler().fit_transform(x_train)
x_val = StandardScaler().fit_transform(x_val)

model = LogisticRegression(random_state=0 , solver='lbfgs', multi_class='auto')
model.fit(x_train, train_y)

y_pred = model.predict(x_val)

probs_y = model.predict_proba(x_val)
probs_y = np.round(probs_y, 2)
res = "{:<10} | {:<10} | {:<10} | {:<13} | {:<5}".format("train_y", "y_pred", "Setosa(%)", "versicolor(%)", "virginica(%)\n")
res += "-"*65+"\n"
res += "\n".join("{:<10} | {:<10} | {:<10} | {:<13} | {:<10}".format(x, y, a, b, c) for x, y, a, b, c in zip(train_y, y_pred, probs_y[:,0], probs_y[:,1], probs_y[:,2]))
res += "\n"+"-"*65+"\n"
print(res)




