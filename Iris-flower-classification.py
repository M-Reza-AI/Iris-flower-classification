# the library we use to download datasets from kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# actual machine learning models are here
from sklearn.linear_model import LogisticRegression

# we use this to split the dataset
from sklearn.model_selection import train_test_split

# makes the values range between 0 and 1
from sklearn.preprocessing import StandardScaler

# we are going to access our data set through this library
import pandas as pd

# will use this later to output the predictions
import numpy as np

# will use this to plot the dataset values and predictions
import matplotlib.pyplot as plt
import seaborn as sns

# this will connect you to kaggle so you can download datasets from it
api = KaggleApi()
api.authenticate()

# you only need to do this once and then remove this line after the dataset is in the project directory

api.dataset_download_file('himanshunakrani/iris-dataset',
                          file_name='iris.csv')

# this is your dataset
Iris_dataset = pd.read_csv('iris.csv')

# this is the value we want to predict
y = Iris_dataset['species']

# after saving the value we want to predict we drop it from our dataset
x = Iris_dataset.drop('species', axis=1)

# split the training data into 4 variables
x_train , x_val , train_y , val_y = train_test_split(x, y, test_size=0.2, random_state=0)

# normalize all the values between 0 and 1 for all float values, in test and validation
x_train = StandardScaler().fit_transform(x_train)
x_val = StandardScaler().fit_transform(x_val)

# make the model (will talk more about it on later chapters , but for now, it will help with catagorical values)
model = LogisticRegression(random_state=0 , solver='lbfgs', multi_class='auto')

# give it the training data for it to start learning from it
model.fit(x_train, train_y)

# use the model to output predictions and see if it did well
y_pred = model.predict(x_val)

# Predict probabilities
probs_y = model.predict_proba(x_val)
### Print results in pretty way
probs_y = np.round(probs_y, 2)
res = "{:<10} | {:<10} | {:<10} | {:<13} | {:<5}".format("train_y", "y_pred", "Setosa(%)", "versicolor(%)", "virginica(%)\n")
res += "-"*65+"\n"
res += "\n".join("{:<10} | {:<10} | {:<10} | {:<13} | {:<10}".format(x, y, a, b, c) for x, y, a, b, c in zip(train_y, y_pred, probs_y[:,0], probs_y[:,1], probs_y[:,2]))
res += "\n"+"-"*65+"\n"
print(res)

# plot the data that you have , and see why the model made certian choices
Iris_dataset.plot(kind ="scatter", x = "sepal_length", y = "sepal_width", c = "petal_length")
plt.grid(True)

iris = sns.load_dataset("iris")
sns.set_style("whitegrid")
sns.FacetGrid(iris, hue = "species", height = 6).map(plt.scatter,
                                                        'sepal_length',
                                                             'sepal_width').add_legend()


plt.show()