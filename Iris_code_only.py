from Helper.AI_Helper import *
from sklearn.tree import DecisionTreeClassifier , plot_tree


Iris_dataset = Get_Dataset('himanshunakrani/iris-dataset', 'iris.csv')
y = Iris_dataset['species']
x = Iris_dataset.drop('species', axis=1)


x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0,stratify=y)
model = DecisionTreeClassifier(max_depth=4,min_samples_split=2,min_samples_leaf=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

plt.figure(figsize=(12, 8))
plot_tree(model,
          feature_names=x.columns,
          class_names=model.classes_,
          filled=True,
          rounded=True)
plt.title("Decision Tree Trained on Train/Test Split")
plt.show()