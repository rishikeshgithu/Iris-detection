import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Define columns
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

# Load the data
df = pd.read_csv('iris.data', names=columns)

# Some basic statistical analysis about the data
stats = df.describe()
print("Basic Statistical Analysis:")
print(stats)

# Visualize the whole dataset
sns.pairplot(df, hue='Class_labels')
plt.show()

# Separate features and target
data = df.values
X = data[:, 0:4]
Y = data[:, 4]

# Calculate average of each feature for all classes
Y_Data = np.array([np.average(X[:, i][Y == j].astype('float32')) for i in range(X.shape[1]) for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns) - 1)
width = 0.25

# Plot the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label='Setosa')
plt.bar(X_axis + width, Y_Data_reshaped[1], width, label='Versicolour')
plt.bar(X_axis + width * 2, Y_Data_reshaped[2], width, label='Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3, 1))
plt.show()

# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Support vector machine algorithm
svm = SVC()
svm.fit(X_train, y_train)

# Predict from the test dataset
predictions = svm.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# A detailed classification report
report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# New data for prediction
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])

# Prediction of the species from the input vector
prediction = svm.predict(X_new)
print("Prediction of Species:")
print(prediction)

# Save the model
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svm, f)

# Load the model
with open('SVM.pickle', 'rb') as f:
    loaded_model = pickle.load(f)
loaded_model.predict(X_new)
