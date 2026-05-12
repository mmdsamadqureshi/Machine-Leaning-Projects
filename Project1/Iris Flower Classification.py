# Import required libraries/dependencies like sklearn(Scikit-learn)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Fetch the iris flower dataset
data = load_iris()
features, labels = data.data, data.target
# Divide into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(
features, labels, test_size=0.3, random_state=42
)
# Build and train the classifier
classifier = DecisionTreeClassifier()
classifier.fit(features_train, labels_train)
# Make predictions and calculate accuracy
predicted_labels = classifier.predict(features_test)
score = accuracy_score(labels_test, predicted_labels)
print(f"Model Accuracy: {score * 100:.1f}%")
