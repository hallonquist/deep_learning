import pandas as pd
from sklearn.datasets import load_wine

wine_data = load_wine()

print(wine_data.data.shape)

# Convert data to pandas dataframe
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Add the target label
wine_df["target"] = wine_data.target

print(wine_df.count())

print(wine_df["target"].value_counts())

# Take a preview
#print(wine_df.head())
#print(wine_df.tail())

print(wine_df.describe())



# ----------------------------------------------------
# 1. Data Scaling:

from sklearn.preprocessing import StandardScaler

# Split data into features and label 
X = wine_df[wine_data.feature_names].copy()
y = wine_df["target"].copy() 

# Instantiate scaler and fit on features
scaler = StandardScaler()
scaler.fit(X)

# Transform features
X_scaled = scaler.transform(X.values)

# ----------------------------------------------------
# 2. Data Splitting:

from sklearn.model_selection import train_test_split

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, train_size=.7, random_state=25)

# Check the splits are correct
print(f"Train size: {round(len(X_train_scaled) / len(X) * 100)}% \n\
Test size: {round(len(X_test_scaled) / len(X) * 100)}%")

#print(type(X_train_scaled))
#print(X_train_scaled.shape)

# ----------------------------------------------------
# 3. Classifier Training:

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Instnatiating the models 
logistic_regression = LogisticRegression()
svm = SVC()
tree = DecisionTreeClassifier()

# Training the models 
logistic_regression.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
tree.fit(X_train_scaled, y_train)

# Making predictions with each model
log_reg_preds = logistic_regression.predict(X_test_scaled)
svm_preds = svm.predict(X_test_scaled)
tree_preds = tree.predict(X_test_scaled)


# ----------------------------------------------------
# 4. Prediction:

from sklearn.metrics import classification_report

# Store model predictions in a dictionary
# this makes it's easier to iterate through each model
# and print the results. 
model_preds = {
    "Logistic Regression": log_reg_preds,
    "Support Vector Machine": svm_preds,
    "Decision Tree": tree_preds
}

for model, preds in model_preds.items():
    print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")