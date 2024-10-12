import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, f1_score
from imblearn.ensemble import BalancedRandomForestClassifier 
from sklearn.feature_selection import SelectFromModel

# Download the dataset
url = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
response = requests.get(url)

# Open the dataset
with ZipFile(io.BytesIO(response.content)) as other_zip:
    with other_zip.open('bank.zip') as inner_zip_file:
        with ZipFile(io.BytesIO(inner_zip_file.read())) as inner_zip:
            with inner_zip.open('bank-full.csv') as csv_file:
                df = pd.read_csv(csv_file, sep=';')

#print(df.head())
# Initial EDA
print(df.isnull().sum())

# Drop columns 'day' and 'month'
df = df.drop(columns=['day', 'month'])

# Loop One-Hot Encoding for categorical columns
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']

encoder = OneHotEncoder(drop='first', sparse_output=False)
for column in categorical_cols:
    encoded_cols = encoder.fit_transform(df[[column]])
    encoded_df = pd.DataFrame(encoded_cols, columns=[f"{column}_{cat}" for cat in encoder.categories_[0][1:]])
    df = pd.concat([df.drop(columns=[column]), encoded_df], axis=1)

# Seperate features and target
X = df.drop(columns=['y'])  
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train the Random Forest Classifier with out any method to handle imbalance    
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=None)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
# print evaluation metrics
print(f"Classification Report:\n{classification_rep}")
print(f"ROC AUC Score: {roc_auc}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Reduce Imbalance with BalancedRandomForestClassifier, and etc
selector = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
selector.fit(X_train, y_train)  
model = SelectFromModel(selector, threshold='median', prefit=True)
selected_mask = model.get_support()
selected_features = X_train.columns[selected_mask]

X_train_selected = model.transform(X_train)
X_test_selected = model.transform(X_test)

# Visualize feature importance of selected features 
importances = selector.feature_importances_
selected_importances = importances[selected_mask]
indices = np.argsort(selected_importances)[::-1]
selected_features_sorted = [selected_features[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.bar(range(X_train_selected.shape[1]), selected_importances[indices])
plt.xticks(range(X_train_selected.shape[1]), selected_features_sorted, rotation=90)
plt.show()

# Define parameters for the BalancedRandomForestClassifier
n_estimators_options = [50, 100]
max_depth_options = [10, 20, 30]

best_f1_score = 0
best_accuracy = 0
best_params = {}
best_classification_report = ""
best_brf = None 

# Nested loop to iterate through hyperparameters
for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        brf = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        brf.fit(X_train_selected, y_train)
        
        # Make predictions on the test set
        y_pred = brf.predict(X_test_selected)
        
        # Calculate performance metrics for the test set
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # If current model has better F1-score, update best model details
        if f1 > best_f1_score or (f1 == best_f1_score and accuracy > best_accuracy):
            best_f1_score = f1
            best_accuracy = accuracy
            best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
            best_classification_report = classification_report(y_test, y_pred)
            best_brf = brf  # Store the best model

# Print the best model performance and hyperparameters
print(f"\nBest F1 Score: {best_f1_score:.4f}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Best Parameters: {best_params}")

# Print the classification report of the best model
print("\nClassification Report for the Best Model:\n")
print(best_classification_report)

# Check for overfitting 
y_train_pred = best_brf.predict(X_train_selected)

# Calculate metrics on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

print("\nTraining Set Performance:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"F1 Score: {train_f1:.4f}")

print("\nTest Set Performance:")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"F1 Score: {best_f1_score:.4f}")

# Simple overfitting check
if train_accuracy > best_accuracy + 0.05:
    print("\nOverfitting Detected: The model performs significantly better on the training set.")
else:
    print("\nNo significant overfitting detected.")