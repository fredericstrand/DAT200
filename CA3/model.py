import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train_path = "assets/train.csv"
test_path = "assets/test.csv"
output_path = "assets/sample_submission.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_data["Ripeness2"] = train_data["Ripeness"] ** 2
test_data["Ripeness2"] = test_data["Ripeness"] ** 2

train_data["Sweetness2"] = train_data["Sweetness"] ** 2
test_data["Sweetness2"] = test_data["Sweetness"] ** 2

train_data["Acidity2"] = train_data["Acidity"] ** 2
test_data["Acidity2"] = test_data["Acidity"] ** 2

train_data["Sweetness_Acidity"] = train_data["Sweetness"] * train_data["Acidity"]
test_data["Sweetness_Acidity"] = test_data["Sweetness"] * test_data["Acidity"]

drop_cols = ["Quality", "Banana Density", "Peel Thickness"]
X_train = train_data.drop(columns=drop_cols)
y_train = train_data["Quality"]

X_test = test_data.drop(columns=["Banana Density", "Peel Thickness"])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(C=10, kernel="rbf", gamma="scale")
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

submission = pd.DataFrame({"ID": range(len(predictions)), "Quality": predictions})
submission.to_csv(output_path, index=False)
print(f"Prediksjoner lagret i {output_path}")
