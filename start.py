"""Loading Dataset"""
import pandas as pd

df = pd.read_csv("ER Wait Time Dataset.csv")
print(df.shape)
df.head()


y = df["Total Wait Time (min)"]

X = df[
    ["Urgency Level",
     "Nurse-to-Patient Ratio",
     "Specialist Availability",
     "Facility Size (Beds)",
     "Time to Registration (min)",
     "Time to Triage (min)",
     "Time to Medical Professional (min)",
     "Day of Week",
     "Season",
     "Time of Day"]
]

# For encoding categorical columns
X = pd.get_dummies(X)

# Train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline Model
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

baseline = DummyRegressor(strategy="median")
baseline.fit(X_train, y_train)

pred_base = baseline.predict(X_test)

print("Baseline MAE:", mean_absolute_error(y_test, pred_base))
