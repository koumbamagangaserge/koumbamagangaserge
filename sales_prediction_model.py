# AI Decision Support - Sales Prediction Model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset (simulated business data)
data = {
    'Marketing_Spend': [1000, 1500, 2000, 2500, 3000],
    'Sales': [200, 300, 400, 500, 600]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Marketing_Spend']]
y = df['Sales']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[3500]])

print("Predicted Sales for $3500 marketing spend:", prediction[0])
