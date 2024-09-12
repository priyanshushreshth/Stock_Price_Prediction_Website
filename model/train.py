import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load and prepare your data
data = pd.read_csv('data/processed.csv')
X = data.drop('target', axis=1)
y = data['target']

# Train the model
model = RandomForestRegressor()
model.fit(X, y)

# Save the trained model
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)
