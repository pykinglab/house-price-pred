import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

#load the dataset
df = pd.read_csv("data/house_prices.csv")

#Input
X = df[["square_feet"]]

#Output
y = df['price']

#create linear model
model = LinearRegression()

#Train the Model
model.fit(X,y)

#Display learned values
print("Slope(m)      = ",model.coef_[0])
print("Intercept(c)  = ",model.intercept_)

#Test prediction
square_feet = [[1500]]

prediction = model.predict(square_feet)

print("Predicted Price is ",prediction[0])

#save model
joblib.dump(model,"model/house_price_model.pkl")

print("Model saved successfully!!")