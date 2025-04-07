import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#If any of this libraries is missing from your computer. Please install them using pip.

filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename)

# analyze data
print(df.head())
print(df.describe())
print(df.isnull().sum())

# visualize
plt.figure(figsize=(10,6)) # Written with the help of AI
sns.histplot(df['ARR_DELAY'], bins=30, kde=True) # Written with the help of AI
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='AIRLINES', y='ARR_DELAY', data=df)
plt.title('Arrival Delays by Airlines')
plt.xlabel('Airlines')
plt.ylabel('Arrival Delay(minutes)')
plt.xticks(rotation=90)
plt.show()


predictors = ['DEP_DELAY', 'DISTANCE', 'AIRLINES', 'ORIGIN']
x = df[predictors]
y = df['ARR_DELAY']

x = pd.get_dummies(x, drop_first=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # Written with the help of AI

x_train = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

x_test = sm.add_constant(x_test)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred) # Written with the help of AI
r2 = r2_score(y_test, y_pred) # Written with the help of AI
print(f"Mean Squared Error: {mse}") # Written with the help of AI
print(f'R-squared: {r2}') # Written with the help of AI


# Chat-GPT4.(2025/03/23). "Walk me through making a code to do data descriptive and predictive analytics." Generated using OpenAi Chat-GPT. https://chat.openai.com/ 
