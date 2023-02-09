#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

sns.set()
plt.style.use('fivethirtyeight')

data = pd.read_csv('C:\MSFT\Dataset\MSFT.csv')
data.head()


# In[2]:


plt.figure(figsize = (10, 4))
plt.title("Microsoft Stock Prices Visualization")
plt.xlabel("Date")
plt.ylabel("Closing Figure")

plt.plot(data["Close"])
plt.show()


# In[3]:


a = data[["Open", "High", "Low"]]
b = data["Close"]

a = a.to_numpy()
b = b.to_numpy()
b = b.reshape(-1, 1)


# In[4]:


a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.2, random_state = 42)

model = DecisionTreeRegressor()
model.fit(a_train, b_train)

prediction = model.predict(a_test)
data = pd.DataFrame(data = {"Predicted Rate" : prediction})
data.head()

