# Databricks notebook source
df = spark.read.table("hive_metastore.default.Superstore")

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

dd = df.toPandas()

# COMMAND ----------

f = dd.loc[dd["Category"] == "Furniture"]
f["Order Date"] = pd.to_datetime(f["Order Date"])
f = f.set_index("Order Date")

# COMMAND ----------

f.columns

# COMMAND ----------

f = f.Quantity.resample('M').sum()

# COMMAND ----------

f = pd.DataFrame(f)

# COMMAND ----------

"""cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode',
       'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State',
       'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category',
       'Product Name', 'Sales', 'Discount', 'Profit']"""

# COMMAND ----------

f["Season"] = f["Quantity"] - f["Quantity"].shift(12)

# COMMAND ----------

f.columns

# COMMAND ----------

from statsmodels.tsa.seasonal import seasonal_decompose
decom = seasonal_decompose(f["Season"].dropna(), model = "pseudoadditive", period = 12)
decom.plot()
plt.show()

# COMMAND ----------

# To Check stationary or not by using Augmented Dickey fuller test

from statsmodels.tsa.stattools import adfuller
adfuller(f["Quantity"])
#test staistics, pvalue, lags, no. of observations

# COMMAND ----------

import itertools

p = d = q = range(0,2)

pdq = list(itertools.product(p,d,q))  # best value for each p or d or q
seasonal_pdq = [(x[0],x[1],x[2], 12) for x in pdq]

print("Few Parameter combinations are :")
print('{} x {}'. format(pdq[0], seasonal_pdq[0]))
print('{} x {}'. format(pdq[1], seasonal_pdq[1]))
print('{} x {}'. format(pdq[2], seasonal_pdq[2]))

#So we can ingest with the parameter combinations moving forward

# COMMAND ----------

for param in pdq:
    for sea in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(f["Quantity"],order = param, seasonal_order = sea,
                                             enforce_stationarity = False,
                                             enforce_invertibility = False)
            results = model.fit()
            print('Arima{}x{} - AIC:{}'.format(param, sea,results.aic))
        except:
            continue
        #Arima(0, 1, 1)x(1, 1, 1, 12) - AIC:208.9779784268398
        # Arima(0, 1, 1)x(0, 1, 1, 12) - AIC:209.40128273297802

# COMMAND ----------

# Arima(0, 1, 1)x(0, 1, 1, 12) - AIC:70.2487704453346
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
model = sm.tsa.statespace.SARIMAX(f["Quantity"],order = (0,1,1), seasonal_order = (1,1,1,12))
result  = model.fit()

# COMMAND ----------

result.summary()

# COMMAND ----------

f["Demand_forecast"] = result.predict(start = 30 , end  = 48 , dynamic = True)
f[["Quantity","Demand_forecast"]].plot()

# COMMAND ----------

#Demand Forecasting

from pandas.tseries.offsets import DateOffset
future_date = [f.index[-1] + DateOffset(months = x) for x in range(0,60)]

# COMMAND ----------

future_date

# COMMAND ----------

future_date = pd.DataFrame(index=future_date[1:],columns=f.columns)
forecast_demand = pd.concat([f,future_date])

# COMMAND ----------

forecast_demand

# COMMAND ----------

forecast_demand["Demand_forecast"] = result.predict(start = 30, end = 80 , dynamic = True)
forecast_demand[["Quantity", "Demand_forecast"]].plot()

# COMMAND ----------

forecast_demand["Demand_forecast"] = result.predict(start = 13, end = 106, dynamic = True)
forecast_demand.head(60)

# COMMAND ----------

forecast_demand = forecast_demand.drop(columns = "Season", axis = 1)

# COMMAND ----------

forecast_demand.tail(25)

# COMMAND ----------

forecast_demand = forecast_demand.reset_index().rename(columns = {forecast_demand.index.name: "Order Date"})

# COMMAND ----------

demand_df2 = spark.createDataFrame(forecast_demand)

# COMMAND ----------

demand_df2.write.saveAsTable("Demand_Store")

# COMMAND ----------


