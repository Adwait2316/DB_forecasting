# Databricks notebook source
df = spark.read.table("hive_metastore.default.superstore")

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

ss = df.toPandas()

# COMMAND ----------

ss.shape

# COMMAND ----------

ss.display()

# COMMAND ----------

ss.describe()

# COMMAND ----------

ss.head(15)

# COMMAND ----------

ss.isnull().sum()

# COMMAND ----------

ss.Category.value_counts()

# COMMAND ----------

fu = ss.loc[ss["Category"] == "Furniture"]
te = ss.loc[ss["Category"]== "Technology"]
os = ss.loc[ss["Category"] == "Office Supplies"]

# COMMAND ----------

fu["Order Date"] = pd.to_datetime(fu["Order Date"])
fu = fu.set_index("Order Date")

# COMMAND ----------

"""for 4 years data - we have 889 Unique dates out of approx 1460 unique dates, So we will resample the data monthly for estimations  and request our client to provide complete data for better output"""
fu.index.nunique()

# COMMAND ----------

fu = fu["Sales"].resample("M").sum()


# COMMAND ----------

furn = pd.DataFrame(fu)

# COMMAND ----------

ss["Order Date"] = pd.to_datetime(ss["Order Date"])

# COMMAND ----------

ss["Year"] = ss["Order Date"].dt.year

# COMMAND ----------

ss["Month"] = ss["Order Date"].dt.month

# COMMAND ----------

ss.columns

# COMMAND ----------

"""df = ss.groupby(['Order Date','Ship Mode', 'Category','Sub-Category','Segment','Region','State','City','Customer ID','Customer Name','Product Name', 'Year','Month'])[ "Sales", "Discount","Quantity", "Profit"].sum().reset_index()
df = df.set_index("Order Date")"""

# COMMAND ----------

"""df = df.rename(columns = {'Ship Mode' : 'Ship_Mode', 'Sub-Category': 'Sub_Category', 'Customer ID': 'CustomerId', 'Customer Name': 'Customer_Name','Product Name':'Product_Name'})"""

# COMMAND ----------

#df[df.eq('}').any(1)]

# COMMAND ----------

top_cust = ss.groupby("Customer Name")["Sales"].sum().sort_values(ascending = False).round(2).head(20)
top_cust = top_cust.to_frame().reset_index()

# COMMAND ----------

plt.figure(figsize=(15,5))
plt.title("Top 20 Customers(2014 -2017)", fontsize = 18) 
plt.bar(top_cust["Customer Name"], top_cust["Sales"], linewidth = 0.5)
plt.xlabel("Customers", fontsize = 15)
plt.ylabel("Revenue", fontsize = 15)
plt.xticks(fontsize = 12, rotation = 90)
plt.yticks(fontsize = 12)
for k, v in top_cust["Sales"].items():
  plt.text(k,v-8000,'$' + str(v), rotation = 90);

# COMMAND ----------

top_cust_pr = ss.groupby("Customer Name")["Profit"].sum().sort_values(ascending = False).round(2).head(20)
top_cust_pr = top_cust_pr.to_frame().reset_index()

plt.figure(figsize=(15,5))
plt.title("Top 20 Customers(2014 -2017)", fontsize = 18) 
plt.bar(top_cust_pr["Customer Name"], top_cust_pr["Profit"], linewidth = 1)
plt.xlabel("Customers", fontsize = 15)
plt.ylabel("Profit", fontsize = 15)
plt.xticks(fontsize = 12, rotation = 90)
plt.yticks(fontsize = 12)
for k, v in top_cust_pr["Profit"].items():
  plt.text(k,v-1700,'$' + str(v), rotation = 90);

# COMMAND ----------

category = ss.groupby(["Category"])["Sales"].sum().sort_values(ascending = False)
category = category.to_frame().reset_index()
total_cat = category["Sales"].sum()
total_cat = str(int(total_cat))
total_cat = "$" + total_cat

# COMMAND ----------

ss.groupby("State")["Sales"].sum().sort_values(ascending = False).head(8).plot.barh();

# COMMAND ----------

ss.groupby("State")["Profit"].sum().sort_values(ascending = False).head(8).plot.barh();

# COMMAND ----------

ss_cf=ss[ss['State']=='California']
ss_ny=ss[ss['State']=='New York']

# COMMAND ----------

ss_cf_cust = pd.DataFrame(ss_cf.groupby("Customer Name")["Sales"].sum())
ss_cf_cust["Profit"] = pd.DataFrame(ss_cf.groupby("Customer Name")["Profit"].sum())

# COMMAND ----------

ss_cf_cust.sort_values(by = "Sales",ascending=False).head(5)

# COMMAND ----------

ss_ny_cust = pd.DataFrame(ss_ny.groupby("Customer Name")["Sales"].sum())
ss_ny_cust["Profit"] = pd.DataFrame(ss_ny.groupby("Customer Name")["Profit"].sum())
ss_ny_cust.sort_values(by = "Sales",ascending=False).head(5)

# COMMAND ----------

ss_ny_cust.describe().T

# COMMAND ----------

ss_ny_noTom=ss_ny_cust[ss_ny_cust.index != 'Tom Ashbrook']
ss_ny_noTom.describe().T
# Avg Store Avg Sale goes down by 749 to 717, also Avg profitability from 178 to 167
# NY is more profitable than California

# COMMAND ----------

ss1 = pd.DataFrame(ss.groupby("Customer Name")["Sales"].sum().sort_values(ascending = False))
ss1.quantile(0.7, interpolation = 'higher')

# COMMAND ----------

ss1[ss1["Sales"]>=3288].sum()/ss["Sales"].sum()
#Top 30 % occupy 61% 0f Sales

# COMMAND ----------

ss2 = pd.DataFrame(ss.groupby('Customer Name')['Profit'].sum().sort_values(ascending=False))
ss2.quantile(0.7, interpolation='higher')

# COMMAND ----------

ss2[ss2['Profit']>=463.269].sum()/ss['Profit'].sum()
# Top 30 % occupy 97% 0f profit

# COMMAND ----------

ss.sort_values(by = ["Sales"], ascending = False).head()

# COMMAND ----------

plt.rcParams["figure.figsize"] = (13,5) # width and height of figure is defined in inches
plt.rcParams['font.size'] = 12.0 # Font size is defined
plt.rcParams['font.weight'] = 6 # Font weight is defined
# we don't want to look at the percentage distribution in the pie chart. Instead, we want to look at the exact revenue generated by the categories.
def autopct_format(values): 
    def my_format(pct): 
        total = sum(values) 
        val = int(round(pct*total/100.0))
        return ' ${v:d}'.format(v=val)
    return my_format
colors = ['#BC243C','#FE840E','#C62168'] # Colors are defined for the pie chart
explode = (0.05,0.05,0.05)
fig1, ax1 = plt.subplots()
ax1.pie(category['Sales'], colors = colors, labels=category['Category'], autopct= autopct_format(category['Sales']), startangle=90,explode=explode)
centre_circle = plt.Circle((0,0),0.82,fc='white') # drawing a circle on the pie chart to make it look better 
fig = plt.gcf()
fig.gca().add_artist(centre_circle) # Add the circle on the pie chart
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal') 
# we can look the total revenue generated by all the categories at the center
label = ax1.annotate('Total Revenue \n'+str(total_cat),color = 'red', xy=(0, 0), fontsize=12, ha="center")
plt.tight_layout()
plt.show()

# COMMAND ----------

#Region 
region = ss.groupby("Region")["Sales"].sum().sort_values(ascending = False).round(2)
region = region.to_frame().reset_index()
total_reg = region["Sales"].sum()
total_reg = '$' + str(int(total_reg))


# COMMAND ----------

region["Region"]

# COMMAND ----------

plt.figure(figsize = (15,5))
plt.title("Region Wise Sales")
plt.bar(region["Region"], region["Sales"], linewidth = 1)
plt.xlabel("Zones", fontsize = 15)
plt.ylabel("Revenue", fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
for k, v in region["Sales"].items():
  plt.text(k,v-40000, '$'+str(v))


# COMMAND ----------

#ShipMode
shipMode = ss.groupby("Ship Mode")["Sales"].sum().sort_values(ascending = False).round(2)
shipMode = shipMode.to_frame().reset_index()

# COMMAND ----------

plt.figure(figsize = (15,5))
plt.title("Top ship Mode")
plt.bar(shipMode["Ship Mode"], shipMode["Sales"])
plt.xlabel("Ship Mode", fontsize = 15)
plt.ylabel("Revenue", fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
for k, v in shipMode["Sales"].items():
  plt.text(k,v-100000,'$'+str(v), ha = "center");

# COMMAND ----------

segment = ss.groupby("Segment")["Sales"].sum()
segment = segment.to_frame().reset_index()
total_seg = segment["Sales"].sum()
total_seg = '$' + str(int(total_seg))

# COMMAND ----------

plt.rcParams["figure.figsize"] = (13,5) # width and height of figure is defined in inches
plt.rcParams['font.size'] = 12.0 # Font size is defined
plt.rcParams['font.weight'] = 6 # Font weight is defined
colors = ['#BC243C','#FE840E','#C62168'] # Colors are defined for the pie chart
explode = (0.05,0.05,0.05)
fig1, ax1 = plt.subplots()
ax1.pie(segment['Sales'], colors = colors, labels=segment['Segment'], autopct = autopct_format(segment['Sales']),
startangle=90, explode=explode)
centre_circle = plt.Circle((0,0),0.85,fc='white') # Draw a circle on the pie chart
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal') 
label = ax1.annotate('Total Revenue \n'+str(total_seg),color = 'red', xy=(0, 0), fontsize=12, ha="center")
plt.tight_layout()
plt.show()

# COMMAND ----------

#Product 
product = ss.groupby("Product Name")["Sales"].sum().sort_values(ascending = False).round(2).head(8)
product = product.to_frame().reset_index()
total_prod = product["Sales"].sum().round(2)
total_prod = '$' + str(total_prod)

# COMMAND ----------

plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams["font.size"] = 12
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#55B4B0','#E15D44','#009B77','#B565A7']
explode = [.05,.05,.05]
fig1, ax1 = plt.subplots()
ax1.pie(product["Sales"], labels = product["Product Name"], colors = colors, autopct = autopct_format(product["Sales"]),startangle = 90)
fig = plt.gcf()
centre_circle = plt.Circle((0,0),0.80, fc = "white")
fig.gca().add_artist(centre_circle)
ax1.axis("equal")
label = ax1.annotate("Total Revenue \n" + str(total_prod), ha = "center", xy = (0,0), fontsize = 12, color = 'red')
plt.tight_layout()
plt.show()

# COMMAND ----------

plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams["font.size"] = 12
colors = ['#BC243C','#FE840E','#C62168']
explode = [0.05,0.05,.05]
fig1, ax1 = plt.subplots()
ax1.pie(segment["Sales"], colors = colors, labels = segment["Segment"], autopct = autopct_format(segment["Sales"]),startangle = 90)
centre_circle = plt.Circle((0,0),0.85,fc = "white")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax1.axis("equal")
label = ax1.annotate("Total Revenue \n"  +str(total_seg),color = "red", fontsize = 12,ha = "center", xy = (0,0))
plt.tight_layout()
plt.show()

# COMMAND ----------

df.groupby('Sub-Category')['Profit','Sales'].agg(['sum']).plot.bar()
plt.title('Total Profit and Sales per Sub-Category')
# plt.legend('Profit')
# plt.legend('Sales')
plt.show()

# COMMAND ----------

#Sub Category
sub_cat = ss.groupby(["Category","Sub-Category"])["Sales"].sum().sort_values(ascending = False).round(2).head(10)
sub_cat = sub_cat.to_frame().reset_index()
sub_cat1 = sub_cat.groupby("Category").sum()

# COMMAND ----------

"""plt.rcParams["figure.figsize"] = (15,10) # width and height of figure is defined in inches
fig, ax = plt.subplots()
ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle
width = 0.1
outer_colors = ['#FE840E','#009B77','#BC243C'] # Outer colors of the pie chart
inner_colors = ['Orangered','tomato','coral',"darkturquoise","mediumturquoise","paleturquoise","lightpink","pink","hotpink","deeppink"] # inner colors of the pie chart
pie = ax.pie(sub_cat1['Sales'], radius= 1, labels = sub_cat1["Category"], colors = outer_colors,wedgeprops=dict(edgecolor='w'))
pie2 = ax.pie(sub_cat['Sales'], radius= 1-width, labels=sub_cat['Sub-Category'],autopct= autopct_format(sub_cat['Sales']),labeldistance=0.7,colors=inner_colors, wedgeprops=dict(edgecolor='w'), pctdistance=0.53, rotatelabels =True)
# Rotate fractions
# [0] = wedges, [1] = labels, [2] = fractions
fraction_text_list = pie2[2]
for text in fraction_text_list: 
    text.set_rotation(315) # rotate the autopct values
centre_circle = plt.Circle((0,0),0.6,fc='white') # Draw a circle on the pie chart
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()"""

# COMMAND ----------

furn

# COMMAND ----------

furn["Seasonality"] = furn["Sales"] - furn["Sales"].shift(12)

# COMMAND ----------

#Decomposition - Taking the Mean Square Avg so as to normalize the missing Days and follow the trend, In Real Time missing dates can not be trusted

from statsmodels.tsa.seasonal import seasonal_decompose
decom = seasonal_decompose(furn["Seasonality"].dropna(), model = "pseudoadditive", period = 12)
decom.plot()
plt.show()

# COMMAND ----------

# To Check stationary or not by using Augmented Dickey fuller test

from statsmodels.tsa.stattools import adfuller
adfuller(furn["Sales"])
#test staistics, pvalue, lags, no. of observations

# COMMAND ----------

def adf_check(ts):
    result = adfuller(ts)
    print("Augmented Dickey Fuller Test - Stationary or Non- Stationary")
    labels = ["ADF Test Statistics", "P-Value", "Lag", "Observations"]
    
    for a,b in zip(result, labels):
        print(b + " - " + str(a))
    
    if result[1] <= 0.05:
        print("Strong evidence asainst null hypo and my ts is stationary")
    else:
        print("Weak evidence against null hypo and my ts is Non- Stationary")

# COMMAND ----------

adf_check(furn["Sales"])

# COMMAND ----------

adf_check(furn["Seasonality"].dropna())

# COMMAND ----------

#Since the data set is stationary, We can move forward with it
# AIC = -2LL + K
# K = Parameter
# Parameter = Trend(p d q) / Seasonality (P D Q)
# Difference d/D = difference - Integrated
# p/P - partial auto correlation (pacf)
# q/Q - Auto correlation (acf)

# Trend = 
# d = 0 , no differences required as data was stationary
# p = 0
# q = 0

# Seasonality
# D = 1
# P = 1
# Q = 1

# COMMAND ----------

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# COMMAND ----------

#For Furniture Trend(p)
plot_pacf(furn["Sales"], lags = 14)
plt.show()
#Furniture, p = 0
#Theshold Value - Shaded part - no correlation
#-1 to 1 (0-0.2 - neutral correlation, 0.2-0.6 - weak, 0.6-1 - strong correlation)

# COMMAND ----------

#For Furniture Trend(q)
plot_acf(furn["Sales"], lags = 14)
plt.show()
# Furniture, q = 0

# COMMAND ----------

#For Furniture Seasonality(P)
plot_pacf(furn["Seasonality"].dropna(), lags = 14)
plt.show()
# Furniture, P = 0

# COMMAND ----------

#For Furniture Seasonality(Q)
plot_acf(furn["Seasonality"].dropna(), lags = 14)
plt.show()
# Furniture, Q = 0

# COMMAND ----------

#Features
#Furniture => P 0= , Q = 0, D = 1, p = 0, q = 0, d = 0

# COMMAND ----------

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

# COMMAND ----------

# For Furniture
model = sm.tsa.statespace.SARIMAX(furn["Sales"],order = (0,1,1), seasonal_order = (0,1,1,12))
result  = model.fit()

# COMMAND ----------

result.summary()

# COMMAND ----------

# Auto Arima Approach

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

# Implementing the above parameters by using P&C approach to get the best results

for param in pdq:
    for sea in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(furn["Sales"],order = param, seasonal_order = sea,
                                             enforce_stationarity = False,
                                             enforce_invertibility = False)
            results = model.fit()
            print('Arima{}x{} - AIC:{}'.format(param, sea,results.aic))
        except:
            continue

# COMMAND ----------

# Best Parameters 
# Arima(0, 1, 1)x(0, 1, 1, 12) - AIC:279.5806233397717
# Arima(0, 1, 1)x(1, 1, 1, 12) - AIC:281.55766214612464
# Arima(1, 1, 1)x(0, 1, 1, 12) - AIC:281.38730069394

#Arima(0, 1, 1)x(0, 1, 1, 12) - AIC:418.360334908048


# COMMAND ----------

furn["forecast"] = result.predict(start = 24 , end  = 48 , dynamic = True)
furn[["Sales","forecast"]].plot()

# COMMAND ----------

#Forecasting
from pandas.tseries.offsets import DateOffset
future_date = [furn.index[-1] + DateOffset(months = x) for x in range(0,60)]

# COMMAND ----------

future_date = pd.DataFrame(index=future_date[1: ], columns=furn.columns)

# COMMAND ----------

forecast_df = pd.concat([furn, future_date])

# COMMAND ----------

forecast_df.tail(10)

# COMMAND ----------

forecast_df["forecast"] = result.predict(start = 49, end =133 , dynamic = True)
forecast_df[["Sales", "forecast"]].plot()

# COMMAND ----------

96+13

# COMMAND ----------

forecast_df["forecast"] = result.predict(start = 15, end =106 , dynamic = True)
forecast_df.head(50)

# COMMAND ----------

forecast_df.tail(5)

# COMMAND ----------



# COMMAND ----------

forecast_df = forecast_df.reset_index().rename(columns = {forecast_df.index.name: "Order Date"})

# COMMAND ----------

forecast_df

# COMMAND ----------

fore_df1 = spark.createDataFrame(forecast_df)

# COMMAND ----------

fore_df1.display()

# COMMAND ----------

#fore_df1.write.saveAsTable("forecast_store_f")
