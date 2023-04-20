# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT * FROM `hive_metastore`.`default`.`sample_superstore_1`;

# COMMAND ----------

df = spark.read.table("hive_metastore.default.sample_superstore_1")

# COMMAND ----------

df.display()

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

ss.info()

# COMMAND ----------

ss.Category.value_counts()

# COMMAND ----------

furn = ss.loc[ss["Category"] == "Furniture"]

# COMMAND ----------

furn.head()

# COMMAND ----------

furn.isnull().sum()

# COMMAND ----------

furn.columns

# COMMAND ----------

cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode',
       'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State',
       'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category',
       'Product Name', 'Quantity', 'Discount', 'Profit']

# COMMAND ----------

furn.drop(cols, axis = 1, inplace = True)

# COMMAND ----------

furn.head(10)

# COMMAND ----------

furn.sort_values("Order Date")

# COMMAND ----------

furn = furn.groupby("Order Date")["Sales"].sum().reset_index()

# COMMAND ----------

furn.head()

# COMMAND ----------

#Decomposition

from statsmodels.tsa.seasonal import seasonal _decompose

# COMMAND ----------

