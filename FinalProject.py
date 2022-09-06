#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


import pandas as pd
import numpy as np


# In[6]:


data = pd.read_csv("dataset.csv")


# # Chosing the most usefull columns - description
# - ID column contains only nan and 0.0 values so we delete it
# - HOURS contains the same information as HRS_BEGIN + HRS_END but in worse data format so we chose HRS_BEGIN & HOURS_END insted of HOURS
# - Agency contains only 97 non-null values, which is only 1.2% of our data so we delete it
# - HRS_BEGIN is the same as FROM_TIME but it has less null values and data format is better for us if we want to use this data set for some ML - example: 
#     - 3am in FROM_TIME = 300 in HRS_BEGIN, 3pm in FROM_TIME = 1500 in HRS_BEGIN
# - HRS_END is better than TIME_TO on the same basis as the above

# In[7]:


columns = ['REGULATION', 'DAYS', 'HRS_BEGIN', 'HRS_END', 'HRLIMIT', 'LAST_EDITED_USER', 'LAST_EDITED_DATE', 'EXCEPTIONS', 'shape']


# In[8]:


df = data[columns]


# In[9]:


df.head()


# In[10]:


print(df.REGULATION)


# # Removing the rows containing an empty value in REGULATION column
# ##### We are deleting these rows, because REGULATION attribute is our ML target

# In[11]:


df.dropna(how = "any", subset = ["REGULATION"], inplace = True)


# In[12]:


df.head()


# # Changing all our string data into lower case
# ##### As we can see below in non-number data we have some regulations that are included multiple times (e.g. 'Time limited' and 'Time LImited') which are done by case sensitivity so we want to transform all strings into lower case to deal with duplicates

# In[13]:


regulations = np.unique([reg for reg in df.REGULATION if not pd.isnull(reg)])
for regulation in regulations:
    print(regulation)


# In[14]:


string_cols = ['REGULATION', 'DAYS', 'LAST_EDITED_USER', 'EXCEPTIONS']
for column in string_cols:
    df[column] = df[column].str.lower()


# In[15]:


df.head()


# # Data unification - changing 'no parking any time' to 'no parking anytime'
# ##### Another example of dealing with duplicated regulation

# In[16]:


df["REGULATION"].replace({"no parking any time": "no parking anytime"}, inplace=True)


# # Removing the rows containing an empty value in shape
# ##### shape attribute will be used to create geodataframe basing on our dataframe so it cannot be empty

# In[17]:


df.dropna(how = "any", subset = ["shape"], inplace = True)


# In[18]:


df.info()


# # Getting regulations names

# In[19]:


regulations = np.unique([reg for reg in df.REGULATION if not pd.isnull(reg)])
print(regulations)


# In[20]:


shapes = [[row['shape'] for _, row in df.iterrows() if row.REGULATION == reg] for reg in regulations]


# # Creating dictionary regulation_name: shape

# In[21]:


shapes_dict = {reg: shape for reg, shape in zip(regulations, shapes)}


# # Open Street Map

# In[22]:


import geopandas as gpd
from shapely.geometry import multilinestring
from shapely import wkt


# In[23]:


df['shape'] = df['shape'].apply(wkt.loads)


# ### Creating geodataframe based on our data

# In[24]:


geo_df = gpd.GeoDataFrame(df, geometry=df['shape'])


# ### Setting CRS to NAD83 (EPSG4269)
# ##### We decided to chose this CRS because this is CRS which is most commonly used by U.S. federal agencies - https://www.nceas.ucsb.edu/sites/default/files/2020-04/OverviewCoordinateReferenceSystems.pdf

# In[25]:


geo_df = geo_df.set_crs(epsg=4269, allow_override=True)


# In[26]:


geo_df


# In[27]:


from pyrosm import get_data
from pyrosm import OSM


# # Downloading San Francisco network dataset from OSM

# In[28]:


sf = get_data("San Francisco")


# ### Setting CRS to NAD83 (EPSG4269)
# ##### We are setting the same CRS as we've set in our geodataframe

# In[29]:


osm = OSM(sf)
roads = osm.get_network()
roads = roads.set_crs(epsg=4269, allow_override=True)
roads.plot()


# ### Anlyzing San Francisco data

# In[30]:


roads.head()


# In[31]:


roads.columns


# # Chosing the most usefull columns for our problem
# #### We are cleaning San Francisco data to have only meaningfull attributes

# In[32]:


roads_usefull_cols = ['highway', 'name', 'geometry']
cleaned_roads = roads[roads_usefull_cols]


# In[33]:


cleaned_roads.head()


# # Combining our GeoDataFrame with cleaned San Francisco GeoDataFrame
# ##### At first we tried sjon function, however it took into account only regulations applied on streets so it did not work correctly when any parking regluation was apllied on sidewalks 
# ##### To prevent our dataset from loosing the majority of records we decided to use sjon_nearest function an play with max_distance attribute - succesfully we found the value that gave as the correct amount of records

# In[34]:


combined_df = gpd.sjoin_nearest(geo_df, cleaned_roads, max_distance=0.0000768)


# In[35]:


combined_df


# In[36]:


print(combined_df.crs)


# # Plotting regulations area

# In[37]:


import matplotlib.pyplot as plt


# In[38]:


regulation_color = {"government permit" : "red",
                    "limited no parking" : "orange",
                    "no overnight parking" : "purple",
                    "no oversized vehicles" : "yellow",
                    "no parking anytime" : "pink",
                    "no stopping" : "brown",
                    "paid + permit" : "green",
                    "time limited" : "blue"}
fig, ax = plt.subplots(figsize = (40, 32))
roads.plot(ax=ax, color="gray")
for reg in regulations:
    print(reg + ": " + regulation_color[reg])
    tmp_df = combined_df[combined_df['REGULATION'] == reg]
    tmp_df.plot(ax=ax, color=regulation_color[reg])


# # Machine Learning idea with San Francisco parking regulations data
# Our idea is to make a machine learning model which will be able to predict which parking regulations are applied in specific area, based on:
# - The type of way that contains the parking area, e.g. service, footway, primary etc
# - Time when parking regulation starts, e.g. 700.0 = 07:00 AM etc
# - Time when parking regulation ends, e.g. 1800.0 = 06:00 PM etc
# - Days when parking regulation is in force, e.g. 'm-f' = (Monday - Friday), 'm, th' = (Monday, Thursday)
# - Hours of parking limit
# 
# So we took only usefull columns from merged dataframe 

# In[39]:


machine_learning_df = combined_df[['highway', 'HRS_BEGIN', 'HRS_END', 'DAYS', 'HRLIMIT', 'REGULATION']]


# In[40]:


machine_learning_df.head()


# # Dealing with DAYS column

# In[41]:


days = [day for day in machine_learning_df.DAYS  if not pd.isnull(day)]
days = list(set(days))
print(days)


# #### Firstly we changed duplicated 'm-sat' to 'm-sa' to make sure that they will be considered as one

# In[42]:


machine_learning_df["DAYS"].replace({"m-sat": "m-sa"}, inplace=True)


# In[43]:


days = [day for day in machine_learning_df.DAYS  if not pd.isnull(day)]
days = list(set(days))
print(days)


# #### Then we tried to find out if 'm-s' means Monday-Saturday or Monday-Sunday

# In[44]:


print(machine_learning_df.loc[machine_learning_df['DAYS'] == 'm-s'])


# In[45]:


print(machine_learning_df.loc[machine_learning_df['DAYS'] == 'm-sa'])


# In[46]:


print(machine_learning_df.loc[machine_learning_df['DAYS'] == 'm-su'])


# #### It is hard to find out if m-s means Monday-Saturday or Monday-Sunday, however it consists of only 19 records of our data, so we decided to delete it

# In[47]:


machine_learning_df = machine_learning_df[machine_learning_df.DAYS != 'm-s']


# In[48]:


days = [day for day in machine_learning_df.DAYS  if not pd.isnull(day)]
days = list(set(days))
print(days)


# # Converting string values into numeric values using LabelEncoder

# In[49]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[50]:


cols_to_encode = ['highway', 'DAYS']
for col in cols_to_encode:
    machine_learning_df[col] = encoder.fit_transform(machine_learning_df[col].values)
    print(encoder.classes_)


# # Filling missing values with its column mean

# In[51]:


machine_learning_df.info()


# In[52]:


for col in machine_learning_df:
    if col != "REGULATION":
        machine_learning_df.fillna(value=machine_learning_df[col].mean(), inplace=True)


# In[53]:


machine_learning_df.info()


# # Our DataFrame is finally ready! Let's start with Machine Learning 
# ##### Creating X - data, and y - target

# In[54]:


X = machine_learning_df[['highway', 'HRS_BEGIN', 'HRS_END', 'DAYS', 'HRLIMIT']]
y = machine_learning_df['REGULATION']


# In[55]:


X.head()


# In[56]:


y.head()


# # Spliting our data into train and test sets

# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # Creating RandomForestClassifier
# ##### Our ML idea is typical classification example. We tried few different classifiers however they all gave similiar results so we decided to go with random forest because we know it quite well 

# In[58]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20)


# In[59]:


clf.fit(X_train, y_train)


# In[60]:


y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)


# # Checking accuracy score

# In[61]:


from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)


# In[62]:


print("Train accuracy: ", train_acc)
print("Test accuracy: ", test_acc)

