```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas as pd
import numpy as np
```


```python
data = pd.read_csv("dataset.csv")
```

# Chosing the most usefull columns - description
- ID column contains only nan and 0.0 values so we delete it
- HOURS contains the same information as HRS_BEGIN + HRS_END but in worse data format so we chose HRS_BEGIN & HOURS_END insted of HOURS
- Agency contains only 97 non-null values, which is only 1.2% of our data so we delete it
- HRS_BEGIN is the same as FROM_TIME but it has less null values and data format is better for us if we want to use this data set for some ML - example: 
    - 3am in FROM_TIME = 300 in HRS_BEGIN, 3pm in FROM_TIME = 1500 in HRS_BEGIN
- HRS_END is better than TIME_TO on the same basis as the above


```python
columns = ['REGULATION', 'DAYS', 'HRS_BEGIN', 'HRS_END', 'HRLIMIT', 'LAST_EDITED_USER', 'LAST_EDITED_DATE', 'EXCEPTIONS', 'shape']
```


```python
df = data[columns]
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REGULATION</th>
      <th>DAYS</th>
      <th>HRS_BEGIN</th>
      <th>HRS_END</th>
      <th>HRLIMIT</th>
      <th>LAST_EDITED_USER</th>
      <th>LAST_EDITED_DATE</th>
      <th>EXCEPTIONS</th>
      <th>shape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Time limited</td>
      <td>M-Sa</td>
      <td>700.0</td>
      <td>1800.0</td>
      <td>1.0</td>
      <td>MTA</td>
      <td>20170513001346</td>
      <td>None. Regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.49074 37.74207, -122.49...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Time limited</td>
      <td>M-F</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>GCHAN</td>
      <td>20190820232238</td>
      <td>None. Regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.38895 37.742348, -122.3...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Time limited</td>
      <td>M-F</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>GCHAN</td>
      <td>20190820232234</td>
      <td>None. Regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.38897 37.742317, -122.3...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Time limited</td>
      <td>M-F</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>GCHAN</td>
      <td>20190820232309</td>
      <td>None. Regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.388504 37.74297, -122.3...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No parking any time</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>RAYNELLCOOPER</td>
      <td>20191031234514</td>
      <td>None. Regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.38898 37.764225, -122.3...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.REGULATION)
```

    0              Time limited
    1              Time limited
    2              Time limited
    3              Time limited
    4       No parking any time
                   ...         
    7742      Government permit
    7743      Government permit
    7744      Government permit
    7745      Government permit
    7746      Government permit
    Name: REGULATION, Length: 7747, dtype: object
    

# Removing the rows containing an empty value in REGULATION column
##### We are deleting these rows, because REGULATION attribute is our ML target


```python
df.dropna(how = "any", subset = ["REGULATION"], inplace = True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REGULATION</th>
      <th>DAYS</th>
      <th>HRS_BEGIN</th>
      <th>HRS_END</th>
      <th>HRLIMIT</th>
      <th>LAST_EDITED_USER</th>
      <th>LAST_EDITED_DATE</th>
      <th>EXCEPTIONS</th>
      <th>shape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Time limited</td>
      <td>M-Sa</td>
      <td>700.0</td>
      <td>1800.0</td>
      <td>1.0</td>
      <td>MTA</td>
      <td>20170513001346</td>
      <td>None. Regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.49074 37.74207, -122.49...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Time limited</td>
      <td>M-F</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>GCHAN</td>
      <td>20190820232238</td>
      <td>None. Regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.38895 37.742348, -122.3...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Time limited</td>
      <td>M-F</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>GCHAN</td>
      <td>20190820232234</td>
      <td>None. Regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.38897 37.742317, -122.3...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Time limited</td>
      <td>M-F</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>GCHAN</td>
      <td>20190820232309</td>
      <td>None. Regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.388504 37.74297, -122.3...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No parking any time</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>RAYNELLCOOPER</td>
      <td>20191031234514</td>
      <td>None. Regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.38898 37.764225, -122.3...</td>
    </tr>
  </tbody>
</table>
</div>



# Changing all our string data into lower case
##### As we can see below in non-number data we have some regulations that are included multiple times (e.g. 'Time limited' and 'Time LImited') which are done by case sensitivity so we want to transform all strings into lower case to deal with duplicates


```python
regulations = np.unique([reg for reg in df.REGULATION if not pd.isnull(reg)])
for regulation in regulations:
    print(regulation)
```

    Government Permit
    Government permit
    Limited No Parking
    No Oversized Vehicles
    No Parking Anytime
    No Stopping
    No overnight parking
    No oversized vehicles
    No parking any time
    Paid + Permit
    Time LImited
    Time Limited
    Time limited
    


```python
string_cols = ['REGULATION', 'DAYS', 'LAST_EDITED_USER', 'EXCEPTIONS']
for column in string_cols:
    df[column] = df[column].str.lower()
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REGULATION</th>
      <th>DAYS</th>
      <th>HRS_BEGIN</th>
      <th>HRS_END</th>
      <th>HRLIMIT</th>
      <th>LAST_EDITED_USER</th>
      <th>LAST_EDITED_DATE</th>
      <th>EXCEPTIONS</th>
      <th>shape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>time limited</td>
      <td>m-sa</td>
      <td>700.0</td>
      <td>1800.0</td>
      <td>1.0</td>
      <td>mta</td>
      <td>20170513001346</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.49074 37.74207, -122.49...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>time limited</td>
      <td>m-f</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>gchan</td>
      <td>20190820232238</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.38895 37.742348, -122.3...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>time limited</td>
      <td>m-f</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>gchan</td>
      <td>20190820232234</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.38897 37.742317, -122.3...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>time limited</td>
      <td>m-f</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>gchan</td>
      <td>20190820232309</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.388504 37.74297, -122.3...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>no parking any time</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>raynellcooper</td>
      <td>20191031234514</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>MULTILINESTRING ((-122.38898 37.764225, -122.3...</td>
    </tr>
  </tbody>
</table>
</div>



# Data unification - changing 'no parking any time' to 'no parking anytime'
##### Another example of dealing with duplicated regulation


```python
df["REGULATION"].replace({"no parking any time": "no parking anytime"}, inplace=True)
```

# Removing the rows containing an empty value in shape
##### shape attribute will be used to create geodataframe basing on our dataframe so it cannot be empty


```python
df.dropna(how = "any", subset = ["shape"], inplace = True)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7723 entries, 0 to 7746
    Data columns (total 9 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   REGULATION        7723 non-null   object 
     1   DAYS              7455 non-null   object 
     2   HRS_BEGIN         7635 non-null   float64
     3   HRS_END           7635 non-null   float64
     4   HRLIMIT           7472 non-null   float64
     5   LAST_EDITED_USER  7723 non-null   object 
     6   LAST_EDITED_DATE  7723 non-null   int64  
     7   EXCEPTIONS        7617 non-null   object 
     8   shape             7723 non-null   object 
    dtypes: float64(3), int64(1), object(5)
    memory usage: 603.4+ KB
    

# Getting regulations names


```python
regulations = np.unique([reg for reg in df.REGULATION if not pd.isnull(reg)])
print(regulations)
```

    ['government permit' 'limited no parking' 'no overnight parking'
     'no oversized vehicles' 'no parking anytime' 'no stopping'
     'paid + permit' 'time limited']
    


```python
shapes = [[row['shape'] for _, row in df.iterrows() if row.REGULATION == reg] for reg in regulations]

```

# Creating dictionary regulation_name: shape


```python
shapes_dict = {reg: shape for reg, shape in zip(regulations, shapes)}
```

# Open Street Map


```python
import geopandas as gpd
from shapely.geometry import multilinestring
from shapely import wkt
```


```python
df['shape'] = df['shape'].apply(wkt.loads)
```

### Creating geodataframe based on our data


```python
geo_df = gpd.GeoDataFrame(df, geometry=df['shape'])
```

### Setting CRS to NAD83 (EPSG4269)
##### We decided to chose this CRS because this is CRS which is most commonly used by U.S. federal agencies - https://www.nceas.ucsb.edu/sites/default/files/2020-04/OverviewCoordinateReferenceSystems.pdf


```python
geo_df = geo_df.set_crs(epsg=4269, allow_override=True)
```


```python
geo_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REGULATION</th>
      <th>DAYS</th>
      <th>HRS_BEGIN</th>
      <th>HRS_END</th>
      <th>HRLIMIT</th>
      <th>LAST_EDITED_USER</th>
      <th>LAST_EDITED_DATE</th>
      <th>EXCEPTIONS</th>
      <th>shape</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>time limited</td>
      <td>m-sa</td>
      <td>700.0</td>
      <td>1800.0</td>
      <td>1.0</td>
      <td>mta</td>
      <td>20170513001346</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>(LINESTRING (-122.49074 37.74207, -122.490746 ...</td>
      <td>MULTILINESTRING ((-122.49074 37.74207, -122.49...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>time limited</td>
      <td>m-f</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>gchan</td>
      <td>20190820232238</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>(LINESTRING (-122.38895 37.742348, -122.38822 ...</td>
      <td>MULTILINESTRING ((-122.38895 37.74235, -122.38...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>time limited</td>
      <td>m-f</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>gchan</td>
      <td>20190820232234</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>(LINESTRING (-122.38897 37.742317, -122.38824 ...</td>
      <td>MULTILINESTRING ((-122.38897 37.74232, -122.38...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>time limited</td>
      <td>m-f</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>gchan</td>
      <td>20190820232309</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>(LINESTRING (-122.388504 37.74297, -122.38902 ...</td>
      <td>MULTILINESTRING ((-122.38850 37.74297, -122.38...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>no parking anytime</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>raynellcooper</td>
      <td>20191031234514</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>(LINESTRING (-122.38898 37.764225, -122.38886 ...</td>
      <td>MULTILINESTRING ((-122.38898 37.76423, -122.38...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7742</th>
      <td>government permit</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>joelmandella</td>
      <td>20180221010035</td>
      <td>sfpd vehicles with a permit from the sfmta all...</td>
      <td>(LINESTRING (-122.431915 37.78001, -122.43196 ...</td>
      <td>MULTILINESTRING ((-122.43192 37.78001, -122.43...</td>
    </tr>
    <tr>
      <th>7743</th>
      <td>government permit</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>mta</td>
      <td>20160909230726</td>
      <td>NaN</td>
      <td>(LINESTRING (-122.40362 37.77529, -122.403206 ...</td>
      <td>MULTILINESTRING ((-122.40362 37.77529, -122.40...</td>
    </tr>
    <tr>
      <th>7744</th>
      <td>government permit</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>mta</td>
      <td>20160909230528</td>
      <td>NaN</td>
      <td>(LINESTRING (-122.40477 37.774487, -122.40455 ...</td>
      <td>MULTILINESTRING ((-122.40477 37.77449, -122.40...</td>
    </tr>
    <tr>
      <th>7745</th>
      <td>government permit</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>joelmandella</td>
      <td>20180103003553</td>
      <td>NaN</td>
      <td>(LINESTRING (-122.412445 37.783684, -122.41247...</td>
      <td>MULTILINESTRING ((-122.41245 37.78368, -122.41...</td>
    </tr>
    <tr>
      <th>7746</th>
      <td>government permit</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>mta</td>
      <td>20160909230456</td>
      <td>NaN</td>
      <td>(LINESTRING (-122.40556 37.77506, -122.4052 37...</td>
      <td>MULTILINESTRING ((-122.40556 37.77506, -122.40...</td>
    </tr>
  </tbody>
</table>
<p>7723 rows × 10 columns</p>
</div>




```python
from pyrosm import get_data
from pyrosm import OSM
```

# Downloading San Francisco network dataset from OSM


```python
sf = get_data("San Francisco")
```

### Setting CRS to NAD83 (EPSG4269)
##### We are setting the same CRS as we've set in our geodataframe


```python
osm = OSM(sf)
roads = osm.get_network()
roads = roads.set_crs(epsg=4269, allow_override=True)
roads.plot()
```




    <AxesSubplot:>




    
![png](output_37_1.png)
    


### Anlyzing San Francisco data


```python
roads.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>access</th>
      <th>area</th>
      <th>bicycle</th>
      <th>bridge</th>
      <th>busway</th>
      <th>cycleway</th>
      <th>est_width</th>
      <th>foot</th>
      <th>footway</th>
      <th>highway</th>
      <th>...</th>
      <th>tunnel</th>
      <th>turn</th>
      <th>width</th>
      <th>id</th>
      <th>timestamp</th>
      <th>version</th>
      <th>tags</th>
      <th>osm_type</th>
      <th>geometry</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>residential</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>5004035</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
      <td>way</td>
      <td>MULTILINESTRING ((-122.41648 37.79905, -122.41...</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>residential</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>7373728</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
      <td>way</td>
      <td>MULTILINESTRING ((-122.40013 37.77428, -122.40...</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>residential</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>7373736</td>
      <td>0</td>
      <td>-1</td>
      <td>{"name:etymology:wikidata":"Q107178240"}</td>
      <td>way</td>
      <td>MULTILINESTRING ((-122.40013 37.77428, -122.39...</td>
      <td>534.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>residential</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>7448875</td>
      <td>0</td>
      <td>-1</td>
      <td>{"name:etymology:wikidata":"Q16105754"}</td>
      <td>way</td>
      <td>MULTILINESTRING ((-122.39110 37.76967, -122.39...</td>
      <td>147.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>track</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>7715657</td>
      <td>0</td>
      <td>-1</td>
      <td>{"created_by":"JOSM","motorcycle":"no"}</td>
      <td>way</td>
      <td>MULTILINESTRING ((-122.32459 37.88995, -122.32...</td>
      <td>274.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
roads.columns
```




    Index(['access', 'area', 'bicycle', 'bridge', 'busway', 'cycleway',
           'est_width', 'foot', 'footway', 'highway', 'junction', 'lanes', 'lit',
           'maxspeed', 'motorcar', 'motor_vehicle', 'name', 'oneway', 'overtaking',
           'path', 'psv', 'ref', 'service', 'segregated', 'sidewalk', 'smoothness',
           'surface', 'tracktype', 'tunnel', 'turn', 'width', 'id', 'timestamp',
           'version', 'tags', 'osm_type', 'geometry', 'length'],
          dtype='object')



# Chosing the most usefull columns for our problem
#### We are cleaning San Francisco data to have only meaningfull attributes


```python
roads_usefull_cols = ['highway', 'name', 'geometry']
cleaned_roads = roads[roads_usefull_cols]
```


```python
cleaned_roads.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>highway</th>
      <th>name</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>residential</td>
      <td>Macondray Lane</td>
      <td>MULTILINESTRING ((-122.41648 37.79905, -122.41...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>residential</td>
      <td>6th Street</td>
      <td>MULTILINESTRING ((-122.40013 37.77428, -122.40...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>residential</td>
      <td>Bluxome Street</td>
      <td>MULTILINESTRING ((-122.40013 37.77428, -122.39...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>residential</td>
      <td>Nelson Rising Lane</td>
      <td>MULTILINESTRING ((-122.39110 37.76967, -122.39...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>track</td>
      <td>None</td>
      <td>MULTILINESTRING ((-122.32459 37.88995, -122.32...</td>
    </tr>
  </tbody>
</table>
</div>



# Combining our GeoDataFrame with cleaned San Francisco GeoDataFrame
##### At first we tried sjon function, however it took into account only regulations applied on streets so it did not work correctly when any parking regluation was apllied on sidewalks 
##### To prevent our dataset from loosing the majority of records we decided to use sjon_nearest function an play with max_distance attribute - succesfully we found the value that gave as the correct amount of records


```python
combined_df = gpd.sjoin_nearest(geo_df, cleaned_roads, max_distance=0.0000768)
```


```python
combined_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REGULATION</th>
      <th>DAYS</th>
      <th>HRS_BEGIN</th>
      <th>HRS_END</th>
      <th>HRLIMIT</th>
      <th>LAST_EDITED_USER</th>
      <th>LAST_EDITED_DATE</th>
      <th>EXCEPTIONS</th>
      <th>shape</th>
      <th>geometry</th>
      <th>index_right</th>
      <th>highway</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>time limited</td>
      <td>m-sa</td>
      <td>700.0</td>
      <td>1800.0</td>
      <td>1.0</td>
      <td>mta</td>
      <td>20170513001346</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>(LINESTRING (-122.49074 37.74207, -122.490746 ...</td>
      <td>MULTILINESTRING ((-122.49074 37.74207, -122.49...</td>
      <td>27070</td>
      <td>residential</td>
      <td>33rd Avenue</td>
    </tr>
    <tr>
      <th>1</th>
      <td>time limited</td>
      <td>m-f</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>gchan</td>
      <td>20190820232238</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>(LINESTRING (-122.38895 37.742348, -122.38822 ...</td>
      <td>MULTILINESTRING ((-122.38895 37.74235, -122.38...</td>
      <td>46750</td>
      <td>footway</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>time limited</td>
      <td>m-f</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>gchan</td>
      <td>20190820232234</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>(LINESTRING (-122.38897 37.742317, -122.38824 ...</td>
      <td>MULTILINESTRING ((-122.38897 37.74232, -122.38...</td>
      <td>46750</td>
      <td>footway</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>time limited</td>
      <td>m-f</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>gchan</td>
      <td>20190820232238</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>(LINESTRING (-122.38895 37.742348, -122.38822 ...</td>
      <td>MULTILINESTRING ((-122.38895 37.74235, -122.38...</td>
      <td>738</td>
      <td>primary</td>
      <td>3rd Street</td>
    </tr>
    <tr>
      <th>2</th>
      <td>time limited</td>
      <td>m-f</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>4.0</td>
      <td>gchan</td>
      <td>20190820232234</td>
      <td>none. regulation applies to all vehicles.</td>
      <td>(LINESTRING (-122.38897 37.742317, -122.38824 ...</td>
      <td>MULTILINESTRING ((-122.38897 37.74232, -122.38...</td>
      <td>27107</td>
      <td>primary</td>
      <td>3rd Street</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7741</th>
      <td>government permit</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>mta</td>
      <td>20160909230609</td>
      <td>NaN</td>
      <td>(LINESTRING (-122.40382 37.776268, -122.4034 3...</td>
      <td>MULTILINESTRING ((-122.40382 37.77627, -122.40...</td>
      <td>2407</td>
      <td>residential</td>
      <td>Harriet Street</td>
    </tr>
    <tr>
      <th>7743</th>
      <td>government permit</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>mta</td>
      <td>20160909230726</td>
      <td>NaN</td>
      <td>(LINESTRING (-122.40362 37.77529, -122.403206 ...</td>
      <td>MULTILINESTRING ((-122.40362 37.77529, -122.40...</td>
      <td>2407</td>
      <td>residential</td>
      <td>Harriet Street</td>
    </tr>
    <tr>
      <th>7742</th>
      <td>government permit</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>joelmandella</td>
      <td>20180221010035</td>
      <td>sfpd vehicles with a permit from the sfmta all...</td>
      <td>(LINESTRING (-122.431915 37.78001, -122.43196 ...</td>
      <td>MULTILINESTRING ((-122.43192 37.78001, -122.43...</td>
      <td>38334</td>
      <td>service</td>
      <td>None</td>
    </tr>
    <tr>
      <th>7744</th>
      <td>government permit</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>mta</td>
      <td>20160909230528</td>
      <td>NaN</td>
      <td>(LINESTRING (-122.40477 37.774487, -122.40455 ...</td>
      <td>MULTILINESTRING ((-122.40477 37.77449, -122.40...</td>
      <td>54937</td>
      <td>secondary</td>
      <td>Bryant Street</td>
    </tr>
    <tr>
      <th>7746</th>
      <td>government permit</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>mta</td>
      <td>20160909230456</td>
      <td>NaN</td>
      <td>(LINESTRING (-122.40556 37.77506, -122.4052 37...</td>
      <td>MULTILINESTRING ((-122.40556 37.77506, -122.40...</td>
      <td>39117</td>
      <td>service</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>7724 rows × 13 columns</p>
</div>




```python
print(combined_df.crs)
```

    epsg:4269
    

# Plotting regulations area


```python
import matplotlib.pyplot as plt
```


```python
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
```

    government permit: red
    limited no parking: orange
    no overnight parking: purple
    no oversized vehicles: yellow
    no parking anytime: pink
    no stopping: brown
    paid + permit: green
    time limited: blue
    


    
![png](output_50_1.png)
    


# Machine Learning idea with San Francisco parking regulations data
Our idea is to make a machine learning model which will be able to predict which parking regulations are applied in specific area, based on:
- The type of way that contains the parking area, e.g. service, footway, primary etc
- Time when parking regulation starts, e.g. 700.0 = 07:00 AM etc
- Time when parking regulation ends, e.g. 1800.0 = 06:00 PM etc
- Days when parking regulation is in force, e.g. 'm-f' = (Monday - Friday), 'm, th' = (Monday, Thursday)
- Hours of parking limit

So we took only usefull columns from merged dataframe 


```python
machine_learning_df = combined_df[['highway', 'HRS_BEGIN', 'HRS_END', 'DAYS', 'HRLIMIT', 'REGULATION']]
```


```python
machine_learning_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>highway</th>
      <th>HRS_BEGIN</th>
      <th>HRS_END</th>
      <th>DAYS</th>
      <th>HRLIMIT</th>
      <th>REGULATION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>residential</td>
      <td>700.0</td>
      <td>1800.0</td>
      <td>m-sa</td>
      <td>1.0</td>
      <td>time limited</td>
    </tr>
    <tr>
      <th>1</th>
      <td>footway</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>m-f</td>
      <td>4.0</td>
      <td>time limited</td>
    </tr>
    <tr>
      <th>2</th>
      <td>footway</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>m-f</td>
      <td>4.0</td>
      <td>time limited</td>
    </tr>
    <tr>
      <th>1</th>
      <td>primary</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>m-f</td>
      <td>4.0</td>
      <td>time limited</td>
    </tr>
    <tr>
      <th>2</th>
      <td>primary</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>m-f</td>
      <td>4.0</td>
      <td>time limited</td>
    </tr>
  </tbody>
</table>
</div>



# Dealing with DAYS column


```python
days = [day for day in machine_learning_df.DAYS  if not pd.isnull(day)]
days = list(set(days))
print(days)
```

    ['sa', 'm-sat', 'm-f', 'm-s', 'm-su', 'm-sa', 'm, th']
    

#### Firstly we changed duplicated 'm-sat' to 'm-sa' to make sure that they will be considered as one


```python
machine_learning_df["DAYS"].replace({"m-sat": "m-sa"}, inplace=True)
```


```python
days = [day for day in machine_learning_df.DAYS  if not pd.isnull(day)]
days = list(set(days))
print(days)
```

    ['sa', 'm-f', 'm-s', 'm-su', 'm-sa', 'm, th']
    

#### Then we tried to find out if 'm-s' means Monday-Saturday or Monday-Sunday


```python
print(machine_learning_df.loc[machine_learning_df['DAYS'] == 'm-s'])
```

              highway  HRS_BEGIN  HRS_END DAYS  HRLIMIT    REGULATION
    4954  residential      700.0   1800.0  m-s      2.0  time limited
    5896  residential      700.0   1800.0  m-s      4.0  time limited
    913       footway      700.0   1800.0  m-s      2.0  time limited
    1830  residential      700.0   1800.0  m-s      2.0  time limited
    6238  residential      700.0   1800.0  m-s      2.0  time limited
    3009      footway      700.0   1800.0  m-s      2.0  time limited
    6276      footway      700.0   1800.0  m-s      2.0  time limited
    3470      footway      700.0   1800.0  m-s      2.0  time limited
    4336      footway      700.0   1800.0  m-s      2.0  time limited
    4651  residential      700.0   1800.0  m-s      2.0  time limited
    4717  residential      700.0   1800.0  m-s      2.0  time limited
    5467      footway      700.0   1800.0  m-s      2.0  time limited
    5467      service      700.0   1800.0  m-s      2.0  time limited
    5896  residential      700.0   1800.0  m-s      4.0  time limited
    6276      service      700.0   1800.0  m-s      2.0  time limited
    6276      footway      700.0   1800.0  m-s      2.0  time limited
    6276      footway      700.0   1800.0  m-s      2.0  time limited
    


```python
print(machine_learning_df.loc[machine_learning_df['DAYS'] == 'm-sa'])
```

              highway  HRS_BEGIN  HRS_END  DAYS  HRLIMIT     REGULATION
    0     residential      700.0   1800.0  m-sa      1.0   time limited
    6         footway      900.0   2200.0  m-sa      4.0   time limited
    11        footway      700.0   1800.0  m-sa      1.0   time limited
    196       footway      700.0   1800.0  m-sa      1.0   time limited
    20        footway      700.0   1800.0  m-sa      1.0   time limited
    ...           ...        ...      ...   ...      ...            ...
    7725     tertiary      700.0   1800.0  m-sa      4.0   time limited
    7699      footway      900.0   1800.0  m-sa      NaN  paid + permit
    7704    secondary      700.0   1800.0  m-sa      4.0   time limited
    7711      service      900.0   1800.0  m-sa      NaN  paid + permit
    7732    secondary      700.0   1800.0  m-sa      4.0   time limited
    
    [1377 rows x 6 columns]
    


```python
print(machine_learning_df.loc[machine_learning_df['DAYS'] == 'm-su'])
```

          highway  HRS_BEGIN  HRS_END  DAYS  HRLIMIT             REGULATION
    3016  footway     2400.0    600.0  m-su      0.0  no oversized vehicles
    3833  footway     2400.0    600.0  m-su      0.0  no oversized vehicles
    5271  footway     2400.0    600.0  m-su      0.0  no oversized vehicles
    6311  footway      700.0   2200.0  m-su      2.0           time limited
    60    footway     2400.0    600.0  m-su      NaN  no oversized vehicles
    ...       ...        ...      ...   ...      ...                    ...
    7636  footway      700.0   2100.0  m-su      2.0           time limited
    7637  footway      700.0   2100.0  m-su      2.0           time limited
    7658  footway        0.0      0.0  m-su      0.0      government permit
    7662  service        0.0      0.0  m-su      0.0      government permit
    7689  primary     2400.0    600.0  m-su      NaN  no oversized vehicles
    
    [786 rows x 6 columns]
    

#### It is hard to find out if m-s means Monday-Saturday or Monday-Sunday, however it consists of only 19 records of our data, so we decided to delete it


```python
machine_learning_df = machine_learning_df[machine_learning_df.DAYS != 'm-s']
```


```python
days = [day for day in machine_learning_df.DAYS  if not pd.isnull(day)]
days = list(set(days))
print(days)
```

    ['m-f', 'm-su', 'm, th', 'm-sa', 'sa']
    

# Converting string values into numeric values using LabelEncoder


```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
```


```python
cols_to_encode = ['highway', 'DAYS']
for col in cols_to_encode:
    machine_learning_df[col] = encoder.fit_transform(machine_learning_df[col].values)
    print(encoder.classes_)
```

    ['corridor' 'cycleway' 'footway' 'living_street' 'path' 'pedestrian'
     'primary' 'primary_link' 'residential' 'secondary' 'secondary_link'
     'service' 'steps' 'tertiary' 'tertiary_link' 'track' 'trunk' 'trunk_link'
     'unclassified']
    ['m, th' 'm-f' 'm-sa' 'm-su' 'sa' nan]
    

# Filling missing values with its column mean


```python
machine_learning_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7707 entries, 0 to 7746
    Data columns (total 6 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   highway     7707 non-null   int32  
     1   HRS_BEGIN   7579 non-null   float64
     2   HRS_END     7579 non-null   float64
     3   DAYS        7707 non-null   int32  
     4   HRLIMIT     7416 non-null   float64
     5   REGULATION  7707 non-null   object 
    dtypes: float64(3), int32(2), object(1)
    memory usage: 361.3+ KB
    


```python
for col in machine_learning_df:
    if col != "REGULATION":
        machine_learning_df.fillna(value=machine_learning_df[col].mean(), inplace=True)
```


```python
machine_learning_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7707 entries, 0 to 7746
    Data columns (total 6 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   highway     7707 non-null   int32  
     1   HRS_BEGIN   7707 non-null   float64
     2   HRS_END     7707 non-null   float64
     3   DAYS        7707 non-null   int32  
     4   HRLIMIT     7707 non-null   float64
     5   REGULATION  7707 non-null   object 
    dtypes: float64(3), int32(2), object(1)
    memory usage: 361.3+ KB
    

# Our DataFrame is finally ready! Let's start with Machine Learning 
##### Creating X - data, and y - target


```python
X = machine_learning_df[['highway', 'HRS_BEGIN', 'HRS_END', 'DAYS', 'HRLIMIT']]
y = machine_learning_df['REGULATION']
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>highway</th>
      <th>HRS_BEGIN</th>
      <th>HRS_END</th>
      <th>DAYS</th>
      <th>HRLIMIT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>700.0</td>
      <td>1800.0</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>900.0</td>
      <td>1800.0</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.head()
```




    0    time limited
    1    time limited
    2    time limited
    1    time limited
    2    time limited
    Name: REGULATION, dtype: object



# Spliting our data into train and test sets


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

# Creating RandomForestClassifier
##### Our ML idea is typical classification example. We tried few different classifiers however they all gave similiar results so we decided to go with random forest because we know it quite well 


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20)
```


```python
clf.fit(X_train, y_train)
```




    RandomForestClassifier(n_estimators=20)




```python
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
```

# Checking accuracy score


```python
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
```


```python
print("Train accuracy: ", train_acc)
print("Test accuracy: ", test_acc)
```

    Train accuracy:  0.9980535279805353
    Test accuracy:  0.9941634241245136
    
