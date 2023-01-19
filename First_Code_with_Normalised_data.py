# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 23:53:42 2023

@author: Srinivas
"""

import pandas as pd # pandas library for manipulation
import numpy as np # numpy library for mathematical approaches
import re # regular expression facilities,  to check string matches a given pattern


def data_rd(path):
  """ 
  Read: To Read in the data from the file. The series name, series code and country code columns are dropped. Selects only the data related to CO2 emissions (kt).
  Country name is set as index. The index is reset and the dataframe is transposed. 
  
  Return: returns the original and transposed dataframes
  
  """
  actual_dat = pd.read_csv(path) # To read the csv file path
  transform_dat = actual_dat.copy() # Copying and storing the original date to transform_dat variable
  # Dropping the series code and country codes 
  transform_dat.drop(['Series Code', 'Country Code'], axis = 1, inplace = True)
  transform_dat = transform_dat[transform_dat['Series Name'] == 'Nitrous oxide emissions (thousand metric tons of CO2 equivalent)']  
  transform_dat.drop(['Series Name'], axis = 1, inplace = True) # Dropping the series names 
  transform_dat = transform_dat.iloc[:217, :]  # Logical slicing of transformed data
  transform_dat = transform_dat.set_index("Country Name") # Indexing a name for countries column
  transform_dat.reset_index(inplace = True) # Resetting the index 
  transform_dat = pd.DataFrame.transpose(transform_dat) # Transposing the data stored in transpose_dat variable
  return actual_dat, transform_dat # Returns the actual and transformed data

# Reading the local csv file data and storing to actual_dat and transformed_dat variables
actual_dat, transform_dat = data_rd('C:/Users\Srinivas\Documents\Data Science\Herts Coarse\Applied Data Science 1\Assignment_3 Clustering and Poster\e3c8e205-f78e-48a5-b480-f5a20de97ae8_Data.csv')

# Converting transformed data to a dataframe and storing in df variable
df = pd.DataFrame.transpose(transform_dat)
df.head() # Prints top 5 rows and columns from df dataframe

df.shape # Number of rows and columns in the data


""" Checking the existence of missing values """

df.isna().sum() 
df.info() # Prints the data count of rows and columns

df['1990 [YR1990]'].value_counts()
# Calculating the sums of years 2020 and 2021
df['2020 [YR2020]'].sum()
df['2021 [YR2021]'].sum()
df.drop(['2020 [YR2020]', '2021 [YR2021]'], axis = 1, inplace = True) # Dropping the years


# To Pre-Process the data before clustering analysis

def preprocess(num):
    
  """
  Function: Obtains the number and attempts to transform it to an integer before returning it. 
  
  Return: If the function fails, any occurrences of dots are substituted with  “0” and the result is returned.
  
  """
  try:
    new_num = int(num)
  except:
    new_num = re.sub(r".*", '0', num)
  return new_num

# Calling the preprocess function for all the years choosed for clustering analysis
df['1990 [YR1990]'] = [int(preprocess(num)) for num in df['1990 [YR1990]']]
df['2000 [YR2000]'] = [int(preprocess(num)) for num in df['2000 [YR2000]']]
df['2012 [YR2012]'] = [int(preprocess(num)) for num in df['2012 [YR2012]']]
df['2013 [YR2013]'] = [int(preprocess(num)) for num in df['2013 [YR2013]']]
df['2014 [YR2014]'] = [int(preprocess(num)) for num in df['2014 [YR2014]']]
df['2015 [YR2015]'] = [int(preprocess(num)) for num in df['2015 [YR2015]']]
df['2016 [YR2016]'] = [int(preprocess(num)) for num in df['2016 [YR2016]']]
df['2017 [YR2017]'] = [int(preprocess(num)) for num in df['2017 [YR2017]']]
df['2018 [YR2018]'] = [int(preprocess(num)) for num in df['2018 [YR2018]']]
df['2019 [YR2019]'] = [int(preprocess(num)) for num in df['2019 [YR2019]']]

df.head() # Prints top 5 rows and columns from df dataframe
df.info() # Prints the data count of rows and columns


def normalised(array):
  """ 
  Function: This function calculates the minimum and maximum value of the array and stores in scaled variable  
  This Array can be a numpy array or a column of a dataframe
  
  Return: To Return array normalised to [0,1]. 
  """
  min_val = np.min(array)
  max_val = np.max(array)
  scaled = (array-min_val) / (max_val-min_val)
  return scaled


def normalised_df(df, first = 0, last = None):
  """
  Function: To Return all columns of the dataframe normalised to [0,1] with the exception of the first (containing the names)
  and doing all in one function is fine but calling the function "normalised" to do the normalisation of one column. 
  
  First and last: Including columns from first to last are normalised and defaulted to all. 
  The default correspondsHere as here empty entry is None.
  
  Return: This function returns the normalised data
  
  """
  # To iterate over all columns which are numerical adnd excluding the first column
  for col in df.columns[first:last]: 
    df[col] = normalised(df[col])
  return df