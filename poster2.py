# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 06:57:21 2023

@author: Yashomadhav Mudgal
"""

import pandas as pd
import numpy as np
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import seaborn as sns
import map
import sklearn.preprocessing as prep
from sklearn import cluster
import sklearn.cluster as cluster

def norm(array): # Normalization
    """ Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe"""
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled


def norm_df(df, first=0, last=None):
    """
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    """
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df

def getdata(filename):
    '''
    This fuction returns two dataframes.One with years as column and other with countries as column
    we have transposed data from row to column as well as column to rows.
    It takes one argument as file name to read the data using pandas.
    
    '''
    df = pd.read_csv(filename, skiprows=(4), index_col=False) #for reading csv file and skipping unused rows from data
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] #for dropping column from data
    df = df.loc[df['Country Name'].isin(countries)]#for selecting specific column from data
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name','Indicator Code'], var_name='Years') # Converting years in a single column 
    del df2['Country Code'] # Deleting coutry code column
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code'],'Country Name').reset_index() # Creating countries separate columns from rows.
    return df, df2 # Returning two dataframes one with countries as column and other with years as column


countries= ['Australia', 'United States', 'China', 'United Kingdom', 'India', 'Canada'] # Countries list to filter data

df, df2 = getdata('API_19_DS2_en_csv_v2_4773766.csv') # reading file to dataframes using defined function.
df2 = df2.loc[df2['Indicator Code'].isin(['SP.POP.GROW'])] # Filter data using indicator code.
print(df2)
print(df2.describe())
print()
df3 = df2.fillna(0)
print(df3)
df4 = df3[(df3['Years']>="1965") & (df3['Years']<="2021")]
#df4.to_csv("myfile6.csv")
print(df4.describe())

# Plotting heat map (Co-relation)

def map_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns␣
    ↪→in the dataframe.
    Input:
    df: pandas DataFrame
    size: vertical and horizontal size of the plot (in inch)
    """
    corr = df4.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='rocket')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
corr = df4.corr()
print(corr)
map_corr(df4)
plt.show()

# Scatter plot 


pd.plotting.scatter_matrix(df4, figsize=(9.0, 9.0))
plt.tight_layout() # helps to avoid overlap of labels
plt.show()


# extract columns for fitting


df_fit = df4[["India", "Canada"]].copy()
# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_fish. This make the plots with the
# original measurements
df_fit = norm_df(df_fit)
print(df_fit.describe())
print()

for ic in range(2, 7):
# set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
# extract labels and calculate silhoutte score
labels = kmeans.labels_
print (ic, skmet.silhouette_score(df_fit, labels))

# Plot for four clusters
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(df_fit)

# extract labels and cluster centres
labels = kmeans.labels_ 
cen = kmeans.cluster_centers_

# Cluster plot by (K-means)
plt.figure(figsize=(6.0, 6.0))

# Individual colours can be assigned to symbols. The label l is used to the␣
#↪→select the
# l-th number from the colour table.
plt.scatter(df_fit["India"], df_fit["Canada"], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(4):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("India")
plt.ylabel("Canada")
plt.title("4 cluster Plot")
plt.show()