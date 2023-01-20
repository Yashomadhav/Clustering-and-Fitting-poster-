# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:19:25 2023

@author: Yashomadhav Mudgal
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import err_ranges as err
import sklearn.metrics as skmet
import seaborn as sns
import sklearn.preprocessing as prep
from sklearn import cluster
import sklearn.cluster as cluster

def get_data_frames(filename,countries,indicator):
    '''
    This function returns two dataframes one with countries as column and other 
    one years as column.
    It tanspose the dataframe and converts rows into column and column into 
    rows of specific column and rows.
    It takes three arguments defined as below. 

    Parameters
    ----------
    filename : Text
        Name of the file to read data.
    countries : List
        List of countries to filter the data.
    indicator : Text
        Indicator Code to filter the data.

    Returns
    -------
    df_countries : DATAFRAME
        This dataframe contains countries in rows and years as column.
    df_years : DATAFRAME
        This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by countries
    df = df.loc[df['Country Name'].isin(countries)]
    # To filter data by indicator code.
    df = df.loc[df['Indicator Code'].eq(indicator)]
    
    # Using melt function to convert all the years column into rows as 1 column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Country Code']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code']
                          ,'Country Name').reset_index()
    
    df_countries = df
    df_years = df2
    
    # Cleaning data droping nan values.
    df_countries.dropna()
    df_years.dropna()
    
    return df, df2

def get_data_frames1(filename,countries,indicator):
    '''
    This function returns two dataframes one with countries as column and other 
    one years as column.
    It tanspose the dataframe and converts rows into column and column into 
    rows of specific column and rows.
    It takes three arguments defined as below. 

    Parameters
    ----------
    filename : Text
        Name of the file to read data.
    countries : List
        List of countries to filter the data.
    indicator : Text
        Indicator Code to filter the data.

    Returns
    -------
    df_countries : DATAFRAME
        This dataframe contains countries in rows and years as column.
    df_years : DATAFRAME
        This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by countries
    # df = df.loc[df['Country Name'].isin(countries)]
    # To filter data by indicator code.
    df = df.loc[df['Indicator Code'].isin(indicator)]
    
    # Using melt function to convert all the years column into rows as 1 column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Indicator Name']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value',['Years','Country Name','Country Code']
                          ,'Indicator Code').reset_index()
    
    
    
    # Cleaning data droping nan values.
    df.dropna()
    df2.dropna()
    
    return df, df2


def poly(x, a, b, c, d):
    '''
    Cubic polynominal for the fitting
    '''
    y = a*x*3 + b*x*2 + c*x + d
    return y

def exp_growth(t, scale, growth):
    ''' 
    Computes exponential function with scale and growth as free parameters
    '''
    f = scale * np.exp(growth * (t-1960))
    return f

def logistics(t, scale, growth, t0):
    ''' 
    Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    '''
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def norm(array):
    '''
    Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe
    '''
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled

def norm_df(df, first=0, last=None):
    '''
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    '''
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df


def map_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns␣
    ↪→in the dataframe.
    Input:
    df: pandas DataFrame
    size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='magma')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
           
   





# Data fitting for China Population with prediction


countries = ['Germany','Australia','United States','China','United Kingdom']
# calling functions to get dataframes and use for plotting graphs.
df, df2 = get_data_frames('API_19_DS2_en_csv_v2_4773766.csv',countries,
                             'SP.POP.TOTL')

df2['Years'] = df2['Years'].astype(int)

popt, covar = curve_fit(exp_growth, df2['Years'], df2['China'])
print("Fit parameter", popt)
# use *popt to pass on the fit parameters
df2['china_exp'] = exp_growth(df2['Years'], *popt)
plt.figure()
plt.plot(df2['Years'], df2["China"], label='data')
plt.plot(df2['Years'], df2['china_exp'], label='fit')
plt.legend()
plt.title("First fit attempt")
plt.xlabel("Year")
plt.ylabel("China Population")
plt.show()

# find a feasible start value the pedestrian way
# the scale factor is way too small. The exponential factor too large.
# Try scaling with the 1950 population and a smaller exponential factor
# decrease or increase exponential factor until rough agreement is reached
# growth of 0.07 gives a reasonable start value
popt = [7e8, 0.01]
df2['china_exp'] = exp_growth(df2['Years'], *popt)
plt.figure()
plt.plot(df2['Years'], df2['China'], label='data')
plt.plot(df2['Years'], df2['china_exp'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("China Population")
plt.title("Improved start value")
plt.show()

popt = [1135185000, 0.02, 1990]
df2['china_log'] = logistics(df2['Years'], *popt)
plt.figure()
plt.plot(df2['Years'], df2['China'], label='data')
plt.plot(df2['Years'], df2['china_log'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("China Population")
plt.title("Improved start value")
plt.show()

popt, covar = curve_fit(logistics,  df2['Years'],df2['China'],
p0=(6e9, 0.05, 1990.0))
print("Fit parameter", popt)
df2['china_log'] = logistics(df2['Years'], *popt)
plt.figure()
plt.plot(df2['Years'], df2['China'], label='data')
plt.plot(df2['Years'], df2['china_log'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("China Population")
plt.title("Logistic Function")


# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)

low, up = err.err_ranges(df2['Years'], logistics, popt, sigma)
plt.figure()
plt.title("logistics function")
plt.plot(df2['Years'], df2['China'], label='data')
plt.plot(df2['Years'], df2['china_log'], label='fit')
plt.fill_between(df2['Years'], low, up, alpha=0.7)
plt.legend()
plt.xlabel("Year")
plt.ylabel("China Population")
plt.show()

print("Forcasted population")
low, up = err.err_ranges(2030, logistics, popt, sigma)
print("2030 between ", low, "and", up)
low, up = err.err_ranges(2040, logistics, popt, sigma)
print("2040 between ", low, "and", up)
low, up = err.err_ranges(2050, logistics, popt, sigma)
print("2050 between ", low, "and", up)

print("Forcasted population")
low, up = err.err_ranges(2030, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)




# Data fitting with ouliners for Total Population

# List of countries 
countries = ['Germany','Australia','United States','China','United Kingdom']
# calling functions to get dataframes and use for plotting graphs.
df, df2 = get_data_frames('API_19_DS2_en_csv_v2_4773766.csv',countries,
                             'SP.POP.TOTL')


df.dropna()
df2.dropna()


df2['Years'] = df2['Years'].astype(int)
x = df2['Years'].values
y = df2['Australia'].values 
z = df2['United States'].values
w = df2['United Kingdom'].values 

param, covar = curve_fit(poly, x, y)
# produce columns with fit values
df2['fit'] = poly(df2['Years'], *param)
# calculate the z-score
df2['diff'] = df2['Australia'] - df2['fit']
sigma = df2['diff'].std()
print("Number of points:", len(df2['Years']), "std. dev. =", sigma)
# calculate z-score and extract outliers
df2["zscore"] = np.abs(df2["diff"] / sigma)
df2 = df2[df2["zscore"] < 3.0].copy()
print("Number of points:", len(df2['Years']))

param1, covar1 = curve_fit(poly, x, z)
param2, covar2 = curve_fit(poly, x, w)


# Data Fitting (line plot)
plt.figure()
plt.title("Total Popolation (Data Fitting)")
plt.scatter(x, y, label='Australia')
plt.scatter(x, z, label='United States')
plt.scatter(x, w, label='United Kingdom')
plt.xlabel('Years')
plt.ylabel('Total Population')
x = np.arange(1960,2021,10)
plt.plot(x, poly(x, *param), 'k')
plt.plot(x, poly(x, *param1), 'k')
plt.plot(x, poly(x, *param2), 'k')
plt.xlim(1960,2021)
plt.legend()
plt.show()



# Bar plot for Terrestrial protected areas


df, df2 = get_data_frames('API_19_DS2_en_csv_v2_4773766.csv',countries
                             ,'ER.LND.PTLD.ZS')
num= np.arange(5)
width= 0.2
# Select specific years data 
df2 = df2.loc[df2['Years'].isin(['2016','2017','2018','2019','2020'])]
years = df2['Years'].tolist() 

#Ploting data on bar chart  
plt.figure(dpi=144)
plt.title('Terrestrial protected areas (% of total land area) ')
plt.bar(num,df2['Germany'], width, label='Germany')
plt.bar(num+0.2, df2['Australia'], width, label='Australia')
plt.bar(num-0.2, df2['United States'], width, label='United States')
plt.bar(num-0.4, df2['China'], width, label='China')
plt.xticks(num, years)
plt.xlabel('Years')
plt.ylabel('Terrestrial protected areas ')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()




# Bar plot for Electric Power Consumption


df, df2 = get_data_frames('API_19_DS2_en_csv_v2_4773766.csv',countries, 'EG.USE.ELEC.KH.PC')# Extracting data from files
# df2 = df2.loc[df2['Indicator Code'].isin(['EG.USE.ELEC.KH.PC'])]
df2 = df2.loc[df2['Years'].isin(['1970','1980','1990','2000','2010'])]
df2.dropna()


num = np.arange(5)
width = 0.2
years = df2['Years'].tolist() # Used (tolist) to convert array values to list

  
plt.figure()
plt.title('Electric power consumption (kWh per capita)  ')
plt.bar(num,df2['Australia'], width, label ='Australia')
plt.bar(num+0.2, df2['United States'], width, label ='United States')
plt.bar(num-0.2, df2['China'], width, label='China')
plt.bar(num+0.4, df2['United Kingdom'], width, label='United Kingdom')

plt.xticks(num, years) # This is for showing years in x asis
plt.xlabel('Years')
plt.ylabel('Electric power consumption')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()




# Clustering Analysis (k-means Clustering)

countries = ['Germany','Australia','United States','China','United Kingdom']
df2, df3 = get_data_frames1('API_19_DS2_en_csv_v2_4773766.csv',countries
                             ,['SP.POP.GROW','SP.POP.TOTL','SP.URB.GROW'
                               ,'SP.URB.TOTL'])


df3 = df3.loc[df3['Years'].eq('2015')]
df3 = df3.loc[~df3['Country Code'].isin(['XKX','MAF'])]

df3.dropna()


# Heat Map Plot
map_corr(df3)
plt.show()

# Scatter Matrix Plot
pd.plotting.scatter_matrix(df3, figsize=(9.0, 9.0))
plt.suptitle("Scatter Matrix Plot For All Countries", fontsize=20)
plt.tight_layout() # helps to avoid overlap of labels
plt.show()


# extract columns for fitting
df_fit = df3[["SP.POP.GROW", "SP.URB.GROW"]].copy()
# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_fish. This make the plots with the
# original measurements
df_fit = norm_df(df_fit)
(df_fit.describe())



for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))


# Plot for four clusters
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fit)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))

# Individual colours can be assigned to symbols. The label l is used to the
# select the l-th number from the colour table.
plt.scatter(df_fit["SP.POP.GROW"], df_fit["SP.URB.GROW"], c=labels
            , cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(3):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Population Growth")
plt.ylabel("Urban Population Growth")
plt.title("3 Clusters For All Countries")
plt.show()


