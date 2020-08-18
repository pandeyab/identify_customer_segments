#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[51]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from collections import Counter

#feature scaler module import, PCA import and Kmeans import
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[52]:


# Load in the general demographics data.
azdias = pd.read_csv("Udacity_AZDIAS_Subset.csv", delimiter = ';')

# Load in the feature summary file.
feat_info = pd.read_csv("AZDIAS_Feature_Summary.csv", delimiter = ';')


# In[53]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
print("size of demographic data",azdias.shape)
print("size of feature summary",feat_info.shape)

display("first 5 rows of azdias",azdias.head())
display("first 5 rows of feat_info",feat_info.head(10))


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[54]:


# Identify missing or unknown data values and convert them to NaNs.

print(azdias.isnull().sum().sum(), "no. of data are missing or unknown in azdias dataset")
display(azdias.isnull().sum())


# In[55]:


# Identify missing or unknown data values and convert them to NaNs.
for idx in range(len(feat_info)):
    miss_or_unknwn = feat_info.iloc[idx]['missing_or_unknown']
    miss_or_unknwn = ((miss_or_unknwn.strip('[')).strip(']')).split(',')
    miss_or_unknwn = [int(value) if (value!='X' and value!='XX' and value!='') else value for value in miss_or_unknwn]
    if miss_or_unknwn != ['']:
        azdias = azdias.replace({feat_info.iloc[idx]['attribute']: miss_or_unknwn}, np.nan)


# In[56]:


print(azdias.isnull().sum().sum(), "no. of data are missing or unknown in azdias dataset")
display(azdias.isnull().sum())


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[57]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.
null_values_percent =(azdias.isnull().sum()/azdias.shape[0]).sort_values(ascending=False)*100
null_values_percent


# In[58]:


# Investigate patterns in the amount of missing data in each column.
plt.hist(null_values_percent, bins=80)
plt.ylabel('Number of Columns')
plt.xlabel('Null Values percent')
plt.show()


# In[59]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
outlier_columns_lst = []
for column in range(azdias.shape[1]):
    current_column = null_values_percent.index[column]
    null_percent_column = null_values_percent[column]
    if null_percent_column > 20:
        outlier_columns_lst.append(current_column)

display("droppable columns",outlier_columns_lst)

#dropping those columns

azdias = azdias.drop(outlier_columns_lst, axis=1)


# In[60]:


#current shape
print("current size of demographic data",azdias.shape)
azdias.head(10)


# In[61]:


plt.hist((azdias.isnull().sum()/azdias.shape[0]).sort_values(ascending=False)*100, bins=80)
plt.ylabel('Number of Columns')
plt.xlabel('Null Values percent')
plt.show()

print(azdias.isnull().sum().sum(), "no. of data are missing or unknown in azdias dataset")


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# (Double click this cell and replace this text with your own text, reporting your observations regarding the amount of missing data in each column. Are there any patterns in missing values? Which columns were removed from the dataset?)
# 
# - We can see the "outlier_columns_lst" columns are having 20% more missing data than other columns so they are considered outliers and dropped , those are 
#  ['TITEL_KZ',
#  'AGER_TYP',
#  'KK_KUNDENTYP',
#  'KBA05_BAUMAX',
#  'GEBURTSJAHR',
#  'ALTER_HH']
#  
#  We can see in histogram that those columns are removed from dataset.
#  

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[62]:


# How much data is missing in each row of the dataset?
missing_data_rows = azdias.isnull().sum(axis=1)
display(missing_data_rows)

#plotting of missing rows data
plt.hist(missing_data_rows, bins=80)
plt.ylabel('Number of Rows')
plt.xlabel('Null Values percent')
plt.show()


# In[63]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
#data can be divided at threshold value of 30 as seen in plot
missing_data_aboveTh = azdias[azdias.isnull().sum(axis=1) > 30]
missing_data_belowTh = azdias[azdias.isnull().sum(axis=1) <= 30]

print('missing data in rows above 30 :', missing_data_aboveTh.shape[0])
print('missing data in rows below 30 :', missing_data_belowTh.shape[0])


# In[64]:


#selected few columns fro comparison
compare_columns =  ['SEMIO_DOM',  'SEMIO_RAT', 'SEMIO_TRADV', 'ANREDE_KZ', 'ZABEOTYP']


# In[65]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.
   
def distr_compare(column):
    fig = plt.figure(figsize=(14,6))
    
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Many missing rows from above th')
    sns.countplot(azdias.loc[missing_data_aboveTh.index,column])

    ax2 = fig.add_subplot(122)
    ax2.title.set_text('Few missing rows')
    sns.countplot(azdias.loc[~azdias.index.isin(missing_data_belowTh),column]);

    fig.suptitle(column)
    plt.show()

#plotting the comparison
for i in compare_columns:
    distr_compare(i)


# In[66]:


# selecting rows with high missing values
azdias_many_missing = azdias.iloc[missing_data_aboveTh.index]

print(f'Total rows in azdias dataset is {azdias.shape[0]}')


# In[67]:


# dropping rrows with high missing values
azdias = azdias[~azdias.index.isin(missing_data_aboveTh.index)]
azdias.head()

print(f'{len(azdias_many_missing)} rows greater than 30% in missing row values were dropped')
print(f'{azdias.shape[0]} rows are remaining')


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# (Double-click this cell and replace this text with your own text, reporting your observations regarding missing data in rows. Are the data with lots of missing values are qualitatively different from data with few or no missing values?)
# 
# - I selected these columns ['SEMIO_DOM',  'SEMIO_RAT', 'SEMIO_TRADV', 'ANREDE_KZ', 'ZABEOTYP']
# We can see that there are less data missing in rows from missing_data_belowTh i.e below 30 where each row was compared. More data loss is visible in dataset beyond 30, only a bit of match can be found with ANREDE_KZ if the values are scaled. SEMIO_DOM section has lots of missing data. Since there are considerable data loss beyond 30, I dropped those rows.

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[68]:


# How many features are there of each data type?
feat_info['type'].value_counts()


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[69]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
cat_columns = feat_info.loc[feat_info['type'] == 'categorical', 'attribute'].values


# In[70]:


# selecting the categorical columns in azdias now
cat_columns = [cat_column for cat_column in cat_columns if cat_column in azdias.columns] 

display(cat_columns)
display(azdias[cat_columns].nunique())


# In[71]:


# Re-encode categorical variable(s) to be kept in the analysis.
binary = []
col_multi = []
for column in cat_columns:
    if azdias[column].nunique() > 2:
        col_multi.append(column)
    else:
        binary.append(column)

print(" multi-level fetures: ",col_multi)
print()
print("binary (two-level) features: ", binary)


# In[72]:


azdias['OST_WEST_KZ'].unique()


# In[73]:


# dropping the multilevel categorical columns
azdias.drop(col_multi, axis=1, inplace=True)


# In[74]:


# checking unique values in categorical binary column
for col in binary:
    print(azdias[col].value_counts())


# In[75]:


azdias.loc[:, 'OST_WEST_KZ'].replace({'W':'0', 'O':'1'}, inplace=True)


# In[76]:


# checking unique values in categorical binary column
for col in binary:
    print(azdias[col].value_counts())


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding categorical features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# - Binary features are kept for analysis
# - Multi-level features are dropped off 
# - OST_WEST_KZ needed one hot enocoding, perfromed and kept in Binary features

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[77]:


#check mixed type var
catg_type_mix = feat_info[feat_info['type'] =='mixed']
catg_type_mix


# ### 1.18. PRAEGENDE_JUGENDJAHRE
# Dominating movement of person's youth (avantgarde vs. mainstream; east vs. west)
# - -1: unknown
# -  0: unknown
# -  1: 40s - war years (Mainstream, E+W)
# -  2: 40s - reconstruction years (Avantgarde, E+W)
# -  3: 50s - economic miracle (Mainstream, E+W)
# -  4: 50s - milk bar / Individualisation (Avantgarde, E+W)
# -  5: 60s - economic miracle (Mainstream, E+W)
# -  6: 60s - generation 68 / student protestors (Avantgarde, W)
# -  7: 60s - opponents to the building of the Wall (Avantgarde, E)
# -  8: 70s - family orientation (Mainstream, E+W)
# -  9: 70s - peace movement (Avantgarde, E+W)
# - 10: 80s - Generation Golf (Mainstream, W)
# - 11: 80s - ecological awareness (Avantgarde, W)
# - 12: 80s - FDJ / communist party youth organisation (Mainstream, E)
# - 13: 80s - Swords into ploughshares (Avantgarde, E)
# - 14: 90s - digital media kids (Mainstream, E+W)
# - 15: 90s - ecological awareness (Avantgarde, E+W)
# 
# ### 4.3. CAMEO_INTL_2015
# German CAMEO: Wealth / Life Stage Typology, mapped to international code
# - -1: unknown
# - 11: Wealthy Households - Pre-Family Couples & Singles
# - 12: Wealthy Households - Young Couples With Children
# - 13: Wealthy Households - Families With School Age Children
# - 14: Wealthy Households - Older Families &  Mature Couples
# - 15: Wealthy Households - Elders In Retirement
# - 21: Prosperous Households - Pre-Family Couples & Singles
# - 22: Prosperous Households - Young Couples With Children
# - 23: Prosperous Households - Families With School Age Children
# - 24: Prosperous Households - Older Families & Mature Couples
# - 25: Prosperous Households - Elders In Retirement
# - 31: Comfortable Households - Pre-Family Couples & Singles
# - 32: Comfortable Households - Young Couples With Children
# - 33: Comfortable Households - Families With School Age Children
# - 34: Comfortable Households - Older Families & Mature Couples
# - 35: Comfortable Households - Elders In Retirement
# - 41: Less Affluent Households - Pre-Family Couples & Singles
# - 42: Less Affluent Households - Young Couples With Children
# - 43: Less Affluent Households - Families With School Age Children
# - 44: Less Affluent Households - Older Families & Mature Couples
# - 45: Less Affluent Households - Elders In Retirement
# - 51: Poorer Households - Pre-Family Couples & Singles
# - 52: Poorer Households - Young Couples With Children
# - 53: Poorer Households - Families With School Age Children
# - 54: Poorer Households - Older Families & Mature Couples
# - 55: Poorer Households - Elders In Retirement
# - XX: unknown

# In[78]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.
# Adding a feature based on wealth
def wealth(x):
    if x // 10 ==1:
        return 1
    if x // 10 ==2:
        return 2
    if x // 10 ==3:
        return 3
    if x // 10 ==4:
        return 4
    if x // 10 ==5:
        return 5

# Adding a feature based on lfe stage
def life_stage(x):
    if x % 10 ==1:
        return 1
    if x % 10 ==2:
        return 2
    if x % 10 ==3:
        return 3
    if x % 10 ==4:
        return 4
    if x % 10 ==5:
        return 5

#coverting data type to numeric
azdias['CAMEO_INTL_2015'] = pd.to_numeric(azdias['CAMEO_INTL_2015'])

#adding new columns "WEALTH"  and 'LIFE_STAGE' 
azdias['WEALTH'] = azdias['CAMEO_INTL_2015'].apply(wealth)
azdias['LIFE_STAGE'] = azdias['CAMEO_INTL_2015'].apply(life_stage)

#dropping CAMEO_INTL_2015 from dataset
azdias.drop('CAMEO_INTL_2015', axis=1, inplace=True)


# In[79]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
def interval(x):
    if x in (1,2):
        return 1
    elif x in (3,4):
        return 2
    elif x in (5,6,7):
        return 3
    elif x in (8,9):
        return 4
    elif x in (10,11,12,13):
        return 5
    elif x in (14,15):
        return 6
    
def movement(x):
    if x in (2,4,6,7,9,11,13,15):
        return 0
    elif x in (1,3,5,8,10,12,14):
        return 1
    
# adding new columns "DECADES"  and 'MOVEMENTS' based on decade of birth and movementrespectively
azdias['DECADES'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(interval)
azdias['MOVEMENTS'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(movement)


# In[80]:


#dropping PRAEGENDE_JUGENDJAHRE from dataset
azdias['DECADES'].value_counts().sort_index()

azdias = azdias.drop('PRAEGENDE_JUGENDJAHRE',axis=1)


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding mixed-value features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# The two mixed-type features are taken here for engineering:
# 
# PRAEGENDE_JUGENDJAHRE
# CAMEO_INTL_2015
# 
# Process:
# 1. Two new feature columns created and values from inital columns copied
# 2. two different function are  created for each feature to map feature values 
# 3. replaced the dictionaries to new feature column
# 4. dropped PRAEGENDE_JUGENDJAHRE and  CAMEO_INTL_2015 from dataset

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[81]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here.
np.unique(azdias.dtypes.values)


# In[82]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.

azdias.loc[:, azdias.dtypes == 'O']


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[83]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    #getting feature summary
    feat_info = pd.read_csv("AZDIAS_Feature_Summary.csv", delimiter = ';')
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...    
    for i,item in zip(range(len(df)), df.iteritems()):
        miss_unknwn = feat_info['missing_or_unknown'][i]
        miss_unknwn = miss_unknwn[1:-1].split(',')
        column_name = item[0]
        if miss_unknwn != ['']:
            miss = [x if x in ['X','XX'] else int(x) for x in miss_unknwn]
            
            df[column_name] = df[column_name].replace(miss,np.nan)
            
    # removing selected columns i.e. rows having > 20% of missing values
    df = df.drop(['AGER_TYP','GEBURTSJAHR','TITEL_KZ','ALTER_HH','KK_KUNDENTYP','KBA05_BAUMAX'],axis=1)
    

    # removing selected rows i.e. rows having > 30% of missing values
    missing_row_values = df.isnull().sum(axis=1)
    missing_row_values.value_counts().sort_index(ascending=False,inplace=True)
    

    # dividing the rows into 2 subsets based on having missing value less than 30% or not<
    missing_data_belowTh = df[df.isnull().sum(axis=1)<30].reset_index(drop=True)
    missing_data_aboveTh = df[df.isnull().sum(axis=1)>=30].reset_index(drop=True)
    
    # selecting rows with high missing values
    df_many_missing = df.iloc[missing_data_aboveTh.index]
    print(f'Total rows in dataset is {df.shape[0]}')
    
    # dropping rows with high missing values
    df = df[~df.index.isin(missing_data_aboveTh.index)]
    print(f'{len(df_many_missing)} rows greater than 30% in missing row values were dropped')
    print(f'{df.shape[0]} rows are remaining')
       
    
    # select, re-encode, and engineer column values.
    # dropping the multilevel categorical columns
    for column in df.columns:
        if column in col_multi:
            df.drop(column, axis=1, inplace=True)
            
   
    #df.drop(col_multi, axis=1, inplace=True)
    
    # encoding the 'OST_WEST_KZ' binary categorical column
    df.loc[:, 'OST_WEST_KZ'].replace({'W':'0', 'O':'1'}, inplace=True) 
    
    # Engineering(converting) "PRAEGENDE_JUGENDJAHRE" and 'CAMEO_INTL_2015'into two new variables each     
    # adding 2 new columns "DECADES" and 'MOVEMENTS' based on decade of birth and movement
    df['DECADES'] = df['PRAEGENDE_JUGENDJAHRE'].apply(interval)
    df['MOVEMENTS'] = df['PRAEGENDE_JUGENDJAHRE'].apply(movement)
    # Dropping 'PRAEGENDE_JUGENDJAHRE' column from the dataframe
    df = df.drop('PRAEGENDE_JUGENDJAHRE',axis=1)
    
        
    # Adding 2 new features based on wealth and life stage and dropping 'CAMEO_INTL_2015'
    df['CAMEO_INTL_2015'] = pd.to_numeric(df['CAMEO_INTL_2015'])
    
    df['WEALTH'] = df['CAMEO_INTL_2015'].apply(wealth)
    df['LIFE_STAGE'] = df['CAMEO_INTL_2015'].apply(life_stage)
    df.drop('CAMEO_INTL_2015', axis=1, inplace=True)
    
    
    col_list = df.columns
       
    # impute NaN 
    imputer = Imputer(strategy='mean', axis=0)
    df_imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=col_list)
    
    print('Done')

    # Return the cleaned dataframe.
    return df_imputed, df_many_missing


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[84]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
azdias_copy=azdias.copy()
col_list = azdias_copy.columns

imputer = Imputer(strategy='mean', axis=0)
azdias_imputed = imputer.fit_transform(azdias_copy)

# changing to dataframe again and checking for missing value
azdias_imputed = pd.DataFrame(azdias_imputed, columns= col_list)
azdias_imputed.isnull().sum()
display(azdias_imputed.isnull().sum().sum())


# In[85]:


# Apply feature scaling to the general population demographics data.

scaler = StandardScaler()
azdias_scaled = scaler.fit_transform(azdias_imputed)
azdias_scaled = pd.DataFrame(azdias_scaled, columns= col_list)
azdias_scaled.head()


# ### Discussion 2.1: Apply Feature Scaling
# 
# (Double-click this cell and replace this text with your own text, reporting your decisions regarding feature scaling.)
# 
# - StandardScaler module is used for feature scaling on "azdias_imputed" dataset, which doesn't have any NaN values, to scale all numerical data to mean 0 and standard deviation of 1.

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[86]:


# Apply PCA to the data.
pca = PCA()
pca_belowTh = pca.fit_transform(azdias_scaled)
pd.DataFrame(pca_belowTh)


# In[87]:


plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.title("Variance explained by each component")
plt.xlabel("Principal component")
plt.ylabel("Ratio of variance explained")
plt.show()


# In[88]:


plt.plot(range(len(pca.explained_variance_ratio_)),np.cumsum(pca.explained_variance_ratio_), '-')
plt.title("Cumulative Variance Explained")
plt.xlabel("Number of Components")
plt.ylabel("Ratio of variance explained")
plt.show()


# In[89]:


# Re-apply PCA to the data while selecting for number of components to retain.
pca = PCA(30)
pca_features2 = pca.fit(azdias_scaled)

plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.title("Variance explained by each component")
plt.xlabel("Principal component")
plt.ylabel("Ratio of variance explained")
plt.show()


# In[90]:


plt.plot(range(len(pca.explained_variance_ratio_)),np.cumsum(pca.explained_variance_ratio_), '-')
plt.title("Cumulative Variance Explained")
plt.xlabel("Number of Components")
plt.ylabel("Ratio of variance explained")
plt.show()


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding dimensionality reduction. How many principal components / transformed features are you retaining for the next step of the analysis?)
# 
# - I applied PCA module to apply component analysis to find the variance. There are close to 64 componenets initially without giving any components, where variance explained is about 100%, but I am going ahead with top 30 components becuse after 30th component variance explained decreases faster as seen in plot, and with these components, I am getting cumulative variance of nearly 88%..

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[91]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.
pca_30 = PCA(n_components=30)
pca_belowTh_2 = pca_30.fit_transform(azdias_scaled)

def pcaWeights(k, pca):
    df = pd.DataFrame(pca.components_, columns=list(azdias_scaled.columns))
    weights = df.iloc[k].sort_values(ascending=False)
    return weights

weight_0 = pcaWeights(0, pca_30)
print("linked values\n", weight_0)


# In[92]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.
weight_1 = pcaWeights(1, pca_30)
print("linked values\n", weight_1)


# In[93]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

weight_2 = pcaWeights(2, pca_30)
print("linked values\n", weight_2)


# ### Discussion 2.3: Interpret Principal Components
# 
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)
# 
# - After seeing interpret positive and negative values of weight, I could get some relationships between those values. For example, in the 1st principle component 'WEALTH' with positive weight as opposed 'MOBI_REGIO' with negative weight. if 'WEALTH' values increases , the 'MOBI_REGIO' values goes more negative. The same pattern observed in 2nd and 3rd PC.
# 
# PC 1-
# 
# Important Negative features are: 
# 
#  KBA05_ANTG1 (Number of 1-2 family houses in the microcell), MOBI_REGIO (Movement patterns - region feature), PLZ8_ANTG1 (Number of 1-2 family houses in the PLZ8 region), FINANZ_MINIMALIST (MINIMALIST: low financial interest)
# 
# Important Positive features are : 
# 
# HH_EINKOMMEN_SCORE (Estimated household net income), PLZ8_ANTG3 (Number of 6-10 family houses in the PLZ8 region), WEALTH (Wealth of household).
# 
# I can see from above data that such factors are the most crucials such as movement level in the region, number of 1-2 houses in the microcell and region, income, number 6-10 family houses in the region and wealth. It appears that the increases in 1-2 family houses will lead to decreases 6-10 family houses which is logical and confirmed by PCA. Increases in number of 1-2 family in microcell affects in increases of 1-2 family in the region.
# 
# 
# PC 2-
# 
# Important Negative features are: 
# 
# DECADE (person's youth in decades), FINANZ_SPARER (financial typology - money saver), SEMIO_REL (religious)
# 
# Important Positive features are: 
# 
# ALTERSKATEGORIE_GROB (Estimated age), FINANZ_VORSORGER (financial typology - be prepared), SEMIO_ERL (event-oriented)
# 
# I can see from above data that such factors are the most crucials such as youth in decades/age, financial typology, religious factor and energey consumption typology. It is also visible here that the more increases money saver then be prepared decreases as they are negatively correlated between each other. I can also conclude that religious factor is in accordance with money saver typology, from life experience I can also confirm that major religious are not supporting money wasting.
# 
# 
# PC 3-
# 
# Important Negative features are: 
# 
# SEMIO_KAEM (personal typology -combative attitude), SEMIO_DOM (dominant-minded), ANREDE_KZ (Gender)
# 
# Important Positive features are: 
# 
# SEMIO_VERT (personal typology - dreamful), SEMIO_SOZ (socially-minded), SEMIO_FAM (family-minded).
# 
# Both gender and personal typology closely related to each other as we can confirm with above data PC3. From the same negative direction we see the correlation: gender factor, combative-dominant are in sync where they increase together. On the positive side dreamful, socially-minded and family-minded are there which are just opposite of negatives.

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[94]:


#defining a function to perform k-means clustering
def k_means_clustering(pcadata, n_cluster):
    '''
    input : pca data, no. of clusters
    output : avg difference [score]
    
    '''
    kmeans = KMeans(n_clusters = n_cluster)
    model_fit = kmeans.fit(pcadata)
    score = np.abs(model_fit.score(pcadata))
    
    return score


# In[95]:


scores = []
n_clusters=[2, 4, 6, 8, 10, 12, 14, 16]

for i in n_clusters:
    scores.append(k_means_clustering(pca_belowTh_2, i))


# In[97]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.
plt.figure(figsize=(15, 10))
plt.plot(n_clusters, scores, linestyle='--', marker='o', color='r')
plt.xlabel('no. of  K clusters')
plt.ylabel('with-in cluster distance')
plt.title('Average Distance vs. K')


# In[98]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
startTime = time() 

kmeans_fit = KMeans(n_clusters = 14).fit(pca_belowTh_2)
clusPred = kmeans_fit.predict(pca_belowTh_2)

endTime = time() 
print("elapsed time is ",endTime - startTime, "sec")


# ### Discussion 3.1: Apply Clustering to General Population
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)
# 
# - Seems K=14 is the elbow in this K-means clustering plot, so I have used K=14 for prediction.

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[100]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv',delimiter=';')
customers.head()


# In[101]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.

customers_updated, customer_many_missing = clean_data(customers)


# In[102]:


customers_updated.shape


# In[103]:


#find out any difference between General data and customer data
list(set(azdias.columns) - set(customers_updated))


# In[104]:


# Apply scaler
customers_updated_scaled = scaler.transform(customers_updated)


# In[105]:


#transform the customers data using pca object
customers_updated_pca = pca_30.transform(customers_updated_scaled)


# In[106]:


#predict clustering using the kmeans object
predict_customers = KMeans(n_clusters = 14).fit(pca_belowTh_2).predict(customers_updated_pca)


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[110]:


#count the elements from predicted cluster
Counter(clusPred)


# In[111]:


#count the elements from customer prdection
Counter(predict_customers)


# In[126]:


#plotting the General cluster and Customer clsuter to observe the differences
figure, axs = plt.subplots(nrows=1, ncols=2, figsize = (18,10))
figure.subplots_adjust(hspace = 1, wspace=.3)

sns.countplot(predict_customers, ax=axs[0] )
axs[0].set_title('Customer Clusters', fontsize=15)
plt.xlabel("Number of Customer Clusters")
plt.ylabel("Customer count")

sns.countplot(clusPred, ax=axs[1])
axs[1].set_title('General Clusters', fontsize=15)
plt.xlabel("Number of General Clusters")
plt.ylabel("General count")


# In[127]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?
data = scaler.inverse_transform(pca_30.inverse_transform(customers_updated_pca[np.where(predict_customers==3)]))
df = pd.DataFrame(data=data, index=np.array(range(0, data.shape[0])), columns=customers_updated.columns)
df.head(10)


# In[128]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?

data = scaler.inverse_transform(pca_30.inverse_transform(customers_updated_pca[np.where(predict_customers==2)]))
df = pd.DataFrame(data=data, index=np.array(range(0, data.shape[0])), columns=customers_updated.columns)
df.head(10)


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)
# 
# - We could see that in Customer cluster, the cluster point 3, 9 & 10 are the highly likely cusomer segments because of the larger customer data are present there. 
# cluster point 2, 5 & 8 are the less likely cusomer segments as general population data dominates these clusters.
# 
# Cluster 3 is overrepresented in the customers data compared to general population data.
# Cluster 2 is underrepresented in the customers data.
# 

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[129]:


get_ipython().getoutput('jupyter nbconvert *.ipynb')


# In[ ]:




