---
layout: article
title: Cleaning data in Python 
key: 20200610
tags: Python, Data-science
pageview: false
modify_date: 2020-06-11
aside:
  toc: true
---

From DataCamp.

<!--more-->
## 1. Common data Problems
### Common data types
- Numeric data types
- Text
- Dates

### Data type constrains
Manipulating and analyzing data with incorrect data types could lead to compromised analysis as you go along the data science workflow.

When working with new data, we could use the `.dtypes` attribute or the `.info()` method. Often times, you'll run into columns that should be converted to different data types before starting any analysis.

### To describe data and check data types:
The bicycle ride sharing data in San Francisco, `ride_sharing`, contains information on the start and end stations, the trip duration, and some user information for a bike sharing service.

The excise will 
- Print the information of `ride_sharing`.
Use `.describe()` to print the summary statistics of the `user_type` column from `ride_sharing`.

```py
# Print the information of ride_sharing
print(ride_sharing.info())

# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())
```

### Summing strings and concatenating numbers
Another common data type problem is importing what should be numerical values as strings, as mathematical operations such as summing and multiplication lead to string concatenation, not numerical outputs.

This exercise will convert the string column `duration` to the type `int`. First, strip `"minutes"` from the column in order to make sure `pandas` reads it as numerical. The `pandas` package has been imported as `pd`.

- Use the `.strip()` method to strip `duration` of `"minutes"` and store it in the `duration_trim` column.
- Convert `duration_trim` to `int` and store it in the `duration_time column`.
- Write an `assert` statement that checks if `duration_time`'s data type is now an `int`.
- Print the average ride duration.

```py
# Strip duration of minutes
ride_sharing['duration_trim'] = ride_sharing['duration'].str.strip('minutes')

# Convert duration to integer
ride_sharing['duration_time'] = ride_sharing['duration_trim'].astype('int')

# Write an assert statement making sure of conversion
assert ride_sharing['duration_time'].dtype == 'int'

# Print formed columns and calculate average ride duration 
print(ride_sharing[['duration','duration_trim','duration_time']])
print(ride_sharing['duration_time'].mean())
```
### Data range constrains
Sometimes there might show up values that is out of the data range. For example, a future time included in the time point; or six stars in a five-star-system.

Ways to deal with it:
-  Drop values using filtering: `movies = movies[movies['avg_rating'] <= 5]`
-  Drop values using: `.drop()`: `movies.drop(movies[movies['avg_rating'] > 5].index, inplace = True)`
-  Assert results: `assert movies['avg_rating'].max() <= 5`

### Tire size constraints
Bicycle tire sizes could be either 26″, 27″ or 29″ and are here correctly stored as a categorical value. In an effort to cut maintenance costs, the ride sharing provider decided to set the maximum tire size to be 27″.

In this exercise, the `tire_sizes` column has the correct range by first converting it to an integer, then setting and testing the new upper limit of 27″ for tire sizes.

- Convert the `tire_sizes` column from category to `'int'`.
- Use `.loc[]` to set all values of tire_sizes above 27 to 27.
- Reconvert back `tire_sizes` to `'category`' from `int`.
- Print the description of the `tire_sizes`.

```py
# Convert tire_sizes to integer
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('int')

# Set all values above 27 to 27
ride_sharing.loc[ride_sharing['tire_sizes'] > 27, 'tire_sizes'] = 27

# Reconvert tire_sizes back to categorical
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('category')

# Print tire size description
print(ride_sharing['tire_sizes'].describe())
```

### Back to the future
A new update to the data pipeline feeding into the `ride_sharing` DataFrame has been updated to register each ride's date. This information is stored in the `ride_date` column of the type `object`, which represents strings in `pandas`.

A bug was discovered which was relaying rides taken today as taken next year. To fix this, you will find all instances of the `ride_date` column that occur anytime in the future, and set the maximum possible value of this column to today's date. Before doing so, you would need to convert `ride_date` to a `datetime` object.

The `datetime` package has been imported as `dt`, alongside all the packages you've been using till now.

```py
import datetime as dt
import pandas as pd

# check data types
ride_sharing['ride_date'].dtypes

# Convert ride_date to datetime
ride_sharing['ride_dt'] = pd.to_datetime(ride_sharing['ride_date'])

# Save today's date
today = dt.date.today()

# Set all in the future to today's date
ride_sharing.loc[ride_sharing['ride_dt'] > today, 'ride_dt'] = today

# Print maximum of ride_dt column
print(ride_sharing['ride_dt'].max())
```

## Duplications
**How big is your subset?**
You have the following `loan` DataFrame which contains loan and credit score data for consumers, and some metadata such as their first and last names. You want to find both complete and incomplete duplicates using `.duplicated()`.

first_name | last_name | credit_score	| has_loan
Justin	| Saddlemeyer	| 600	| 1
Hadrien	| Lacroix |	450	| 0

Choose the correct usage of `.duplicated()` below:

- `loans.duplicated()`. Because the default method returns both complete and incomplete duplicates. [Wrong](red) :x:
- `loans.duplicated(subset = 'first_name')`. Because constraining the duplicate rows to the first name lets me find incomplete duplicates as well. [Wrong](red) :x:
- `loans.duplicated(subset = ['first_name', 'last_name'], keep = False)`. Because subsetting on consumer metadata and not discarding any duplicate returns all duplicated rows. [Right](blue) :white_check_mark:
- `loans.duplicated(subset = ['first_name', 'last_name'], keep = 'first')`. Because this drops all duplicates. [Wrong](red) :x:
  
### Fiding duplicatess
A new update to the data pipeline feeding into `ride_sharing` has added the `ride_id` column, which represents a unique identifier for each ride.

The update however coincided with radically shorter average ride duration times and irregular user birth dates set in the future. Most importantly, the number of rides taken has increased by 20% overnight, leading you to think there might be both complete and incomplete duplicates in the `ride_sharing` DataFrame.

In this exercise, you will confirm this suspicion by finding those duplicates. A sample of `ride_sharing` is in your environment, as well as all the packages you've been working with thus far.

- Find duplicated rows of `ride_id` in the `ride_sharing` DataFrame while setting `keep` to `False`.
- Subset `ride_sharing` on `duplicates` and sort by `ride_id` and assign the results to `duplicated_rides`.
- Print the `ride_id`, `duration` and `user_birth_year` columns of `duplicated_rides` in that order.

```py
# Find duplicates
duplicates = ride_sharing.duplicated(subset='ride_id', keep=False)

# Sort your duplicated rides
duplicated_rides = ride_sharing[duplicates].sort_values('ride_id')

# Print relevant columns of duplicated_rides
print(duplicated_rides[['ride_id','duration','user_birth_year']])
```

### Treating duplicates
In the last exercise, you were able to verify that the new update feeding into `ride_sharing` contains a bug generating both complete and incomplete duplicated rows for some values of the `ride_id` column, with occasional discrepant values for the `user_birth_year` and `duration` columns.

In this exercise, you will be treating those duplicated rows by first dropping complete duplicates, and then merging the incomplete duplicate rows into one while keeping the average `duration`, and the minimum `user_birth_year` for each set of incomplete duplicate rows.

- Drop complete duplicates in `ride_sharing` and store the results in `ride_dup`.
- Create the `statistics` dictionary which holds minimum aggregation for `user_birth_year` and mean aggregation for `duration`.
- Drop incomplete duplicates by grouping by `ride_id` and applying the aggregation in `statistics`.
- Find duplicates again and run the `assert` statement to verify de-duplication.

```py
# Drop complete duplicates from ride_sharing
ride_dup = ride_sharing.drop_duplicates()

# Create statistics dictionary for aggregation function
statistics = {'user_birth_year': 'min', 'duration': 'mean'}

# Group by ride_id and compute new statistics
ride_unique = ride_dup.groupby('ride_id').agg(statistics).reset_index()

# Find duplicated values again
duplicates = ride_unique.duplicated(subset = 'ride_id', keep = False)
duplicated_rides = ride_unique[duplicates == True]

# Assert duplicates are processed
assert duplicated_rides.shape[0] == 0
```

## 2. Text and categorical data problems

Different types of constraints:
- Data type constraints: 数据类型问题
- Data range constraints: 数值范围问题
- Uniqueness constraints: 重复值问题
- Membership contstraints: 资格问题
- 处理方式问题
- 分类问题
- 格式转化问题
- 缺失值问题
- 不同表格合并资格问题

Membership constraints: when recording content that should not exist. F. eks. when recording blood type, misspell the type from A+ to Z+.
Other examples:
- A `has_loan` column with the value 12.
- A `day_of_week` column with the value "Satermonday".
- A `month` column with the value 14. 
- A `GPA` column containing a "Z grade".

### Finding consistency

In this exercise and throughout this chapter, we will be working with the `airlines` DataFrame which contains survey responses on the San Francisco Airport from airline customers.

The DataFrame contains flight metadata such as the airline, the destination, waiting times as well as answers to key questions regarding cleanliness, safety, and satisfaction. Another DataFrame named `categories` was created, containing all correct possible values for the survey columns.

In this exercise, we will use both of these DataFrames to find survey answers with inconsistent values, and drop them, effectively performing an outer and inner join on both these DataFrames as seen in the video exercise. The `pandas` package has been imported as `pd`, and the `airlines` and `categories` DataFrames are in your environment.

- Print the `categories` DataFrame and take a close look at all possible correct categories of the survey columns.
- Print the unique values of the survey columns in `airlines` using the `.unique()` method.

```py
# Print categories DataFrame
print(categories)

# Print unique values of survey columns in airlines
print('Cleanliness: ', airlines['cleanliness'].unique(), "\n")
print('Safety: ', airlines["safety"].unique(), "\n")
print('Satisfaction: ', airlines['satisfaction'].unique(), "\n")
```

The output looks like this:

```
<script.py> output:
          cleanliness           safety          satisfaction
    0           Clean          Neutral        Very satisfied
    1         Average        Very safe               Neutral
    2  Somewhat clean    Somewhat safe    Somewhat satisfied
    3  Somewhat dirty      Very unsafe  Somewhat unsatisfied
    4           Dirty  Somewhat unsafe      Very unsatisfied
    Cleanliness:  [Clean, Average, Unacceptable, Somewhat clean, Somewhat dirty, Dirty]
    Categories (6, object): [Clean, Average, Unacceptable, Somewhat clean, Somewhat dirty, Dirty] 
    
    Safety:  [Neutral, Very safe, Somewhat safe, Very unsafe, Somewhat unsafe]
    Categories (5, object): [Neutral, Very safe, Somewhat safe, Very unsafe, Somewhat unsafe] 
    
    Satisfaction:  [Very satisfied, Neutral, Somewhat satisfied, Somewhat unsatisfied, Very unsatisfied]
    Categories (5, object): [Very satisfied, Neutral, Somewhat satisfied, Somewhat unsatisfied,
                             Very unsatisfied] 
```      

Take a look at the output. Out of the cleanliness, safety and satisfaction columns, which one has an inconsistent category and what is it?
- `cleanliness` because it has an `Unacceptable` category. [Right](blue) :white_check_mark:
- `cleanliness` because it has a `Terribly dirty` category.  [Wrong](red) :x:
- `satisfaction` because it has a `Very satisfied` category.  [Wrong](red) :x:
- `safety` because it has a `Neutral` category.  [Wrong](red) :x:

Next, find the column with different values using `set()` and `difference`:
- Create a set out of the `cleanliness` column in `airlines`-dataset using `set()` and find the inconsistent category by finding the **difference** in the `cleanliness` column of `categories`-dataset.
- Find rows of `airlines` with a `cleanliness` value not in `categories` and print the output.
- Print the rows with the consistent categories of `cleanliness` only.

```py
# Find the cleanliness category in airlines not in categories
cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])

# Find rows with that category
cat_clean_rows = airlines['cleanliness'].isin(cat_clean)

# Print rows with inconsistent category
print(airlines[cat_clean_rows])

# Print rows with consistent categories only
print(airlines[~cat_clean_rows])
```

And this gives the following output when exploring the data:

```
In [1]: categories
Out[1]:
      cleanliness           safety          satisfaction
0           Clean          Neutral        Very satisfied
1         Average        Very safe               Neutral
2  Somewhat clean    Somewhat safe    Somewhat satisfied
3  Somewhat dirty      Very unsafe  Somewhat unsatisfied
4           Dirty  Somewhat unsafe      Very unsatisfied

cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])

In [2]: cat_clean
Out[2]:
{'Unacceptable'}

cat_clean_rows = airlines['cleanliness'].isin(cat_clean)

In [3]: cat_clean_rows
Out[3]:
0       False
1       False
2       False
3       False
4        True
        ...  
2804    False
2805    False
2806    False
2807    False
2808    False
Name: cleanliness, Length: 2477, dtype: bool

In [4]: print(airlines[cat_clean_rows])
       id        day           airline  destination  dest_region dest_size  \
4    2992  Wednesday          AMERICAN        MIAMI      East US       Hub   
18   2913     Friday  TURKISH AIRLINES     ISTANBUL  Middle East       Hub   
100  2321  Wednesday         SOUTHWEST  LOS ANGELES      West US       Hub   

    boarding_area   dept_time  wait_min   cleanliness         safety  \
4     Gates 50-59  2018-12-31     559.0  Unacceptable      Very safe   
18   Gates 91-102  2018-12-31     225.0  Unacceptable      Very safe   
100   Gates 20-39  2018-12-31     130.0  Unacceptable  Somewhat safe   

           satisfaction  
4    Somewhat satisfied  
18   Somewhat satisfied  
100  Somewhat satisfied  
```

### Categories of errors
To address common problems affecting categorical variables in the data includes white spaces and inconsistencies in the categories, and the problem of creating new categories and mapping existing ones to new ones.

First, we can take a look at the values for a column using:
- `df['colname'].value_counts()
- or perform value counts on DataFrame: `df['col2'].groupby(df['colname']).count()`

This will give an overview of numbers of values/categories for the variable. Than we can address the problems by:

**White spaces and inconsistencies**:
- `.str.strip()`: removes all spaces before or after the column name. Strips all spaces.
- `.str.upper()`: Capitalize all labels so that every label is spelled with capital letters.
- `.str.lower()`: Lowercase, make all labels spelled with lowercase

```
# Capitalize
df['col'] = df['col'].str.upper()
df['col'].value_counts()
```

**Creating or remapping categories**:
- .replace():
- pandas.cut(): 
- pandas.qcut(): 

Collapsing data into categories: Create categories out of data - `income_group` column from `income` column

```py
# Using qcut()
import pandas as pd
group_names = ['0-200K', '200K-500K', '500K+']
demographics['income_group'] = pd.qcut(demographics['household_income'], q = 3,                                        
                                                   labels = group_names)                                                 
 # Print income_group column
 demographics[['income_group', 'household_income']]
 ```
Another method: 

```py
Using cut() - create category ranges and names
ranges = [0,200000,500000,np.inf]
group_names = ['0-200K', '200K-500K', '500K+']

# Create income group column
demographics['income_group'] = pd.cut(demographics['household_income'], bins=ranges,
                                                                        labels=group_names)
demographics[['income_group', 'household_income']]
```





