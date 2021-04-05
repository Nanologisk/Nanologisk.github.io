---
layout: article
title: Cleaning data in Python 
key: 20200610
tags: Python, Data science
pageview: false
modify_date: 2020-06-11
aside:
  toc: true
---

Based on DataCamp.

<!--more-->
## 1. Common data Problems
### a. Common data types
- Numeric data types
- Text
- Dates

Manipulating and analyzing data with incorrect data types could lead to compromised analysis as you go along the data science workflow.

When working with new data, we could use the `.dtypes` attribute or the `.info()` method. Often times, you'll run into columns that should be converted to different data types before starting any analysis.

### b. To describe data and check data types:
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

### c. Summing strings and concatenating numbers
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

### d. Tire size constraints
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

### e. Back to the future
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

### f. Fiding duplicates

### g. Treating duplicates


## 2. Text and categorical data problems
