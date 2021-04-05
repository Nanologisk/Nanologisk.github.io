---
layout: article
title: Cleaning data in Python (1)
key: 20200610
tags: Python, Data science
pageview: false
modify_date: 2020-06-11
aside:
  toc: true
---

Based on DataCamp.

<!--more-->
## Common data Problems
### Common data types
- Numeric data types
- Text
- Dates

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
