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

```py
# Print the information of ride_sharing
print(ride_sharing.info())

# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())
```
