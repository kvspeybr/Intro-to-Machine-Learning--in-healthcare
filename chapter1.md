---
title: 'Loading the Data Set'
description: 'Loading Wisconsin Breast Cancer Data Set'
---

## Review input data

```yaml
type: MultipleChoiceExercise
key: a661985a47
xp: 50
```

The breast cancer data set is available, pre-loaded for you through the `pandas.read_csv()` function into a Python dataframe, named ''breast_cancer''. You can now explore this dataframe by executing the following commands:

- `breast_cancer.head()`: gives the top 5 rows of the data set

- `breast_cancer.describe()` : descriptive statistics of the individual columns

- `breast_cancer.iloc[x,:]`: fulll content of the x-th record.

- `breast_cancer.columns`: gives the full list of column names

- `breast_cancer['column_name']`: gives all values for all subjects for the given column name. This syntaxis can be used in combination with the previous functions e.g. `breast_cancer['perimeter_mean'].describe()` gives the descriptive statistics only of this specific column


Use the above commands to indicate which answer is **NOT** true

`@possible_answers`
1. Yes
2. No
3. Sometimes

`@hint`


`@pre_exercise_code`
```{python}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
plt.style.use('ggplot')

#read Wisconsin breast cancer data set
breast_cancer = pd.read_csv('http://assets.datacamp.com/production/repositories/3810/datasets/0eb6987cb9633e4d6aa6cfd11e00993d2387caa4/data.csv')

# Convert diagnosis to binary : M=1, B=0
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})
```

`@sct`
```{python}

```
