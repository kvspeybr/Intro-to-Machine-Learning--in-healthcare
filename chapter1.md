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

Our source data is available in a flatfile (csv file). The data has been pre-loaded for you through the `pandas.read_csv()` function into a Python dataframe, named ''breast_cancer''. You can now explore that dataframe by executing commands such as `breast_cancer.head(n=x)` to retrieve the first x rows of the data set (replace x by an appropriate number) or `breast_cancer.describe()` to obtain descriptive statistics of the individual columns. `breast_cancer.iloc[x,:]` whereby x is the index will give you all attributes of the x-th record.

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
