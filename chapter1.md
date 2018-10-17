---
title: 'Loading the Data Set'
description: 'Loading Wisconsin Breast Cancer Data Set'
---

## Insert exercise title here

```yaml
type: MultipleChoiceExercise
key: a661985a47
xp: 50
```



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
breast_cancer = pd.read_csv('http://assets.datacamp.com/production/repositories/3733/datasets/0eb6987cb9633e4d6aa6cfd11e00993d2387caa4/data.csv', skiprows = 1)

# Convert diagnosis to binary : M=1, B=0
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})
```

`@sct`
```{python}

```
