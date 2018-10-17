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

names = ['id_number', 'diagnosis', 'radius_mean', 
         'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean', 
         'fractal_dimension_mean', 'radius_se', 'texture_se', 
         'perimeter_se', 'area_se', 'smoothness_se', 
         'compactness_se', 'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se', 
         'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst', 
         'compactness_worst', 'concavity_worst', 
         'concave_points_worst', 'symmetry_worst', 
         'fractal_dimension_worst'] 

#read Wisconsin breast cancer data set
breast_cancer = pd.read_csv('http://assets.datacamp.com/production/repositories/3733/datasets/0eb6987cb9633e4d6aa6cfd11e00993d2387caa4/data.csv', names=names)

# Convert diagnosis to binary : M=1, B=0
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})
```

`@sct`
```{python}

```
