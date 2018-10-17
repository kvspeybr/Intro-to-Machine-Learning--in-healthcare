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

- `breast_cancer.iloc[x,:]`: fulll content of the record with index x. Note that the first record has index 0 (so x=0)

- `breast_cancer.columns`: gives the full list of column names

- `breast_cancer['column_name']`: gives all values for all subjects for the given column name. This syntaxis can be used in combination with the previous functions e.g. `breast_cancer['perimeter_mean'].describe()` gives the descriptive statistics only of this specific column


Use the above commands to indicate which answer is **NOT** true

`@possible_answers`
1. There are 569 records in the data set "breast_cancer"
2. The first three patients in the dataset have a diagnosis of 0 (benign)
3. The "compactness_mean" value for the first patient (index=0) is equal to 0.277600
4. The minimum value for "perimeter_mean" is 43.790000
5. The last column (variable) in the data set is "fractal_dimension_worst"

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
breast_cancer = pd.read_csv('http://assets.datacamp.com/production/repositories/3810/datasets/7c19b7d9c1db98790fcf3efc234807a478e6a53e/data.csv')

# Convert diagnosis to binary : M=1, B=0
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})
```

`@sct`
```{python}

```

---

## Insert exercise title here

```yaml
type: NormalExercise
key: 78466eec7f
xp: 100
```



`@instructions`


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
breast_cancer = pd.read_csv('http://assets.datacamp.com/production/repositories/3810/datasets/7c19b7d9c1db98790fcf3efc234807a478e6a53e/data.csv')

# Convert diagnosis to binary : M=1, B=0
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})

%matplotlib inline
```

`@sample_code`
```{python}
cols = ['concave points_worst', 'concavity_mean', 
        'perimeter_worst', 'radius_worst', 
        'area_worst', 'diagnosis']

sns.pairplot(breast_cancer,
             x_vars = cols,
             y_vars = cols,
             hue = 'diagnosis', 
             palette = ('Green','Red'), 
             markers=["o", "D"]
             #, plot_kws={'scatter_kws': {'alpha': 0.1}}
            ) 
```

`@solution`
```{python}
cols = ['concave points_worst', 'concavity_mean', 
        'perimeter_worst', 'radius_worst', 
        'area_worst', 'diagnosis']

sns.pairplot(breast_cancer,
             x_vars = cols,
             y_vars = cols,
             hue = 'diagnosis', 
             palette = ('Green','Red'), 
             markers=["o", "D"]
             ) 
```

`@sct`
```{python}
Ex().has_equal_value()
```
