---
title: 'Loading and Exploring the Data'
description: 'Loading Wisconsin Breast Cancer Data Set and performing Exploratory Data Analysis'
---

## Review input data

```yaml
type: MultipleChoiceExercise
key: a661985a47
xp: 50
```

The breast cancer data set is available, pre-loaded for you through the `pandas.read_csv()` function into a Python dataframe, named ''bc''. You can now explore this dataframe by executing the following commands:

- `bc.head()`: gives the top 5 rows of the data set

- `bc.describe()` : descriptive statistics of the individual columns

- `bc.iloc[x,:]`: fulll content of the record with index x. Note that the first record has index 0 (so x=0)

- `bc.columns`: gives the full list of column names

- `bc['column_name']`: gives all values for all subjects for the given column name. This syntaxis can be used in combination with the previous functions e.g. `bc['perimeter_mean'].describe()` gives the descriptive statistics only of this specific column


Use the above commands to indicate which answer is **NOT** true

`@possible_answers`
1. There are 569 records in the data set. 
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
bc = pd.read_csv('http://assets.datacamp.com/production/repositories/3810/datasets/7c19b7d9c1db98790fcf3efc234807a478e6a53e/data.csv')

# Convert diagnosis to binary : M=1, B=0
bc['diagnosis'] = bc['diagnosis'].map({'M':1, 'B':0})
```

`@sct`
```{python}

```

---

## Exploratory Data Analysis - Check for Class Imbalance

```yaml
type: MultipleChoiceExercise
key: 4a9f7a78bd
xp: 50
```

An important step in exploring the data is to check whether there is class-imbalance. Class imbalance occurs when certain categories of outcomes (in our case diagnosis) are over-represented vs other categories. In case of class-imbalance the weight of classes will need to be corrected.

In the case of the breast cancer data set, there are only two possible classes: 0 or 1. You will perform a count of the number of occurrences of both classes and select the correct answer from the options below

For calculating the number of occurrences per class, remember from the first exercise, that you could obtain all values for a given column by using `dataframe['columnname']`. To have the corresponding count per value for this column we need to extend that command by invoking the `.value_counts()` function. So the resulting command becomes: `breast_cancer['diagnosis'].value_counts()`

`@possible_answers`
1. There are 212 samples with malignant tumors vs 357 benign - no significant class imbalance
2. There are 41 malignant tumors vs 357 benign - a significant class imbalance

`@hint`


`@pre_exercise_code`
```{python}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read Wisconsin breast cancer data set
bc = pd.read_csv('http://assets.datacamp.com/production/repositories/3810/datasets/7c19b7d9c1db98790fcf3efc234807a478e6a53e/data.csv')

# Convert diagnosis to binary : M=1, B=0
bc['diagnosis'] = bc['diagnosis'].map({'M':1, 'B':0})
```

`@sct`
```{python}

```

---

## Exploratory Data Analysis - Pairplot

```yaml
type: NormalExercise
key: 78466eec7f
xp: 100
```

After loading the data, the Exploratory Data Analysis (EDA) is used to get an understanding of the data and possible indications of relationships. One of the possible tools to visualize different variables is to use a pairplot. in the exercise below you will create such a pairplot for 3 variables in function of the associated diagnosis

`@instructions`
Create a pairplot for 3 variables from the breast_cancer data set: perimeter_worst , concavity_mean and area_se. In the template code below replace the variables (replace_me) in the cols-array by the respective variables that are to be displayed. Use 'diagnosis' as the variable to set the hue.

`@hint`


`@pre_exercise_code`
```{python}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read Wisconsin breast cancer data set
bc = pd.read_csv('http://assets.datacamp.com/production/repositories/3810/datasets/7c19b7d9c1db98790fcf3efc234807a478e6a53e/data.csv')

# Convert diagnosis to binary : M=1, B=0
bc['diagnosis'] = bc['diagnosis'].map({'M':1, 'B':0})
```

`@sample_code`
```{python}
cols = [__, __, __]

sns.pairplot(bc,
             x_vars = cols,
             y_vars = cols,
             hue = 'diagnosis', 
             palette = ('Green','Red'), 
             markers=["o", "D"]) 
            
plt.show()
```

`@solution`
```{python}
cols = ['concave points_worst', 'concavity_mean','perimeter_worst']

sns.pairplot(bc,
             x_vars = cols,
             y_vars = cols,
             hue = 'diagnosis', 
             palette = ('Green','Red'), 
             markers=["o", "D"]) 
            
plt.show()
```

`@sct`
```{python}
Ex().has_equal_value()
```

---

## Identifying highly correlated features

```yaml
type: MultipleChoiceExercise
key: 85c85a4f15
xp: 50
```

Prior to building the model, highly correlated features need to be removed as they are redundant. In this step we'll visualize the correlations between all features. Assume that we've set the cutoff for highly correlated features at 0.9 or above (Pearson correlation coefficient). To keep everything visible, the correlation heatmap has been reduced to 13  features. 

The syntaxis for showing the correlation matrix as a heatmap is as follows:

`sns.heatmap(bc_red.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)`

and in a second step, execute the following command:

`plt.xticks(rotation=90);plt.yticks(rotation=0);plt.show()`

`@possible_answers`


`@hint`


`@pre_exercise_code`
```{python}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read Wisconsin breast cancer data set
bc = pd.read_csv('http://assets.datacamp.com/production/repositories/3810/datasets/7c19b7d9c1db98790fcf3efc234807a478e6a53e/data.csv')

# Convert diagnosis to binary : M=1, B=0
bc['diagnosis'] = bc['diagnosis'].map({'M':1, 'B':0})

bc_red=bc[['smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se','compactness_se', 'concavity_se']]


f,ax = plt.subplots(figsize=(20, 20))
```

`@sct`
```{python}

```
