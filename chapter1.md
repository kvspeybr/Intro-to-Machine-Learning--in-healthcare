---
title: 'Loading and Exploring the Data'
description: 'Loading Wisconsin Breast Cancer Data Set and performing Exploratory Data Analysis'
---

## Explore Data - Step 1

```yaml
type: MultipleChoiceExercise
key: a661985a47
xp: 50
```

The breast cancer data set is available, pre-loaded for you through the `pandas.read_csv()` function into a Python dataframe, named ''bc''. You can now explore this dataframe by executing the following commands:

- `bc.head()`: gives the top 5 rows of the data set

- `bc.describe()` : descriptive statistics of the individual columns

- `bc.iloc[x,:]`: full content of the record with index x. Note that the first record has index 0 (so x=0). `bc.iloc[x:y,m:n]` gives rows x to y for columns m to n.

- `bc.columns`: gives the full list of column names

- `bc['column_name']`: gives all values for all subjects for the given column name. This syntaxis can be used in combination with the previous functions e.g. `bc['perimeter_mean'].describe()` gives the descriptive statistics of this specific column


Use the above commands to indicate which answer is **NOT** true

`@possible_answers`
- There are 569 records in the data set. 
- The first three patients in the dataset have a `diagnosis` of 0 (benign)
- The `compactness_mean` value for the first patient (index=0) is equal to 0.277600
- The minimum value for `perimeter_mean` is 43.790000
- The last column (variable) in the data set is `fractal_dimension_worst`

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
msg1 = "This statement is true. Using the bc.describe() command you'll see there are 569 rows for the first column (id)"
msg2 = "This statement is false. The first three patients have a diagnosis of 1"
msg3 = "This statement is true. Try using bc.iloc[0,:]"
msg4 = "This statement is true. Try with bc[perimeter_mean].describe()"
msg5 = "This statement is true. Try with bc.columns"
Ex().has_chosen(2,[msg1, msg2, msg3, msg4, msg5])
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

For calculating the number of occurrences per class, remember from the first exercise, that you could obtain all values for a given column by using `dataframe['columnname']`. To have the corresponding count per value for this column we need to extend that command by invoking the `.value_counts()` function. So the resulting command becomes: `bc['diagnosis'].value_counts()`

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
msg1 = "Congratulations. There is indeed no significant imbalance"
msg2 = "Looks like you just guessed. Too bad as you guessed wrong !"
Ex().has_chosen(1,[msg1, msg2])
```

---

## Identifying highly correlated features

```yaml
type: MultipleChoiceExercise
key: 85c85a4f15
xp: 50
```

Prior to building the model, highly correlated features need to be removed as they are redundant. In this step we'll visualize the correlations between all features. Assume that we've set the cutoff for highly correlated features at 0.9 or above (Pearson correlation coefficient). To keep all cells visible, the correlation heatmap has been reduced to 10  features. 

The syntaxis for showing the correlation matrix as a heatmap is as follows:

`sns.heatmap(bc_red.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)`

and in a second step, execute the following command:

`plt.xticks(rotation=90);plt.yticks(rotation=0);plt.show()`

Indicate the statement below that is **NOT** true

`@possible_answers`
- There is no significant correlation between `texture_mean` and any other variable
- `radius_mean` , `perimeter_mean` and `area_mean` can be replaced by one variable
- `fractal_dimension_mean` and `texture_mean` have a significant (negative) correlation
- If you are removing highly correlated features, only 6 variables would be withheld

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

bc_red=bc[['radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]


f,ax = plt.subplots(figsize=(20, 20))
```

`@sct`
```{python}
msg1 = "This is a correct statement"
msg2 = "This is a correct statement."
msg3 = "This is indeed incorrect. The correlation coefficient is -0.1, so not significant"
msg4 = "This is a correct statement"
Ex().has_chosen(3,[msg1, msg2, msg3, msg4])
```

---

## Exploratory Data Analysis - Pairplot

```yaml
type: NormalExercise
key: 78466eec7f
xp: 100
```

To visualize the relationship between different variables , you can also use a pairplot. In the exercise below you will create such a pairplot for 3 variables in function of the associated diagnosis

`@instructions`
Create a pairplot for 3 variables from the data set: `radius_mean`, `texture_mean`, `smoothness_mean`. Use `diagnosis` as the outcome variable to set the hue.

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
#Define feastures to be displayed - fill in
cols = [__, __, __]

#Create pairplot - fill in 'hue'
sns.pairplot(bc,
             x_vars = cols,
             y_vars = cols,
             hue = __, 
             palette = ('Green','Red'), 
             markers=["o", "D"]) 

#Show pairplot
plt.show()
```

`@solution`
```{python}
#Define feastures to be displayed - fill in
cols = ['radius_mean', 'texture_mean', 'smoothness_mean']

#Create pairplot - fill in 'hue'
sns.pairplot(bc,
             x_vars = cols,
             y_vars = cols,
             hue = 'diagnosis', 
             palette = ('Green','Red'), 
             markers=["o", "D"]) 

#Show pairplot       
plt.show()
```

`@sct`
```{python}
Ex().check_object("cols").has_equal_value()
Ex().check_function("seaborn.pairplot").check_args("hue").has_equal_value()
success_msg("You just became master of the pairplot !")
```

---

## Test

```yaml
type: VideoExercise
key: 68c0cf9150
xp: 50
```

`@projector_key`
55bf6b1cb3af65cccb423efd8bea0304
