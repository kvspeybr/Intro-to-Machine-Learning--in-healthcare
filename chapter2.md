---
title: 'Exploratory Data Analysis'
description: 'Exploratory Data Analysis'
---

## Machine Learning

```yaml
type: TabExercise
key: 318e9a3c2f
xp: 100
```

Prior to executing a machine learning model, a data normalisation step is normally included. This will scale all variables so that the absolute value of a variable, does not influence the model build. Secondly, data is split between a training set and a test set. The training set is used to calculate the model parameters against, while the test set is used to validate the calculated model

`@pre_exercise_code`
```{python}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#read Wisconsin breast cancer data set
bc = pd.read_csv('http://assets.datacamp.com/production/repositories/3810/datasets/7c19b7d9c1db98790fcf3efc234807a478e6a53e/data.csv')

# Convert diagnosis to binary : M=1, B=0
bc['diagnosis'] = bc['diagnosis'].map({'M':1, 'B':0})
```

***

```yaml
type: NormalExercise
key: 90152f6a90
xp: 100
```

`@instructions`
In a first step the data will be normalized.

Split data in a training and test set. You will do this using the `train_test_split` method from the `sklearn.model_selection` library. Typically, the ratio training vs test data is in the range of 0.7 to 0.8. In this specific exercise, use a ratio of 0.25 for the fraction of test data.

`@hint`


`@sample_code`
```{python}
#Split the dataframe into an array 'X' with the input variables and an array 'y' with the outcome variable
X = bc[['fractal_dimension_mean', 'smoothness_se']].values
y = bc['diagnosis'].values

#Scale features
X = sc.fit_transform(StandardScaler())

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = __, random_state = 0)
```

`@solution`
```{python}
X = bc[['fractal_dimension_mean', 'smoothness_se']].values
y = bc['diagnosis'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

`@sct`
```{python}
Ex().has_equal_value()
```

***

```yaml
type: NormalExercise
key: 5c790eff84
```

`@instructions`
In this step, you will actually run a Random Forest Classifier algorithm. After instantiating the RandomForestClassifier class, run the `.fit(X,y)` method. You should use the training subset to perform this `.fit(X,y)` method. The first parameter in the fit method will be the feature training set (`X-train`), the second parameter is the outcome variable i.e. `y_train`

`@hint`


`@sample_code`
```{python}
#instantiation, random_state is set to a constant so that we obtain the same result when re-executing
classifier = RandomForestClassifier(random_state=43)

#perform the fit, using the training subset
classifier = classifier.fit(__,__)
```

`@solution`
```{python}
#instantiation, random_state is set to a constant so that we obtain the same result when re-executing
classifier = RandomForestClassifier(random_state=43)

#perform the fit, using the training subset
classifier = classifier.fit(X_train,y_train)
```

`@sct`
```{python}

```
