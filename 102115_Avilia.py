import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import plot_confusion_matrix , log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import KFold

SEED = 42

import warnings
warnings.simplefilter(action = "ignore")

df = pd.read_csv("diabetes.csv")

df.head()
df.shape
df.info()

# Descriptive statistics of the data set accessed
df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
# The distribution of the Outcome variable was examined
df["Outcome"].value_counts()*100/len(df)

# The histagram of the Age variable was reached
df["Age"].hist(edgecolor = "black")
print("Max Age: " + str(df["Age"].max()) + " Min Age: " + str(df["Age"].min()))

# Histogram and density graphs of all variables
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df.Age, bins = 20, ax=ax[0,0]) 
sns.distplot(df.Pregnancies, bins = 20, ax=ax[0,1]) 
sns.distplot(df.Glucose, bins = 20, ax=ax[1,0]) 
sns.distplot(df.BloodPressure, bins = 20, ax=ax[1,1]) 
sns.distplot(df.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(df.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sns.distplot(df.BMI, bins = 20, ax=ax[3,1])
plt.show()

df.groupby("Outcome").agg({"Pregnancies":"mean"})
df.groupby("Outcome").agg({"Age":"mean"})
df.groupby("Outcome").agg({"Age":"max"})
df.groupby("Outcome").agg({"Insulin": "mean"})
df.groupby("Outcome").agg({"Insulin": "max"})
df.groupby("Outcome").agg({"Glucose": "mean"})
df.groupby("Outcome").agg({"Glucose": "max"})
df.groupby("Outcome").agg({"BMI": "mean"})
df.corr()

# The distribution of the outcome variable in the data was examined and visualized
f,ax=plt.subplots(1,2,figsize=(18,8))
df['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot('Outcome',data=df,ax=ax[1])
ax[1].set_title('Outcome')
plt.show()

# Correlation matrix graph of the data set
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#Checking for any missing values in the dataset.
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df.head()
# we can look at where are missing values

print(df.isnull().sum())

msno.bar(df)
plt.show()

#Plotting relationships in the dataset.
# The missing values ​​will be filled with the median values ​​of each variable
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

#The values to be given for incomplete observations are given the median value of people who are not sick and the median values of people who are sick
columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]
df.head()
print(df.isnull().sum())

# In the data set, there were asked whether there were any outlier observations compared to the 25% and 75% quarters
# It was found to be an outlier observation
for feature in df:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1- 1.5*IQR
    upper = Q3 + 1.5*IQR
    if df[(df[feature] > upper)].any(axis=None):
        print(feature,"yes")
    else:
        print(feature, "no")

#visualizing variable with boxplot
sns.boxplot(x = df["Pregnancies"])
plt.show()
sns.boxplot(x = df["BMI"])
plt.show()
sns.boxplot(x = df["BloodPressure"])
plt.show()
sns.boxplot(x = df["SkinThickness"])
plt.show()
sns.boxplot(x = df["Age"])
plt.show()
sns.boxplot(x = df["Insulin"])
plt.show()
sns.boxplot(x = df["DiabetesPedigreeFunction"])
plt.show()

#conduct a stand alone observation review for the Insulin variable
#suppress contradictory values
Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["Insulin"] > upper,"Insulin"] = upper
sns.boxplot(x = df["Insulin"])
# sns.set()
plt.show()

#training the data
#splitting the dataset
x = df.drop('Outcome', axis=1)
y= df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=101)

#check the column with zero values
print("Total number of rows: {0}", format(len(df)))
print("Number of rows missing Pregnancies: {0}", format(len(df.loc[df['Pregnancies']== 0])))
print("Number of rows missing Glucose: {0}", format(len(df.loc[df['Glucose']== 0])))
print("Number of rows missing BloodPressure: {0}", format(len(df.loc[df['BloodPressure']== 0])))
print("Number of rows missing SkinThickness: {0}", format(len(df.loc[df['SkinThickness']== 0])))
print("Number of rows missing Insulin: {0}", format(len(df.loc[df['Insulin']== 0])))
print("Number of rows missing BMI: {0}", format(len(df.loc[df['BMI']== 0])))
print("Number of rows missing DiabetesPedigreeFunction: {0}", format(len(df.loc[df['DiabetesPedigreeFunction']== 0])))
print("Number of rows missing Age: {0}", format(len(df.loc[df['Age']== 0])))

#Machine Learning Model

models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC()))
models.append(('XGB', GradientBoostingClassifier()))
models.append(("LightGBM", LGBMClassifier()))
models.append(("LinearSVC", LinearSVC()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    
        kfold = KFold(n_splits = 10)
        cv_results = cross_val_score(model, x, y, cv = 10, scoring= "accuracy")
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

## Baseline model performance evaluation
# kf = KFold(n_splits=5, shuffle=True, random_state=SEED)   # this may result in imbalance classes in each fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# to give model baseline report in dataframe 
def baseline_report(model, x_train, x_test, y_train, y_test, name):
    model.fit(x_train, y_train)
    accuracy     = np.mean(cross_val_score(model, x_train, y_train, cv=kf, scoring='accuracy'))
    precision    = np.mean(cross_val_score(model, x_train, y_train, cv=kf, scoring='precision'))
    recall       = np.mean(cross_val_score(model, x_train, y_train, cv=kf, scoring='recall'))
    f1score      = np.mean(cross_val_score(model, x_train, y_train, cv=kf, scoring='f1'))
    rocauc       = np.mean(cross_val_score(model, x_train, y_train, cv=kf, scoring='roc_auc'))
    y_pred = model.predict(x_test)
    logloss      = log_loss(y_test, y_pred)   # SVC & LinearSVC unable to use cvs

    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'rocauc'       : [rocauc],
                             'logloss'      : [logloss],
                             'timetaken'    : [0]       })   # timetaken: to be used for comparison later
    return df_model

# to evaluate baseline models
logit = LogisticRegression()
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
svc = SVC()
XGB = GradientBoostingClassifier()
LGB = LGBMClassifier()
linearsvc = LinearSVC()

# to concat all models
df_models = pd.concat([baseline_report(logit, x_train, x_test, y_train, y_test, 'LogisticRegression'),
                       baseline_report(knn, x_train, x_test, y_train, y_test, 'KNN'),
                       baseline_report(decisiontree, x_train, x_test, y_train, y_test, 'DecisionTree'),
                       baseline_report(randomforest, x_train, x_test, y_train, y_test, 'RandomForest'),
                       baseline_report(svc, x_train, x_test, y_train, y_test, 'SVC'),
                       baseline_report(XGB, x_train, x_test, y_train, y_test, 'XGB'),
                       baseline_report(LGB, x_train, x_test, y_train, y_test, 'LightGBM'),
                       baseline_report(linearsvc, x_train, x_test, y_train, y_test, 'LinearSVC')
                       ], axis=0).reset_index()
df_models = df_models.drop('index', axis=1)
print(df_models)

#reporting

dt_clf = XGBClassifier()
dt_clf.fit(x_train, y_train)
labels = ['no diabetes', 'diabetes']
disp = plot_confusion_matrix(dt_clf, 
                             x_test, y_test, 
                             display_labels=labels, 
                             cmap=plt.cm.Reds, 
                             normalize=None)
disp.ax_.set_title('Confusion Matrix');
plt.show()