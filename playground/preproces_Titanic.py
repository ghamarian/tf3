from scipy import stats
import numpy as np
import pandas as pd

def feature_normalize(dataset):
   mu = np.mean(dataset,axis=0)
   sigma = np.std(dataset,axis=0)
   return (dataset - mu)/sigma

def str_to_int(df):
   str_columns = df.select_dtypes(['object']).columns
   print(str_columns)
   for col in str_columns:
       df[col] = df[col].astype('category')

   cat_columns = df.select_dtypes(['category']).columns
   df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
   return df

def count_space_except_nan(x):
   if isinstance(x,str):
       return x.count(" ") + 1
   else :
       return 0

# https://stackoverflow.com/a/42523230
def one_hot(df, cols):
   """
   @param df pandas DataFrame
   @param cols a list of columns to encode
   @return a DataFrame with one-hot encoding
   """
   for each in cols:
       dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
       del df[each]
       df = pd.concat([df, dummies], axis=1)
   return df

df_train = pd.read_csv('data/train.csv')

print (df_train.isnull().sum())
delete_columns = ["Ticket", "Name", "PassengerId", "Cabin", "Embarked"]

def pre_processing(df):
   df.drop(delete_columns, axis=1, inplace=True)
   # Count room nubmer
   # df_train["Cabin"] = df_train["Cabin"].apply(count_space_except_nan)
   # Replace NaN with mean value
   df["Age"].fillna(df["Age"].mean(), inplace=True)
   # Pclass, Embarked one-hot
   df = one_hot(df, df.loc[:, ["Pclass"]].columns)
   # String to int
   df = str_to_int(df)
   # Age Normalization
   df["Age"] = feature_normalize(df["Age"])
   stats.describe(df).variance
   return df

df_train = pre_processing(df_train)
#save PassengerId for evaluation
df_train.to_csv('titanic_train.csv')