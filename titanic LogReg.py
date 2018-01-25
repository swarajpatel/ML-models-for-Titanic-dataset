"""
Created on Mon Jan 08 17:15:39 2018

@author: swaraj
"""

'''Using Logistic regression to predict if the passenger survived Titanic Crash
using the Titanic dataset on kaggle
https://www.kaggle.com/c/titanic/
'''

'''Importing the libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Importing the dataset'''
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv')

train_data = pd.read_csv('C:/Users/swara/Desktop/DS/kaggle/titanic/train.csv')
test_data = pd.read_csv('C:/Users/swara/Desktop/DS/kaggle/titanic/test.csv')

'''Take a look at data'''
print(train_data.info())
print(train_data.head())

'''
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)

   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  

After looking at data we can see that the columns PassengerId 
and Name wouldnt be much of use for our model 
We also see that much of cabin data is not present
We will try to find more about it.
'''
train_data.isnull().sum()
'''
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
'''
train_data['PassengerId'].count()
''' 891 
We can see that Cabin has many missing values and imputing this feature
won't be good for our model. So we'll drop this column.
On the other hand we can impute Age and Embarked columns.
We'll further see the how to impute these columns
'''

train_data["Embarked"].value_counts()
'''
S    644
C    168
Q     77
So we can just impute the null values by S as it has the largest occurence
'''
train_data["Embarked"].fillna("S", inplace=True)

'''For Age, we find out the mean of traveller ages in corresponding Class
and use that to fill in the Null values'''

train_data.groupby("Pclass").mean()

'''      
        PassengerId  Survived        Age     SibSp     Parch       Fare
Pclass                                                                 
1        461.597222  0.629630  38.233441  0.416667  0.356481  84.154687
2        445.956522  0.472826  29.877630  0.402174  0.380435  20.662183
3        439.154786  0.242363  25.140620  0.615071  0.393075  13.675550
'''

def fill_age(data):
    Pclass = data[1]
    Age = data[0]
    
    if(pd.isnull(Age)):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 25
    else:
        return Age
train_data['Age'] = train_data[['Age','Pclass']].apply(fill_age,axis=1)
test_data['Age'] = test_data[['Age','Pclass']].apply(fill_age,axis=1)

'''Encoding categorical variables
We will encode variables like Sex, Embarked so that our model can use them while making decisions'''
sex = pd.get_dummies(train_data['Sex'],drop_first=True)
embark = pd.get_dummies(train_data['Embarked'],drop_first=True)

'''On some research I found that the variables SibSp and Parch denote if the traveller
was alone or with family, so we'll make use of this information by encoding the two variables into one'''
train_data['Travel_Family']=train_data["SibSp"]+train_data["Parch"]
train_data['isAlone']=np.where(train_data['Travel_Family']>0, 0, 1)
train_data.drop('Travel_Family', axis=1, inplace=True)

'''Drop the variables which are not needed'''
train_data.drop(['PassengerId','Cabin','Sex','Embarked','Name','Ticket','SibSp','Parch'],axis=1,inplace=True)

'''Need to include the encoded categorical variable'''
train_data = pd.concat([train_data,sex,embark],axis=1)

'''Perform the same process for test data'''
test_data.isnull().sum()
''' Drop Cabin and impute Fare'''
test_data.at[152, 'Fare'] = 8

test_data['Travel_Family']=test_data["SibSp"]+test_data["Parch"]
test_data['isAlone']=np.where(test_data['Travel_Family']>0, 0, 1)
test_data.drop('Travel_Family', axis=1, inplace=True)

sex = pd.get_dummies(test_data['Sex'],drop_first=True)
embark = pd.get_dummies(test_data['Embarked'],drop_first=True)
test_data.drop(['Sex','Cabin','Embarked','Name','Ticket','SibSp','Parch'],axis=1,inplace=True)
test_data = pd.concat([test_data,sex,embark],axis=1)

'''Dividing the train data and label'''
train_label = train_data['Survived']
train_data.drop(['Survived'],axis=1,inplace=True)

'''Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_data = sc.fit_transform(train_data)
test_data1 = sc.transform(test_data.drop(['PassengerId'],axis=1))'''

'''Splitting the training dataset into the Training set and Test set'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size = 0.2, random_state = 0)


'''We will use Logistic Regression for this problem'''

''' Importing the Keras libraries and packages'''

from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train,y_train)


y_pred = lm.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''We get accuracy of 78.21% for this model'''