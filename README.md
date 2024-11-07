# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.

2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output.

5.End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MADHANRAJ P
RegisterNumber: 212223220052
*/
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy```

```

## Output:

Result Output:

![328146137-ebca3817-9ad2-4374-bf00-f29b9b6d0598](https://github.com/RamkumarGunasekaran/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870820/93bef49e-2c2b-4fd7-a971-457bfd33130a)

Data.head():

![328146217-482bcc29-05cd-4eaf-bd7a-a0ae5f96deb9](https://github.com/RamkumarGunasekaran/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870820/a6882d64-6ef1-49d5-9ad1-446b1b66e99d)

Data.info():

![328146313-78b3b8ad-750e-424a-853f-b876add8be72](https://github.com/RamkumarGunasekaran/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870820/4f699cf0-de22-4b1f-bbcd-b28d8a45b175)

Data.insull().sum():

![328146410-bf878d0f-b9cf-429d-a9bd-a6c64aeb9357](https://github.com/RamkumarGunasekaran/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870820/93dd19a5-4b0c-4812-a981-781ef2d09a6f)

![328146462-ca376c9b-f59d-46eb-b97a-e751e5a05d38](https://github.com/RamkumarGunasekaran/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870820/3ddd2f9e-4a0f-46d2-beb4-adde48315e41)

Y_Prediction Value:

![328146747-b1b11a82-da48-4063-85cb-ee7bfef2860c](https://github.com/RamkumarGunasekaran/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870820/c3aafbac-8b17-4858-b5e3-89333c19a0a7)

Accuracy Value:

![328146839-c090233c-85d3-4285-a19a-f6b8e4327d60](https://github.com/RamkumarGunasekaran/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870820/d524c12f-e273-482e-a7df-24405badac33)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
