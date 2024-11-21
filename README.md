# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Read the CSV file and detect its encoding to ensure proper data loading.

2.Load the data into a DataFrame and inspect it to understand its structure and contents

3.Extract features (X) and labels (y) from the DataFrame. Here, v2 is the message text, and v1 is the label indicating whether the message is spam or not.

4.Split the data into training and testing sets to evaluate the model's performance.

5.Convert the text data into numerical data using CountVectorizer, which transforms the text into a matrix of token counts.

6.Train the SVM model on the training data and evaluate its accuracy on the test data.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MADHANRAJ P
RegisterNumber:212223220052
*/
import chardet
with open('spam.csv','rb') as file:
    result = chardet.detect(file.read(10000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')


data.head()
data.info()
data.isnull().sum()

x=data["v2"].values
y=data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc



```

## Output:
### Encoding:
![image](https://github.com/user-attachments/assets/fae5d304-fe30-43aa-9eba-8255a181fb9f)

### Head():
![image](https://github.com/user-attachments/assets/d824d63e-db25-42b9-823f-7cf6c2a12a78)

### Info():
![image](https://github.com/user-attachments/assets/028ed981-94bf-404b-941b-06098c18339d)

### isnull().sum():
![image](https://github.com/user-attachments/assets/d5f2de14-34de-4a6b-a3db-0a7ee283d451)

### Prediction of y:
![image](https://github.com/user-attachments/assets/ac26701d-27e6-4d43-b0c2-96adad8b97ac)

### Accuracy:
![image](https://github.com/user-attachments/assets/994ee6e0-6c4f-45cc-b051-ab63f6d013b1)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
