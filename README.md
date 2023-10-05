# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Prepare your data Clean and format your data Split your data into training and testing sets

2.Define your model Use a sigmoid function to map inputs to outputs Initialize weights and bias terms

3.Define your cost function Use binary cross-entropy loss function Penalize the model for incorrect predictions

4.Define your learning rate Determines how quickly weights are updated during gradient descent

5.Train your model Adjust weights and bias terms using gradient descent Iterate until convergence or for a fixed number of iterations

6.Evaluate your model Test performance on testing data Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters Experiment with different learning rates and regularization techniques

8.Deploy your model Use trained model to make predictions on new data in a real-world application.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DIVYA.A
RegisterNumber: 212222230034
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data.isnull()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Initial data set:

![1](https://github.com/Divya110205/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404855/a5e6c0f4-c7da-4179-8fa1-4866c0987ea3)

### Data info:

![2](https://github.com/Divya110205/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404855/756e1cc9-134a-4af1-8646-ec789a04decb)

### Optimization of null values:

![3](https://github.com/Divya110205/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404855/21abfb8e-01e2-43d4-b7a6-74a9a1c0294c)

### Assignment of x and y values:

![4](https://github.com/Divya110205/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404855/1a4ae067-f71e-494b-8f9f-5cf85ea2d28a)
![6](https://github.com/Divya110205/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404855/83978941-5382-41d2-8078-ab22dfd639ef)

### Converting string literals to numerical values using label encoder:

![5](https://github.com/Divya110205/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404855/bec05dde-c210-473f-a727-54edf6d8d031)

### Accuracy:
![7](https://github.com/Divya110205/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404855/f3807974-e26b-433c-bc90-6c60bf0eda5b)

### Prediction:

![8](https://github.com/Divya110205/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404855/b794688c-0a66-41e1-984a-07393cfe4e06)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
