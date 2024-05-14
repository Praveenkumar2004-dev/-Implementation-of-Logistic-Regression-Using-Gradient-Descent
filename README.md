# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction.

## Program:
```
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/classes/ML/New folder/Placement_Data.csv")
dataset

#dropping the serial no and salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('sl_no',axis=1)

#catogorising col for further labegling
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

#labelling the colums
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

#display dataset
dataset

#selecting the features and labels
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#display dependent variables
Y

#initialize the model parameter
theta = np.random.randn(X.shape[1])
y=Y

#define the sigmoid function 
def sigmoid(z):
    return 1/(1+np.exp(-z))

#define the loss function 
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

#define the gradient descent algorithm
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta

#train the model
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

# make the predictions
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)

#evaluate the model
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy : ",accuracy)
print(y_pred)
print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

```

## Output:
#### Array Value of x
![Screenshot 2023-05-11 155230](https://github.com/Yamunaasri/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115707860/033e44c7-01d8-4694-af58-e47e586bc326)

#### Array Value of y
![Screenshot 2023-05-11 155238](https://github.com/Yamunaasri/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115707860/3d9afb2e-3520-4ab7-a959-1685a95c48cb)

#### Exam 1 - score graph
![Screenshot 2023-05-11 161150](https://github.com/Yamunaasri/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115707860/6c725e84-829f-459b-9dab-ec3c9c4316fb)


#### Sigmoid function graph
![Screenshot 2023-05-11 155309](https://github.com/Yamunaasri/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115707860/b114a544-ffeb-42c5-a36d-e6bdb52a65ea)

#### X_train_grad value
![Screenshot 2023-05-11 155324](https://github.com/Yamunaasri/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115707860/6d006193-6d73-44d9-8290-4d82eb608d7e)

#### Y_train_grad value
![Screenshot 2023-05-11 155335](https://github.com/Yamunaasri/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115707860/71194e2e-7353-4bf9-93d7-027575624256)

#### Print res.x
![Screenshot 2023-05-11 155723](https://github.com/Yamunaasri/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115707860/16c592fc-3657-4346-bdb5-c27acad5a7f8)

#### Decision boundary - graph for exam score
![Screenshot 2023-05-11 155730](https://github.com/Yamunaasri/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115707860/beecf6cd-c08f-47df-bb02-00eadb43d41f)


#### Proability value 
![Screenshot 2023-05-11 155822](https://github.com/Yamunaasri/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115707860/b92b63f4-f532-462a-8593-d86f4cc83efd)

#### Prediction value of mean
![Screenshot 2023-05-11 155829](https://github.com/Yamunaasri/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115707860/f8ee18ed-c208-47d9-ac94-7704d6852df1)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
