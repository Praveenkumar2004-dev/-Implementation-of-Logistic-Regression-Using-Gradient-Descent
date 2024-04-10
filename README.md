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
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PRAVEENKUMAR S
RegisterNumber: 212222230108
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize    #to remove unwanted data and memory storage

data=np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

Visualizing the data
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

Sigmoid fuction
def sigmoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFuction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J= -(np.dot(y, np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y) / X.shape[0]
  return J,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J, grad=costFuction(theta, X_train, y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J, grad=costFuction(theta, X_train, y)
print(J)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J= -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  return J
  
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad= np.dot(X.T, h-y) / X.shape[0]
  return grad
  
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta= np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method="Newton-CG",jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min() - 1, X[:,0].max()+1
  y_min, y_max = X[:,1].min() - 1, X[:,1].max()+1
  xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                       np.arange(y_min,y_max,0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot, theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
  plotDecisionBoundary(res.x,X,y)
  
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return(prob >= 0.5).astype(int)
  
np.mean(predict(res.x,X)==y)
*/
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
