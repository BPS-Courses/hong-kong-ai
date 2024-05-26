

# Python program using NumPy  
# for some basic mathematical  
# operations 
  
import numpy as np 
  
# Creating two arrays of rank 2 
x = np.array([[1, 0], [0, 1]]) 
y = np.array([[1, 2], [3, 4]]) 
  
# Creating two arrays of rank 1 
v = np.array([1, 2]) 
w = np.array([2, 1]) 
  
# Inner product of vectors 
print(np.dot(v, w), "\n") 
  
# Matrix and Vector product 
print(np.dot(x, v), "\n") 
  
# Matrix and matrix product 
print(np.dot(x, y)) 