# importing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression as make_reg
import random
from datetime import datetime

# calculates mean squared error 
def MSE(m, c, x, y):
    totalError=0.0
    y=y.reshape(y.shape[0], 1)
    for i in range(len(x)):
        y_predi = m * x[i]+ c
        error = (y[i]-y_predi) ** 2
        totalError += error
    return float(totalError) / len(x) 

# calculates dynamic learning rate according to gradient
def get_dynamic_alpha(m, c, x, y):
    c_grad=0
    n_points=len(x)
    y=y.reshape(y.shape[0], 1)
    for i in range(n_points):
        dc = -((2/n_points) * (y[i] - (m*x[i] + c)))
        c_grad+=dc
    alpha= abs(c_grad*0.001)
    return alpha

# calculates one step of the gradient descent model 
def gradient_descent_one_step(m,c, x, y, alpha):
    n_points = len(x) #size of data
    m_grad=0
    c_grad=0
    for i in range(n_points):
        dm = -((2/n_points) * x[i] * (y[i] - (m*x[i] + c)))
        dc = -((2/n_points) * (y[i] - (m*x[i] + c)))
        m_grad += dm
        c_grad += dc
    m_updated = m - alpha*m_grad
    c_updated = c - alpha*c_grad
    return float(m_updated), float(c_updated)

def error(m,c,x,y, alpha):
    error1=list()
    for step in range(1000):
        sum_error1=0
        for i in range(len(x)):
            m, c= gradient_descent_one_step(m, c, x, y, alpha)
            error=y[i]- ((m*x[i]+c)) 
            sum_error1 += error
        error1.append(sum_error1/len(x))        
    return np.array(error1)

#returns one point from dataset of points
def samp_from(px, py): 
      k=random.randint(0, len(px)-1)
      return px[k], py[k]

# Generates data
px20, py20 = make_reg(20, 1,noise = 10, bias=10)
py20=py20.reshape(py20.shape[0],1)
px50, py50 = make_reg(50, 1, noise = 10, bias=10)
py50=py50.reshape(py50.shape[0],1)
px100, py100 = make_reg(100, 1, noise= 10, bias=10)
py100=py100.reshape(py100.shape[0],1)
px200, py200 = make_reg(200, 1, noise= 10, bias=10)
py200=py200.reshape(py200.shape[0],1)

#bringing all together
all_sets= np.array([[px20, py20],[px50, py50], [px100, py100], [px200, py200]]) 
 
# show plots
fig, ((ax11, ax12), (ax21, ax22))= plt.subplots(2,2)
ax11.scatter(px20, py20)
ax11.set_title('20 samples')
ax12.scatter(px50, py50)
ax12.set_title('50 samples')
ax21.scatter(px100, py100)
ax21.set_title('100 samples')
ax22.scatter(px200, py200)
ax22.set_title('200 samples')
plt.show()

# initializing m, c
initial_m=random.randint(0, 15)
initial_c=random.randint(0, 15)

# defining learning rates and number of steps needed according to the assignment
dynamic_alpha= 0.1 #just a temporary value
alphas= np.array([0.00001, 0.000001, dynamic_alpha]) #learning rates
steps_list= np.array([10, 500, 1000])

# ## Part one
# BATCH
print('BATCH')
for index in range(len(all_sets)):
    x=all_sets[index][0]
    y=all_sets[index][1]
    m=initial_m
    c=initial_c
    for alpha_index in range(len(alphas)):
          if alpha_index==2:
            alphas[2]= get_dynamic_alpha(m, c, x, y)
          for steps in steps_list:
            for step in range(steps):
                m,c = gradient_descent_one_step(m,c,x,y,alphas[alpha_index])
                mse = MSE(m,c, x, y)
            print(len(x), m, c, alphas[alpha_index], steps, mse)
          
# STOCHASTIC
print('\nSTOCHASTIC')
for i in range(len(all_sets)):
    m=initial_m
    c=initial_c
    k=random.randint(0, len(all_sets[i][0])-1)
    x=all_sets[i][0][k]
    y=all_sets[i][1][k]
    for alpha_index in range(len(alphas)):
        if alpha_index==2:
            alphas[2]= get_dynamic_alpha(m, c, x, y)
        for steps in steps_list:
            for step in range(steps):
                m,c = gradient_descent_one_step(m,c,x,y,alphas[alpha_index])
                mse = MSE(m,c, x, y)
            print(len(all_sets[i][0]), m, c, alphas[alpha_index], steps, mse)

#MINI_BATCH
print('\nMINI BATCH')
for i in range(len(all_sets)):
    m=initial_m
    c=initial_c
    if len(all_sets[i][0]==20): N=[5, 10]
    else: N=[5,10,20]
    for n in N:
        random_indexes=random.sample(range(len(all_sets[i][0])-1), n)
        x=np.take(all_sets[i][0], random_indexes)
        y=np.take(all_sets[i][1], random_indexes)
        for alpha_index in range(len(alphas)):    
            if alpha_index==2:
                alphas[2]= get_dynamic_alpha(m, c, x, y)             
            for steps in steps_list:
                for step in range(steps):
                    m,c = gradient_descent_one_step(m,c,x,y,alphas[alpha_index])
                    mse = MSE(m,c, x, y)
                print(len(all_sets[i][0]), m, c, alphas[alpha_index], steps, mse, n)


##Part two           
''' Let's plot histograms of the mean errors for each step where:
   steps=1000, alpha1=0.00007, alpha2=0.0003'''
# For Batch
a=0.00007
error20_00007_B=error(initial_m, initial_c, px20, py20, a)
error50_00007_B=error(initial_m, initial_c, px50, py50,a)
error100_00007_B=error(initial_m, initial_c, px100, py100, a)
error200_00007_B=error(initial_m, initial_c, px200, py200, a)
fig, ((ax11, ax12), (ax21, ax22))= plt.subplots(2,2, figsize=(16,16))
ax11.hist(error20_00007_B)
ax11.set_title('Batch of 20 samples, alpha=0.00007', fontsize=10)
ax12.hist(error50_00007_B)
ax12.set_title('Batch of 50 samples, alpha=0.00007', fontsize=10)
ax21.hist(error100_00007_B)
ax21.set_title('Batch of 100 samples, alpha=0.00007', fontsize=10)
ax22.hist(error200_00007_B)
ax22.set_title('Batch of 200 samples, alpha=0.00007', fontsize=10)

a=0.0003
error20_0003_B=error(initial_m, initial_c, px20, py20, a)
error50_0003_B=error(initial_m, initial_c, px50, py50,a)
error100_0003_B=error(initial_m, initial_c, px100, py100, a)
error200_0003_B=error(initial_m, initial_c, px200, py200, a)
fig, ((ax11, ax12), (ax21, ax22))= plt.subplots(2,2, figsize=(16,16))
ax11.hist(error20_0003_B)
ax11.set_title('Batch of 20 samples, alpha=0.0003', fontsize=10)
ax12.hist(error50_0003_B)
ax12.set_title('Batch of 50 samples, alpha=0.0003', fontsize=10)
ax21.hist(error100_0003_B)
ax21.set_title('Batch of 100 samples, alpha=0.0003', fontsize=10)
ax22.hist(error200_0003_B)
ax22.set_title('Batch of 200 samples, alpha=0.0003', fontsize=10)
   
#For Stochastic
px20_1, py20_1=samp_from(px20,py20)
px50_1, py50_1=samp_from(px50,py50)
px100_1, py100_1=samp_from(px100,py100)
px200_1, py200_1=samp_from(px200,py200)
a=0.00007
error20_00007_S=error(initial_m, initial_c, px20_1, py20_1, a)
error50_00007_S=error(initial_m, initial_c, px50_1, py50_1,a)
error100_00007_S=error(initial_m, initial_c, px100_1, py100_1, a)
error200_00007_S=error(initial_m, initial_c, px200_1, py200_1, a)
fig, ((ax11, ax12), (ax21, ax22))= plt.subplots(2,2, figsize=(16,16))
ax11.hist(error20_00007_S)
ax11.set_title('SGD of 20 samples, alpha=0.00007', fontsize=10)
ax12.hist(error50_00007_S)
ax12.set_title('SGD of 50 samples, alpha=0.00007', fontsize=10)
ax21.hist(error100_00007_S)
ax21.set_title('SGD of 100 samples, alpha=0.00007', fontsize=10)
ax22.hist(error200_00007_S)
ax22.set_title('SGD of 200 samples, alpha=0.00007', fontsize=10)

a=0.0003
error20_0003_S=error(initial_m, initial_c, px20_1, py20_1, a)
error50_0003_S=error(initial_m, initial_c, px50_1, py50_1,a)
error100_0003_S=error(initial_m, initial_c, px100_1, py100_1, a)
error200_0003_S=error(initial_m, initial_c, px200_1, py200_1, a)
fig, ((ax11, ax12), (ax21, ax22))= plt.subplots(2,2, figsize=(16,16))
ax11.hist(error20_0003_S)
ax11.set_title('SGD of 20 samples, alpha=0.0003', fontsize=10)
ax12.hist(error50_0003_S)
ax12.set_title('SGD of 50 samples, alpha=0.0003', fontsize=10)
ax21.hist(error100_0003_S)
ax21.set_title('SGD of 100 samples, alpha=0.0003', fontsize=10)
ax22.hist(error200_0003_S)
ax22.set_title('SGD of 200 samples, alpha=0.0003', fontsize=10)


# #Part three
'''Now compare running time between Batch, SGD with 20 and 200 samples
and finaly Mini-Batch with 5, 10, 20 samples from 20 and from 200'''
plt.scatter(px20, py20, color='black')
print('20 samples')
print('BATCH')
m=initial_m
c=initial_c
now=datetime.now()
for step in range(1800):
    m,c = gradient_descent_one_step(m,c,px20,py20,0.01)
time=datetime.now()-now
print(MSE(m,c,px20,py20))
print(time, '\n')
plt.plot(px20, px20*m+c, color='red')

print('SGD')
m=initial_m
c=initial_c
now=datetime.now()
x, y=samp_from(px20, py20)
for step in range(1800):
    m,c = gradient_descent_one_step(m,c,x,y,0.01)
time=datetime.now()-now
print(MSE(m,c,px20,py20))
print(time, '\n')
plt.plot(px20, px20*m+c, color='blue')
plt.legend(['Batch', 'SGD'])

plt.scatter(px200, py200, color='black')
print('200 samples')
print('BATCH')
m=initial_m
c=initial_c
now=datetime.now()
for step in range(1800):
    m,c = gradient_descent_one_step(m,c,px200,py200,0.01)
print(MSE(m,c,px200,py200))
time=datetime.now()-now
plt.plot(px200, px200*m+c, color='red')
print(time,'\n')

print('SGD')
m=initial_m
c=initial_c
now=datetime.now()
x, y=samp_from(px200, py200)
for step in range(1800):
    m,c = gradient_descent_one_step(m,c,x,y,0.01)
print(MSE(m,c,px200,py200))
time=datetime.now()-now
plt.plot(px200, px200*m+c, color='blue')
plt.legend(['Batch', 'SGD'])
print(time, '\n')

print('MINI BATCH FROM 20')
plt.scatter(px20, py20, color='black')
print('take 5 samples')
m=initial_m
c=initial_c
now=datetime.now()
random_indexes=random.sample(range(len(px20)-1), 5)
x=np.take(px20, random_indexes)
y=np.take(py20, random_indexes)    
for step in range(1800):
    m,c = gradient_descent_one_step(m,c,x,y,0.01)
time=datetime.now()-now
print(time,'\n')
plt.plot(px20,px20*m+c, color='red')

print('take 10 samples')
m=initial_m
c=initial_c
now=datetime.now()
random_indexes=random.sample(range(len(px20)-1), 10)
x=np.take(px20, random_indexes)
y=np.take(py20, random_indexes)
for step in range(1800):
    m,c = gradient_descent_one_step(m,c,x,y,0.01)
time=datetime.now()-now
print(time,'\n')
plt.plot(px20,px20*m+c, color='blue')

print('take 20 samples')
m=initial_m
c=initial_c
now=datetime.now()
x=px20
y=py20
for step in range(1800):
    m,c = gradient_descent_one_step(m,c,x,y,0.01)
time=datetime.now()-now
print(time,'\n')
plt.plot(px20,px20*m+c, color='yellow')
plt.legend(['5 samples', '10 samples', '20 samples'])

print('MINI BATCH FROM 200')
plt.scatter(px200, py200, color='black')
print('take 5 samples')
m=initial_m
c=initial_c
now=datetime.now()
random_indexes=random.sample(range(len(px200)-1), 5)
x=np.take(px200, random_indexes)
y=np.take(py200, random_indexes)    
for step in range(1800):
    m,c = gradient_descent_one_step(m,c,x,y,0.01)
time=datetime.now()-now
print(MSE(m,c, px200, py200))
print(time,'\n')
plt.plot(px200, px200*m+c, color='red')

print('take 10 samples')
m=initial_m
c=initial_c
now=datetime.now()
random_indexes=random.sample(range(len(px200)-1), 10)
x=np.take(px200, random_indexes)
y=np.take(py200, random_indexes)    
for step in range(1800):
    m,c = gradient_descent_one_step(m,c,x,y,0.01)
time=datetime.now()-now
print(MSE(m,c, px200, py200))
print(time,'\n')
plt.plot(px200, px200*m+c, color='blue')

print('take 20 samples')
m=initial_m
c=initial_c
now=datetime.now()
random_indexes=random.sample(range(len(px200)-1), 20)
x=np.take(px200, random_indexes)
y=np.take(py200, random_indexes)    
for step in range(1800):
    m,c = gradient_descent_one_step(m,c,x,y,0.01)
time=datetime.now()-now
print(MSE(m,c, px200, py200))
print(time,'\n')
plt.plot(px200,px200*m+c, color='yellow')
plt.legend(['5 samples', '10 samples', '20 samples'])

print('FINIHED')