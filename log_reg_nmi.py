import numpy as np
import scipy.io
import cv2
import time

np.set_printoptions(suppress=True)

def show_img(img):
    t=img.copy()
    t=t.reshape(20,20)
    t=cv2.resize(t, (200, 200))
    cv2.imshow('image',t)
    cv2.resizeWindow('image', 200,200)
    cv2.waitKey(0)
    cv2.destroyAllWindows

mat = scipy.io.loadmat('ex3data1.mat')
m=5000
y=mat['y']
_x=mat['X']

_y=np.zeros([5000,10])

for i in range(5000):
    if y[i][0]==10:
        _y[i][0]=1
    else:
        _y[i][y[i][0]]=1

theta=np.zeros([10,400])

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def hyp():
    global theta,_x
    z=np.dot(_x,np.transpose(theta))
    return sigmoid(z)

def cost():
    global _x,_y,m,theta
    h=hyp()
    return -(1.0/m)*np.sum((np.multiply(np.log(h),_y)+np.multiply(np.log(1-h),1-_y)))


def update_theta(alpha=0.01,num_iter=10000):
        global _x,_y,theta,m
        for it in range(num_iter):
            ha=hyp()-_y

            val=alpha*(1.0/m)*np.dot(np.transpose(ha),_x)
            theta=theta-val
            if it%(num_iter/100)==0:
                print 'iterations =',it,'cost = ',cost()
        print 'theta = ',theta
def predict(img_x):
    z=np.dot(img_x,np.transpose(theta))
    return sigmoid(z)
update_theta()
print 'trained successfully'

print predict(_x[i])
show_img(_x[i])
