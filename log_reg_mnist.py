
# coding: utf-8

# In[3]:


import numpy as np
import scipy.io
import cv2
import time


# In[110]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[111]:


def plot_onehidden(image):
    plt.axis('off')
    plt.imshow(image.reshape([20,20]),cmap='gray')


# In[4]:


np.set_printoptions(suppress=True)


# In[5]:


def show_img(img):
    t=img.copy()
    t=t.reshape(20,20)
    t=cv2.resize(t, (200, 200))
    cv2.imshow('image',t)
    cv2.resizeWindow('image', 200,200)
    cv2.waitKey(0)
    cv2.destroyAllWindows


# In[103]:


mat = scipy.io.loadmat('..\pg_Eg\my_projects\ex3data1.mat')

m=5000
y=mat['y']
x=mat['X']
_y=np.zeros([5000,10])
theta=np.zeros([10,400])
_x=x


# In[104]:


def create_training_and_test():
    
    for i in range(5000):
        if y[i][0]==10:
            _y[i][0]=1
        else:
            _y[i][y[i][0]]=1


# In[105]:


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



# In[106]:


def update_theta(alpha=0.5,num_iter=10000):
        global _x,_y,theta,m
        for it in range(num_iter):
            ha=hyp()-_y

            val=alpha*(1.0/m)*np.dot(np.transpose(ha),_x)
            theta=theta-val
            if it%(num_iter/100)==0:
                print 'iterations =',it,'cost = ',cost()
        print 'theta = ',theta


# In[116]:


def predict(img_x):
    ret=sigmoid(np.dot(img_x,np.transpose(theta)))
    return ret,np.argmax(ret)


# In[108]:


create_training_and_test()


# In[109]:


update_theta()
print 'trained successfully'


# In[16]:


print cost()
print theta


# In[146]:


no=np.random.randint(5000)
print y[no]
print predict(_x[no])
plot_onehidden(_x[no])

