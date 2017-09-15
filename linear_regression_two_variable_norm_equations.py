import numpy as np

fil = open('../data/ex1data2.txt','r')

nof=2
n=nof+1

m=47
alpha=0.001
num_iter=10000

x=np.ones([n,m])
y=np.ones([1,m])
i=0
#obtaining data
for line in fil:
    t=line.strip('\n').split(',')
    for j in range(nof):
        x[:,i][j+1]=float(t[j])
    y[:,i][0]=float(t[nof])
    i= i+1


#normalizing data
for i in range(n):
    rang=float(abs(max(x[i])))
    if(rang!=0):
        x[i]=(1.0/rang)*x[i]

y[0]=(1.0/max(y[0]))*y[0]


norm_fact=max(y[0])

theta=np.zeros(n)

def hyp():
    global x,theta
    return np.dot(np.transpose(x),theta)

def cost():
    global x,y,m
    return (1.0/(2.0*m))*np.sum(np.square(hyp()-y))


def update_theta():
    val1=np.dot(x,np.transpose(x))
    val1=np.linalg.inv(val1)

    val2=np.dot(y,np.transpose(x))

    print val1.shape,val2.shape
    theta=np.dot(val2,val1)
    print 'cost = ',cost()
    print 'theta = ',theta



update_theta()
