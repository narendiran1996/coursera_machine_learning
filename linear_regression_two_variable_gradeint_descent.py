import numpy as np

fil = open('../data/ex1data2.txt','r')

nof=2
n=nof+1

m=47
alpha=0.005
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

rang=[]
#normalizing data
for i in range(n):
    rang.append(float(abs(max(x[i]))))
    if(rang!=0):
        x[i]=(1.0/rang[i])*x[i]

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
    global x,y,theta,m
    for it in range(num_iter):
        ha=(hyp()-y).reshape(m,1)
        val=alpha*(1.0/m)*np.dot(x,ha)
        theta=theta-val.reshape(n)
    print 'theta = ',theta
    print 'cost = ',cost()

update_theta()
