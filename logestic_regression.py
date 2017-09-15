import numpy as np

fil = open('../ex2data2.txt','r')

nof=2
n=nof+1

m=118
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
    z=np.dot(np.transpose(x),theta)
    return 1.0/(1.0+np.exp(-z))

def cost():
    global x,y,m,theta
    h=hyp()
    return -(1.0/m)*(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))

def update_theta():
        global x,y,theta,m
        for it in range(num_iter):
            ha=(hyp()-y).reshape(m,1)
            val=alpha*(1.0/m)*np.dot(x,ha)
            theta=theta-val.reshape(n)

        print 'theta = ',theta,'cost = ',cost()


def predict(aa):
    global theta
    z=np.sum(np.multiply(aa,theta))
    return 1.0/(1.0+np.exp(-z))
print hyp().shape
print 'Inital Cost = ',cost()
update_theta()
predict(np.array([1,45,85]))
