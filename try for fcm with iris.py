import numpy as np
import pandas as pd

# 准则函数
def J(u,z,x):
    J=0
    for j in range(c):
        for i in range(m):
            J+=(u[i,j]**2)*(np.linalg.norm(x[i,:]-z[j,:])**2)
    return J

# 修改聚类中心
def change_z(z,data,u):
    for j in range(c):
        a,b=0,0
        for i in range(m):
            a+=u[i,j]**2*data[i,:]
            b+=u[i,j]**2
        z[j]=a/b

# 修改隶属度矩阵
def change_u(u,data,z):
    for i in range(m):
        for j in range(c):
            t=0
            for k in range(c):
                t+=(np.linalg.norm(data[i,:]-z[j,:])/np.linalg.norm(data[i,:]-z[k,:]))**2
            u[i,j]=1/t

def fcm(data):
    z=np.zeros((c,4))
    u = np.random.random((m, c))
    change_z(z,data,u)
    t=0
    while abs(J(u,z,data)-t)>0.00001:
        t = J(u, z, data)
        change_u(u,data,z)
        change_z(z,data,u)
    return u

def test(flag,idx):
    a,b,c=0,0,0
    for i in range(flag*50,(flag+1)*50):
        if idx[i]==0:
            a+=1
        elif idx[i]==1:
            b+=1
        else:
            c+=1
    return max(a,b,c)


path = r'iris.data'
gakki = pd.read_csv(path, header=None)
iris=gakki.values[0:150,0:4]
iris=np.array(iris)
m,c=150,3
u=fcm(iris)
idx = np.argmax(u, axis=1)
L=[]
for i in range(3):
    L.append(test(i,idx))
print('每一类聚类正确的个数分别为：{}、{}、{},正确率分别为{}、{}、{}'.format(L[0],L[1],L[2],L[0]/50,L[1]/50,L[2]/50))