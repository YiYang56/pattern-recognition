import numpy as np
import math as m

# n:变量数  a、b:上下界 k:设定迭代次数
n,a,b,k=2,-500,500,10000
# 算法相关超参数
c1,c2,N,w=2,2,40,0.5

# 计算待验证的函数值（根据需要验证的函数形式修改）
def fun(x):
    s=0
    for i in range(n):
        s+=(-1*x[i]*m.sin((abs(x[i]))**0.5))
    return s

def PSO(N,n,a,b,k):
    # 对种群进行初始化
    x=np.random.uniform(a,b,(N,n))
    v=np.random.uniform(a,b,(N,n))
    f=np.zeros(N)
    p_best=np.zeros((N,n))
    for i in range(N):
        f[i]=fun(x[i])
    for i in range(N):
        for j in range(n):
            p_best[i][j]=x[i][j]
    g_best=x[np.argmin(f)]
    # 开始迭代
    for i in range(k):
        for j in range(N):
            # 更新粒子速度和位置
            v[j]=w*v[j]+c1*np.random.uniform(0,1)*(p_best[j]-x[j])+c2*np.random.uniform(0,1)*(g_best-x[j])
            # 防止粒子速度越界
            for t in range(n):
                if v[j][t] > b:
                    v[j][t] = b
                if v[j][t] < a:
                    v[j][t] = a
            x[j]=x[j]+v[j]
            # 限制范围防止越界
            for t in range(n):
                if x[j][t]>b:
                    x[j][t]=b
                if x[j][t]<a:
                    x[j][t]=a
        # 修改适应度函数值
        for j in range(N):
            if f[j]>fun(x[j]):
                p_best[j]=x[j]
                f[j]=fun(x[j])
        # 更新全局最优位置
        g_best = x[np.argmin(f)]
    return g_best

g=PSO(N,n,a,b,k)
print(fun(g))