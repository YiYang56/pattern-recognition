import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold

def k_means(n,k,m,data):
    global ind
    z = np.zeros((k, n))
    for i in range(k):
        z[i] = data[np.random.randint(0, n)]
    t=np.zeros((k,n))

    while (1):
        w1 = np.zeros((1, n))
        w2 = np.zeros((1, n))
        ind=np.zeros(m)
        for i in range(m):
            d=np.zeros(k)
            for j in range(k):
                d[j]=np.linalg.norm(data[i] - z[j])
            flag=d.argmin()
            ind[i] = flag
            if flag==0:
                w1 = np.row_stack((w1, data[i]))
            else:
                w2 = np.row_stack((w2, data[i]))
        w1 = np.delete(w1, 0, axis=0)
        w2 = np.delete(w2, 0, axis=0)

        # 记录旧的聚类中心后更新聚类中心
        t[0]=z[0]
        t[1] = z[1]
        if len(w1!=0):
            z[0] = np.mean(w1, axis=0)
        if len(w2 != 0):
            z[1] = np.mean(w2, axis=0)
        if (t[0] == z[0]).all() and (t[1] == z[1]).all():
            break
    # 以下为画图部分
    w = np.vstack((w1, w2))
    label1 = np.zeros((len(w1), 1))
    label2 = np.ones((len(w2), 1))
    label = np.vstack((label1, label2))
    label = np.ravel(label)
    plot_PCA(w, label)

    return ind

def plot_PCA(*data):
    X,Y=data
    pca=decomposition.PCA(n_components=2)
    pca.fit(X)
    X_r=pca.transform(X)
 #   print(X_r)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors=((1,0,0),(0.33,0.33,0.33),)
    for label,color in zip(np.unique(Y),colors):
        position=Y==label
        ax.scatter(X_r[position,0],X_r[position,1],label="category=%d"%label,color=color)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    plt.show()

def liu(yui):
    q=np.zeros(2)
    for i in range(len(yui)):
        if yui[i]==0:
            q[0]+=1
        else:
            q[1]+=1
    return max(q)

# 分别定义样本的特征数、类别数、样本总数
n,k,m=60,2,208
# 读入数据并存为numpy数组形式
path = r'sonar.all-data'
gakki = pd.read_csv(path, header=None)
sonar=gakki.values[0:208,0:60]
sonar=np.array(sonar)
idx=k_means(n,k,m,sonar)
idx1=idx[0:97]
idx2=idx[97:208]
num=np.zeros(2)
num[0]=liu(idx1)
num[1]=liu(idx2)
print('每一类聚类正确的个数分别为：{}/97、{}/111,正确率分别为{}、{}'.format(num[0],num[1],num[0]/97,num[1]/111))