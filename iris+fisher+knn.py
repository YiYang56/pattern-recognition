import pandas as pd
import numpy as np

# 定义fisher函数，输入为所有类别的集合、全局均值、维度和测试集所属的类别，输出为w*
def fisher(A,B,C,m,n,flag):
    m1 = np.mean(A, axis=0)
    m2 = np.mean(B, axis=0)
    m3 = np.mean(C, axis=0)

    s1 = np.zeros((n,n))
    s2 = np.zeros((n,n))
    s3 = np.zeros((n, n))

    if flag==1:
        for i in range(0, 49):
            a = A[i, :] - m1
            a = np.array([a])
            b = a.T
            s1 = s1 + np.dot(b, a)
        for i in range(0, 50):
            a = B[i, :] - m2
            a = np.array([a])
            b = a.T
            s2 = s2 + np.dot(b, a)
        for i in range(0, 50):
            a = C[i, :] - m3
            a = np.array([a])
            b = a.T
            s3 = s3 + np.dot(b, a)
    if flag==2:
        for i in range(0, 50):
            a = A[i, :] - m1
            a = np.array([a])
            b = a.T
            s1 = s1 + np.dot(b, a)
        for i in range(0, 49):
            a = B[i, :] - m2
            a = np.array([a])
            b = a.T
            s2 = s2 + np.dot(b, a)
        for i in range(0, 50):
            a = C[i, :] - m3
            a = np.array([a])
            b = a.T
            s3 = s3 + np.dot(b, a)
    if flag==3:
        for i in range(0, 50):
            a = A[i, :] - m1
            a = np.array([a])
            b = a.T
            s1 = s1 + np.dot(b, a)
        for i in range(0, 50):
            a = B[i, :] - m2
            a = np.array([a])
            b = a.T
            s2 = s2 + np.dot(b, a)
        for i in range(0, 49):
            a = C[i, :] - m3
            a = np.array([a])
            b = a.T
            s3 = s3 + np.dot(b, a)

    sw = s1 + s2 +s3
    sw = np.array(sw, dtype='float')

    m1 = m1.reshape((4, 1))
    m2 = m2.reshape((4, 1))
    m3 = m3.reshape((4, 1))
    m = m.reshape((4, 1))

    w=np.zeros((4,1))
    w=w+(np.dot(np.linalg.inv(sw), (m1-m)))
    w = w + (np.dot(np.linalg.inv(sw), (m2 - m)))
    w = w + (np.dot(np.linalg.inv(sw), (m3 - m)))

    return w

# 定义knn函数，输入为预处理后的数据集、测试集的索引、k值和fisher中求得的投影方向
def knn(iris,g,k,w):
    I = np.zeros((150, 2))
    # 根据w对输入进行投影
    for i in range(150):
        I[i, 0] = np.dot(iris[i], w)
        if i < 50:
            I[i, 1] = 1
        elif i < 100:
            I[i, 1] = 2
        else:
            I[i, 1] = 3

    # 定义全局变量，以计算分类正确个数
    global a1,a2,a3

    count1 = 0
    count2 = 0
    count3 = 0
    prediction = 0
    test = I[g]
    train = np.delete(I, g, axis=0)
    distance = np.zeros((149, 2))
    # 计算距离信息，并将其和对应的标签存入distance数组，最后从大到小进行排序
    for t in range(149):
        distance[t, 1] = np.linalg.norm(test[0] - train[t, 0])
        distance[t, 0] = train[t, 1]
    order = distance[np.lexsort(distance.T)]
    # 计算前k个最近样本的类别，并统计出现次数最多的类别，取出现次数最多的为预测值
    for n in range(k):
        if order[n, 0] == 1:
            count1 += 1
        if order[n, 0] == 2:
            count2 += 1
        if order[n, 0] == 3:
            count3 += 1
    if count1 >= count2 and count1 >= count3:
        prediction = 1
    if count2 >= count1 and count2 >= count3:
        prediction = 2
    if count3 >= count1 and count3 >= count2:
        prediction = 3
    if prediction == test[1]:
        if g<50:
            a1+=1
        elif g<100:
            a2+=1
        else:
            a3+=1

path=r'iris.data'
gakki = pd.read_csv(path, header=None)
iris=gakki.values[0:150,0:4]
iris1=gakki.values[0:50,0:4]
iris2=gakki.values[50:100,0:4]
iris3=gakki.values[100:150,0:4]

a1,a2,a3=0,0,0

# 采用留一法
for i in range(150):
    if i<50:
        test=iris1[i]
        A=np.delete(iris1, i, axis=0)
        B=iris2
        C=iris3
        flag=1
    elif i<100:
        test=iris2[i-50]
        B = np.delete(iris2, i-50, axis=0)
        A=iris1
        C=iris3
        flag=2
    else:
        test=iris3[i-100]
        C = np.delete(iris3, i-100, axis=0)
        A = iris1
        B = iris2
        flag=3
    m = np.mean(np.delete(iris, i, axis=0), axis=0)
    w = fisher(A, B, C, m, 4, flag)
    k=5
    knn(iris,i,k,w)
print('总体精度OA为：{}'.format((a1+a2+a3)/150))
print('平均精度AA为：{}'.format((a1/50+a2/50+a3/50)/3))
print('每一类分对的个数分别为{}、{}、{}'.format(a1,a2,a3))
