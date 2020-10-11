import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义fisher函数，输入为两个类别的集合、维度和测试机所属的类别，输出为w*和阈值w0
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

# 定义knn函数，输入为预处理后的数据集和k值，输出为OA和三个AA
def knn(iris,k):
    I = np.zeros((150, 2))

    for i in range(150):
        I[i, 0] = np.dot(iris[i], w)
        if i < 50:
            I[i, 1] = 1
        elif i < 100:
            I[i, 1] = 2
        else:
            I[i, 1] = 3

    accuracy = 0
    a1,a2,a3=0,0,0
    for i in range(150):
        count1 = 0
        count2 = 0
        count3 = 0
        prediction = 0
        test = I[i]
        train = np.delete(I, i, axis=0)
        distance = np.zeros((149, 2))
        for t in range(149):
            distance[t, 1] = np.linalg.norm(test[0] - train[t, 0])
            distance[t, 0] = train[t, 1]
        order = distance[np.lexsort(distance.T)]
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
            prediction = 3  # 取出现次数最多的为预测值
        if prediction == test[1]:
            accuracy += 1
            if i<50:
                a1+=1
            elif i<100:
                a2+=1
            else:
                a3+=1
    return accuracy/150,a1/50,a2/50,a3/50

path=r'iris.data'
gakki = pd.read_csv(path, header=None)

iris=gakki.values[0:150,0:4]
iris1=gakki.values[0:50,0:4]
iris2=gakki.values[50:100,0:4]
iris3=gakki.values[100:150,0:4]
m = np.mean(iris, axis=0)
w=fisher(iris1,iris2,iris3,m,4,1)
k=9
OA,AA1,AA2,AA3=knn(iris,k)
print('总体精度OA为：{}'.format(OA))
print('三类的平均精度AA分别为：{}、{}、{}'.format(AA1,AA2,AA3))