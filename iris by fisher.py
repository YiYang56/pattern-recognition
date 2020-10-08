import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义fisher函数，输入为两个类别的集合、维度和测试机所属的类别，输出为w*和阈值w0
def fisher(A,B,n,flag):
    m1 = np.mean(A, axis=0)
    m2 = np.mean(B, axis=0)

    s1 = np.zeros((n,n))
    s2 = np.zeros((n,n))

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

    sw = s1 + s2
    a = np.array([m1 - m2])
    sw = np.array(sw, dtype='float')

    # np.linalg.inv() numpy求逆函数
    w = (np.dot(np.linalg.inv(sw), a.T)).T
    w0 = (np.dot(w, m1) + np.dot(w, m2)) / 2

    return w,w0

path=r'iris.data'
gakki = pd.read_csv(path, header=None)

iris1=gakki.values[0:50,0:4]
iris2=gakki.values[50:100,0:4]
iris3=gakki.values[100:150,0:4]
A11,A12,A21,A22,A31,A32=[],[],[],[],[],[]

# 第一类&第二类
count=0
for i in range(100):
    if i < 50:
        test = iris1[i]
        test = test.reshape(4, 1)
        train = np.delete(iris1, i, axis=0)  # 训练样本是一个列数为t的矩阵
        w, w0 = fisher(train, iris2, 4, 1)
        if np.dot(w, test) >= w0:
            count += 1
        A11.append(np.dot(w, test)[0][0])
    else:
        test = iris2[i - 50]
        test = test.reshape(4, 1)
        train = np.delete(iris2, i - 50, axis=0)
        w, w0 = fisher(iris1, train, 4, 2)
        if np.dot(w, test) < w0:
            count += 1
        A12.append(np.dot(w, test)[0][0])
print("第一类和第二类的分类准确率为:%.3f" % (count/100))

# 第一类&第三类
count=0
for i in range(100):
    if i < 50:
        test = iris1[i]
        test = test.reshape(4, 1)
        train = np.delete(iris1, i, axis=0)  # 训练样本是一个列数为t的矩阵
        w, w0 = fisher(train, iris3, 4, 1)
        if np.dot(w, test) >= w0:
            count += 1
        A21.append(np.dot(w, test)[0][0])
    else:
        test = iris3[i - 50]
        test = test.reshape(4, 1)
        train = np.delete(iris3, i - 50, axis=0)
        w, w0 = fisher(iris1, train, 4, 2)
        if np.dot(w, test) < w0:
            count += 1
        A22.append(np.dot(w, test)[0][0])
print("第一类和第三类的分类准确率为:%.3f" % (count/100))

# 第二类&第三类
count=0
for i in range(100):
    if i < 50:
        test = iris2[i]
        test = test.reshape(4, 1)
        train = np.delete(iris2, i, axis=0)  # 训练样本是一个列数为t的矩阵
        w, w0 = fisher(train, iris3, 4, 1)
        if np.dot(w, test) >= w0:
            count += 1
        A31.append(np.dot(w, test)[0][0])
    else:
        test = iris3[i - 50]
        test = test.reshape(4, 1)
        train = np.delete(iris3, i - 50, axis=0)
        w, w0 = fisher(iris2, train, 4, 2)
        if np.dot(w, test) < w0:
            count += 1
        A32.append(np.dot(w, test)[0][0])
print("第一类和第三类的分类准确率为:%.3f" % (count/100))

# 画图部分
A11=np.array(A11)
A12=np.array(A12)
A21=np.array(A21)
A22=np.array(A22)
A31=np.array(A31)
A32=np.array(A32)
y1=np.zeros((50,))
y2=np.ones((50,))

plt.figure(1)
plt.plot(A11,y1, "ro")
plt.plot(A12,y2, "bo")
plt.legend(['setosa','versicolor'])

plt.figure(2)
plt.plot(A21,y1, "ro")
plt.plot(A22,y2, "bo")
plt.legend(['setosa','virginica'])

plt.figure(3)
plt.plot(A31,y1, "ro")
plt.plot(A32,y2, "bo")
plt.legend(['versicolor','versicolor'])
plt.show()