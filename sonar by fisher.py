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
        for i in range(0, 96):
            a = A[i, :] - m1
            a = np.array([a])
            b = a.T
            s1 = s1 + np.dot(b, a)
        for i in range(0, 111):
            a = B[i, :] - m2
            a = np.array([a])
            b = a.T
            s2 = s2 + np.dot(b, a)
    if flag==2:
        for i in range(0, 97):
            a = A[i, :] - m1
            a = np.array([a])
            b = a.T
            s1 = s1 + np.dot(b, a)
        for i in range(0, 110):
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

path = r'sonar.all-data'
gakki = pd.read_csv(path, header=None)
Accuracy = np.zeros(60)
y1,y2=[],[]

for n in range(1, 61):  # n是当前的维数
    sonar1 = gakki.values[0:97,0:n]
    sonar2 = gakki.values[97:208,0:n]
    count=0
    for i in range(208):
        if i <= 96:
            test = sonar1[i]
            test = test.reshape(n, 1)
            train = np.delete(sonar1, i, axis=0)  # 训练样本是一个列数为t的矩阵
            w, w0 = fisher(train, sonar2, n, 1)
            if np.dot(w,test)>=w0:
                count+=1
            if n == 60:
                y1.append(np.dot(w, test)[0][0])
        else:
            test = sonar2[i - 97]
            test = test.reshape(n, 1)
            train = np.delete(sonar2, i - 97, axis=0)
            w, w0 = fisher(sonar1, train, n, 2)
            if np.dot(w,test)<w0:
                count+=1
            if n == 60:
                y2.append(np.dot(w, test)[0][0])
    Accuracy[n-1]=count/208
    print("%d维时，准确率为:%.3f" % (n, Accuracy[n-1]))

# 画图部分
x = np.arange(1, 61, 1)
plt.xlabel('dimension')
plt.ylabel('Accuracy')
plt.ylim((0.5, 1.0))
plt.plot(x, Accuracy, 'b')
plt.show()

y1=np.array(y1)
y2=np.array(y2)
y11=np.zeros((97,))
y22=np.ones((111,))
plt.plot(y1,y11, "ro")
plt.plot(y2,y22, "bo")
plt.legend(['R','M'])
plt.show()