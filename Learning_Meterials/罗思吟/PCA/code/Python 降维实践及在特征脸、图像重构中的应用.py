#!/usr/bin/env python
# coding: utf-8

# # 目录
# 1. [PCA 算法的 Python 实现](#1)<br> 
# 2. [基于 PCA 的特征脸提取和人脸重构](#2)<br> 
# &emsp;&emsp;2.1 [加载 olivettifaces 人脸数据集](#2.1)<br>
# &emsp;&emsp;2.2 [使用 PCA 对人脸进行降维和特征脸提取](#2.2)<br> 
# &emsp;&emsp;2.3 [基于特征脸和平均脸的人脸重构](#2.3)<br>
# 3. [使用 MNIST 手写数字数据集理解 PCA 主成分的含义](#3)<br> 
# &emsp;&emsp;3.1 [加载 MNIST 数据集](#3.1)<br>
# &emsp;&emsp;3.2 [使用 PCA 对 MNIST 进行降维](#3.2)<br> 
# &emsp;&emsp;3.3 [基于手写数字可视化理解主成分含义](#3.3)<br> 
# 4. [基于 AutoEncoder 的图像压缩与重构](#4)<br> 
# &emsp;&emsp;4.1 [使用 TensorFlow 构建自编码器](#4.1)<br>
# &emsp;&emsp;4.2 [使用 TensorFlow 构建多层自编码器](#4.2)<br> 

# ## <a id=1></a>1 PCA 算法的 Python 实现
# Numpy 的 `linalg` 模块实现了常见的线性代数运算，包括矩阵的特征值求解。其中 `eig` 函数能够计算出给定方阵的特征值和对应的右特征向量。我们实现函数 `principal_component_analysis`，其输入为数据集 $X$ 和主成分数量 $l$，返回降维后的数据、 $l$ 个主成分列表和对应的特征值列表。主成分按照特征值大小降序排序。

# In[2]:


import numpy as np
def principal_component_analysis(X, l):
    X = X - np.mean(X, axis=0)#对原始数据进行中心化处理
    sigma = X.T.dot(X)/(len(X)-1) # 计算协方差矩阵
    a,w = np.linalg.eig(sigma)# 计算协方差矩阵的特征值和特征向量
    sorted_index = np.argsort(-a)# 将特征向量按照特征值进行排序
    X_new = X.dot(w[:,sorted_index[0:l]])#对数据进行降维
    return X_new,w[:,sorted_index[0:l]],a[sorted_index[0:l]]#返回降维后的数据、主成分、对应特征值


# 生成一份随机的二维数据集。为了直观查看降维效果，我们借助 `make_regression` 生成一份用于线性回归的数据集。将自变量和标签进行合并，组成一份二维数据集。同时对两个维度均进行归一化。

# In[3]:


from sklearn import datasets
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
x, y = datasets.make_regression(n_samples=200,n_features=1,noise=10,bias=20,random_state=111)
x = (x - x.mean())/(x.max()-x.min())
y = (y - y.mean())/(y.max()-y.min())
fig, ax = plt.subplots(figsize=(6, 6)) #设置图片大小
ax.scatter(x,y,color="#E4007F",s=50,alpha=0.4)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


# 使用 PCA 对数据集进行降维

# In[4]:


import pandas as pd
X = pd.DataFrame(x,columns=['x1'])
X['x2'] = y
X_new,w,a = principal_component_analysis(X,1)


# 将第一个主成分方向的直线绘制出来。直线的斜率为 `w[1,0]/w[0,0]`。将主成分方向在散点图中绘制出来。

# In[5]:


import numpy as np
x1 = np.linspace(-.5, .5, 50)
x2 = (w[1,0]/w[0,0])*x1 
fig, ax = plt.subplots(figsize=(6, 6)) #设置图片大小
X = pd.DataFrame(x,columns=["x1"])
X["x2"] = y
ax.scatter(X["x1"],X["x2"],color="#E4007F",s=50,alpha=0.4)
ax.plot(x1,x2,c="gray") # 画出第一主成分直线
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


# 我们还可以将降维后的数据集使用散点图进行绘制

# In[6]:


import numpy as np
fig, ax = plt.subplots(figsize=(6, 2)) #设置图片大小
ax.scatter(X_new,np.zeros_like(X_new),color="#E4007F",s=50,alpha=0.4)
plt.xlabel("First principal component")


# # <a id=2></a>2 基于 PCA 的特征脸提取和人脸重构

# ## <a id=2.1></a>2.1 加载 olivettifaces 人脸数据集

# 数据集包括40个不同的对象，每个对象都有10个不同的人脸图像。对于某些对象，图像是在不同的时间、光线、面部表情（睁眼/闭眼、微笑/不微笑）和面部细节（眼镜/不戴眼镜）下拍摄。所有的图像都是在一个深色均匀的背景下拍摄的，被摄者处于直立的正面位置（可能有细微面部移动）。原始数据集图像大小为 $92 \times 112$，而 Roweis 版本图像大小为 $64 \times 64$。
# 
# 首先，我们使用 Sklearn 实现的方法读取人脸数据集。

# In[7]:


from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()
faces.data.shape


# 选取部分人脸，使用 `matshow` 函数将其可视化

# In[8]:


rndperm = np.random.permutation(len(faces.data))
plt.gray()
fig = plt.figure(figsize=(9,4) )
for i in range(0,18):
    ax = fig.add_subplot(3,6,i+1 )
    ax.matshow(faces.data[rndperm[i],:].reshape((64,64)))
    plt.box(False) #去掉边框
    plt.axis("off")#不显示坐标轴
plt.show()


# ## <a id=2.2></a> 2.2 使用 PCA 对人脸进行降维和特征脸提取

# 使用 PCA 将人脸数据降维到 20 维。其中 `%time` 是由 IPython 提供的魔法命令，能够打印出函数的执行时间。

# In[9]:


get_ipython().run_line_magic('time', 'faces_reduced,W,lambdas = principal_component_analysis(faces.data,20)')


# PCA 中得到的转换矩阵 $\mathbf{W}$ 是一个 $d \times l$ 矩阵。其中每一列称为一个主成分。在人脸数据集中，得到的 20 个主成分均为长度为 4096 的向量。我们可以将其形状转换成 $64 \times 64$，这样每一个主成分也可以看作是一张人脸图像，在图像分析领域，称为**特征脸**。使用 `matshow` 函数，将特征脸进行可视化如下。

# In[58]:


fig = plt.figure( figsize=(18,4))
plt.gray()
for i in range(0,20):
    ax = fig.add_subplot(2,10,i+1 )
    ax.matshow(W[:, i].reshape((64,64)))
    plt.title("Face(" + str(i) + ")")
    plt.box(False) #去掉边框
    plt.axis("off")#不显示坐标轴
plt.show()


# ## <a id=2.3></a>2.3 基于特征脸和平均脸的人脸重构

# 降维后数据每一个维度的作用，可以看做每一个主成分的权重。将原始图片的平均值加上不同主成分的权重和，可以对图像进行重构，如下图示例。
# <img src="http://cookdata.cn/media/note_images/图片_1_1592432730992_5d14.jpg" width = "600" height = "400"/>
# 
# 我们来随机选择一些人脸图片，观察一下图片的重构效果。

# In[61]:


sample_indx = np.random.randint(0,len(faces.data)) #随机选择一个人脸的索引
#显示原始人脸
plt.matshow(faces.data[sample_indx].reshape((64,64)))


# 注意，由于 PCA 算法对图像进行了中心化，在重构人脸时还需要加上数据的平均值（这里称为平均脸）。

# In[62]:


# 显示重构人脸
plt.matshow(faces.data.mean(axis=0).reshape((64,64)) + W.dot(faces_reduced[sample_indx]).reshape((64,64)))


# 对于每一个样本，降维后的每一个维度的取值代表的是对应主成分的权重。实际上就是利用不同特征脸的权重和再加上平均脸，才对原始人脸进行了重构。我们将上述样本的重构过程用图画出来。

# In[63]:


fig = plt.figure( figsize=(20,4))
plt.gray()
ax = fig.add_subplot(2,11,1)
ax.matshow(faces.data.mean(axis=0).reshape((64,64))) #显示平均脸
for i in range(0,20):
    ax = fig.add_subplot(2,11,i+2 )
    ax.matshow(W[:,i].reshape((64,64)))
    plt.title( str(round(faces_reduced[sample_indx][i],2)) + "*Face(" + str(i) + ")")
    plt.box(False) #去掉边框
    plt.axis("off")#不显示坐标轴
plt.show()


# # <a id=3></a> 3 使用 MNIST 手写数字数据集理解 PCA 主成分的含义
# MNIST 手写数字数据集是在图像处理和深度学习领域一个著名的图像数据集。该数据集包含一份 60000 个图像样本的训练集和包含 10000 个图像样本的测试集。每一个样本是 $28 \times 28$ 的图像，每个图像有一个标签，标签取值为 0-9 。 MNIST 数据集下载地址为  [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)。

# ## <a id=3.1></a>3.1 加载 MNIST 数据集

# 我们已经将数据集下载到 `input/mnist.npz`，可以直接读取。

# In[14]:


import numpy as np
f = np.load("C:/Users/17154/input/mnist.npz") 
X_train, y_train, X_test, y_test = f['x_train'], f['y_train'],f['x_test'], f['y_test']
f.close()


# 将图像拉平成 784 维的向量表示，并对像素值进行归一化。

# In[15]:


x_train = X_train.reshape((-1, 28*28)) / 255.0
x_test = X_test.reshape((-1, 28*28)) / 255.0


# 筛选出一个数字 `8` 的数据。

# In[16]:


digit_number = x_train[y_train == 8]


# 随机选择部分数据进行可视化展示

# In[17]:


rndperm = np.random.permutation(len(digit_number))
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.gray()
fig = plt.figure( figsize=(12,12) )
for i in range(0,100):
    ax = fig.add_subplot(10,10,i+1)
    ax.matshow(digit_number[rndperm[i]].reshape((28,28)))
    plt.box(False) #去掉边框
    plt.axis("off")#不显示坐标轴  
plt.show()


# ## <a id=3.2></a> 3.2 使用 PCA 对 MNIST 进行降维

# 利用我们实现的 PCA 算法函数 `principal_component_analysis` 将数字 `8` 的图片降维到二维。

# In[18]:


number_reduced,number_w,number_a = principal_component_analysis(digit_number,2)


# 将降维后的二维数据集使用散点图进行可视化。

# In[19]:


import warnings
warnings.filterwarnings('ignore') #该行代码的作用是隐藏警告信息
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize=(8, 8)) #设置图片大小
ax.scatter(number_reduced[:,0],number_reduced[:,1],color="#E4007F",s=20,alpha=0.4)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")


# ## <a id=3.3></a> 3.3 基于手写数字可视化理解主成分含义

# 我们首先提取降维之后的数据 `number_reduced` 的第一列数据 `number_reduced[:,0]`，代表每一个样本在第一个主成分上的取值大小。按照取值从小到大进行排序，抽样提取原始的手写数字图片并进行可视化。

# In[21]:


sorted_indx = np.argsort(number_reduced[:,0])
print(number_reduced.shape)
plt.gray()
fig = plt.figure( figsize=(12,12) )
for i in range(0,100):
    ax = fig.add_subplot(10,10,i+1)
    ax.matshow(digit_number[sorted_indx[i*50]].reshape((28,28)))
    plt.box(False) #去掉边框
    plt.axis("off")#不显示坐标轴  
plt.show()


# 观察上图，可以看到数字 `8` 顺时针倾斜的角度不断加大。可见，第一个主成分的含义很可能代表的就是数字的倾斜角度。
# 
# 同理，我们观察第二主成分的结果。

# In[75]:


sorted_indx = np.argsort(number_reduced[:,1])
plt.gray()
fig = plt.figure( figsize=(12,12) )
for i in range(0,100):
    ax = fig.add_subplot(10,10,i+1)
    ax.matshow(digit_number[sorted_indx[i*50]].reshape((28,28)))
    plt.box(False) #去掉边框
    plt.axis("off")#不显示坐标轴  
plt.show()


# # <a id=4></a> 4 基于 AutoEncoder 的图像压缩与重构

# 从 MNIST 手写数字训练集中采样一个图像子集，使用 Matplotlib 中的 `matshow` 方法将图像可视化。

# In[77]:


rndperm = np.random.permutation(len(X_train))
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.gray()
fig = plt.figure(figsize=(16,7))
for i in range(0,30):
    ax = fig.add_subplot(3,10,i+1, title='Label:' + str(y_train[rndperm[i]]) )
    ax.matshow(X_train[rndperm[i]])
    plt.box(False) #去掉边框
    plt.axis("off")#不显示坐标轴  
plt.show()


# ## <a id=4.1></a>4.1 使用 TensorFlow 构建自编码器

# TensorFlow 是谷歌公司著名的开源深度学习工具。我们借助改工具可以很方便地进行神经网络的构建和训练。自编码器是本质上是一种特殊结构的全连接神经网络。全连接层在 TensorFlow 中称为 `Dense` 层，可以通过 `tensorflow.keras.layers.Dense` 类进行定义。
# 
# 首先，导入 TensorFlow 对应的模块。

# In[78]:


import tensorflow as tf
import tensorflow.keras.layers as layers


# 将图像拉平成 784 维的向量表示，并对像素值进行归一化。

# In[79]:


x_train = X_train.reshape((-1, 28*28)) / 255.0
x_test = X_test.reshape((-1, 28*28)) / 255.0


# 构建一个自编码器，输入层大小为784，隐含层大小为 10， 输出层大小为 784 。在隐含层使用 ReLU 作为激活函数（非线性映射）。输出层则使用 Softmax 作为激活函数。自编码器模型构建步骤如下：

# In[80]:


inputs = layers.Input(shape=(28*28,), name='inputs')
hidden = layers.Dense(10, activation='relu', name='hidden')(inputs)
outputs = layers.Dense(28*28, name='outputs')(hidden)
auto_encoder = tf.keras.Model(inputs,outputs)
auto_encoder.summary()


# 使用 `compile` 方法对模型进行编译。

# In[81]:


auto_encoder.compile(optimizer='adam',loss='mean_squared_error') #定义误差和优化方法


# 使用 `fit` 在手写数字训练数据集上进行自编码器的训练。

# In[82]:


get_ipython().run_line_magic('time', 'auto_encoder.fit(x_train, x_train, batch_size=100, epochs=100,verbose=0) #模型训练')


# 对训练集中的数据，使用自编码器进行预测，得到重建的图像。然后将重建的图像与原始图像进行对比

# In[83]:


x_train_pred = auto_encoder.predict(x_train)
# Plot the graph
plt.gray()
fig = plt.figure( figsize=(16,4) )
n_plot = 10
for i in range(n_plot):
    ax1 = fig.add_subplot(2,10,i+1, title='Label:' + str(y_train[rndperm[i]]) )
    ax1.matshow(X_train[rndperm[i]])
    ax2 = fig.add_subplot(2,10,i + n_plot + 1)
    ax2.matshow(x_train_pred[rndperm[i]].reshape((28,28)))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
plt.show()


# ## <a id=4.2></a> 4.2 使用 TensorFlow 构建多层自编码器

# 现在我们构建一个更深的多层自编码器，结构为 `784->200->50->10->50->200->784`。

# In[84]:


inputs = layers.Input(shape=(28*28,), name='inputs')
hidden1 = layers.Dense(200, activation='relu', name='hidden1')(inputs)
hidden2 = layers.Dense(50, activation='relu', name='hidden2')(hidden1)
hidden3 = layers.Dense(10, activation='relu', name='hidden3')(hidden2)
hidden4 = layers.Dense(50, activation='relu', name='hidden4')(hidden3)
hidden5 = layers.Dense(200, activation='relu', name='hidden5')(hidden4)
outputs = layers.Dense(28*28, activation='softmax', name='outputs')(hidden5)
deep_auto_encoder = tf.keras.Model(inputs,outputs)
deep_auto_encoder.summary()


# In[85]:


deep_auto_encoder.compile(optimizer='adam',loss='binary_crossentropy') #定义误差和优化方法
get_ipython().run_line_magic('time', 'deep_auto_encoder.fit(x_train, x_train, batch_size=100, epochs=200,verbose=1) #模型训练')


# In[86]:


x_train_pred2 = deep_auto_encoder.predict(x_train)
# Plot the graph
plt.gray()
fig = plt.figure( figsize=(16,3) )
n_plot = 10
for i in range(n_plot):
    ax1 = fig.add_subplot(2,10,i+1, title='Label:' + str(y_train[rndperm[i]]) )
    ax1.matshow(X_train[rndperm[i]])
    ax3 = fig.add_subplot(2,10,i + n_plot + 1 )
    ax3.matshow(x_train_pred2[rndperm[i]].reshape((28,28)))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




