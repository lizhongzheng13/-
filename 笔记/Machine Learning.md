# 				          		Machine Learning

​									date:2024-9-20

<hr/>

## What is machine learning?

> ![image-20240920163727434](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240920163727434.png)
>
> performance measure：性能度量（性能度量是用于评估和衡量一个系统、算法、模型或过程在特定任务或目标方面表现的<font color='red'>**指标**</font>或标准。）
>
> test:	![image-20240920164238945](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240920164238945.png)

​	(B选项，是E；C选项，是P)

> filter spam email：过滤垃圾邮件

<font color='red' size=5>E经验（知识）、T任务（要解决什么）、P性能（指标）</font>



**Machine Learning algorithms:**

​	**-监督学习supervised learning** 

​	**-无监督学习unsupervised learning** 



## Supervised Learning

在机器学习和统计学中，分类问题（Classification）和回归问题（Regression）是两种常见的预测模型类型，它们用于解决不同类型的问题。

### 分类问题（Classification）
分类问题的目标是预测一个离散的标签或类别。在分类问题中，输出变量（目标变量）通常是有限数量的类别之一。分类问题通常用于以下场景：

- **二分类问题**：预测一个二元结果，例如垃圾邮件检测（是垃圾邮件/不是垃圾邮件），疾病诊断（健康/患病）。
- **多分类问题**：预测多个类别中的一个，例如图像识别（识别图片中的物体是猫、狗还是汽车），情感分析（正面、负面或中性）。

常用的分类算法包括逻辑回归、决策树、随机森林、支持向量机（SVM）、神经网络等。

### 回归问题（Regression）
回归问题的目标是预测一个连续的数值。在回归问题中，输出变量（目标变量）是一个实数值，可以是任何连续的数值范围。回归问题通常用于以下场景：

- **预测房价**：根据房屋的特征（如面积、位置、房间数量等）预测房屋的价格。
- **股票价格预测**：根据历史数据预测股票的未来价格。
- **天气预测**：预测未来几天的气温、降水量等。

常用的回归算法包括线性回归、多项式回归、决策树回归、随机森林回归、支持向量回归（SVR）、神经网络等。

### 区别
- **输出类型**：分类问题的输出是离散的类别，而回归问题的输出是连续的数值。
- **评估指标**：分类问题通常使用准确率、精确率、召回率、F1分数等指标来评估模型性能，而回归问题则使用均方误差（MSE）、均方根误差（RMSE）、绝对平均误差（MAE）等指标。
- **应用场景**：分类问题适用于需要将数据划分到预定义类别的场景，而回归问题适用于需要预测连续数值的场景。

选择分类还是回归模型，取决于问题的性质和业务需求。

##### test:

> ![image-20240921110857189](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921110857189.png)

> regression:回归	classification:分类



## Unsupervised Learning

> 无监督学习和监督学习的区别：监督学习在数据标记上可能有多种标签，但是无监督学习没有标签或者都是同一种标签

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921111807889.png" alt="image-20240921111807889" style="zoom: 67%;" />

​		

> 聚类算法：聚类算法是一种无监督学习算法，用于将数据集中的样本根据相似性分组。它试图将数据分成多个簇，使得同一个簇内的样本相似度较高，而不同簇之间的样本相似度较低。常见的聚类算法包括K-means、层次聚类、DBSCAN等。Example: google news

##### test

> ![image-20240921115328954](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921115328954.png)





## Linear regression with one variable (线性回归) 

###  Cost function intuition （代价函数）

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921151544110.png" alt="image-20240921151544110" style="zoom:67%;" />

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921151858530.png" alt="image-20240921151858530" style="zoom:67%;" />

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921152100835.png" alt="image-20240921152100835" style="zoom:67%;" />

> 线性函数的举例:
>
> <img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921152329526.png" alt="image-20240921152329526" style="zoom:67%;" />

> 如何根据坐标上的参数得出，拟合较好的线性函数？

$$
在机器学习中，损失函数（Loss Function）是衡量模型预测值与实际值差异的函数。\\对于线性回归问题，常用的损失函数是均方误差（Mean Squared Error, MSE），其公式如下：\\

\text{MSE} = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2
\\
其中：
- m 是样本数量。\\
- h_{\theta}(x^{(i)}) 是模型对第 i 个样本的预测值。\\
- y^{(i)} 是第 i 个样本的真实值。\\
- \theta 表示模型参数。\\

在文件中提到的“未修正的样本方差”可能是指损失函数中的分母部分 \frac{1}{2m}。\\这里的 \frac{1}{2} 是为了在后续求导时简化计算。\\具体来说，当我们对损失函数求导时，平方项会引入一个额外的2，因此通过在损失函数中除以2，可以抵消求导时产生的2，简化计算过程。\\
$$

------

> 均方误差函数常用于求解回归问题

> 代价函数目前小总结：
>
> ![image-20240921155237525](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921155237525.png)

![image-20240921164144475](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921164144475.png)

> **假设theta0=0的情况下**：(注意：红色的标记是实际值)
>
> ![image-20240921164759024](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921164759024.png)
>
> ![image-20240921165603510](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921165603510.png)
>
> 
>
> 最后在theta1等于不同值的情况下，逐渐描绘出对应的图表：![image-20240921170158770](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921170158770.png)

----



### Gradient descent(梯度下降)

![image-20240921172815151](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921172815151.png)

> (通俗理解：梯度下降算法就是指站在山顶上向哪个方向走下山最快)![image-20240921185240162](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921185240162.png)
>
> 在优化问题中，我们通常将目标函数的最小值比作山的最低点。梯度下降算法的目标就是找到这个最低点。
>
> 1. **梯度**：梯度是一个向量，它指向函数增长最快的方向。在山顶，梯度指向上方；在山脚，梯度指向下方。
> 2. **负梯度方向**：为了下山，我们需要沿着梯度的反方向走，即沿着负梯度方向。这是因为负梯度方向是函数值下降最快的方向。
> 3. **步长**：在梯度下降中，每次迭代都会沿着负梯度方向走一步。步长（学习率）决定了每次迭代的步幅。如果步长太大，可能会越过最低点；如果步长太小，收敛速度会很慢。
> 4. **迭代**：通过不断迭代，逐步逼近最低点。每次迭代都会计算当前位置的梯度，并沿着负梯度方向更新位置。

所以<font color='red'>梯度下降算法可以形象地理解为：站在山顶上，不断朝着下山最快的方向（负梯度方向）走，直到接近山脚（目标函数的最小值）</font>。



![image-20240921190718429](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240921190718429.png)

>alpha指的是下降的步伐，是跨的小步还是跨的大步。(它决定了我们沿着能让代价函数下降程度最大的方向向下迈出的**步子有多大**。)
>
>:=是指赋值运算。
>
>这个算法要同时对theta0和theta1进行赋值。
>
>Correct所代表的是同步更新，常使用。

<p>一般提到梯度下降算法就是指的是同步更新。</p>

![image-20240922102046503](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240922102046503.png)

![image-20240922102710660](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240922102710660.png)

![image-20240922155723110](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240922155723110.png)

> 当我们逐渐接近局部最小值时，梯度下降会自动缩小步伐，所以不需要再让alpha随时间而减小了。



----

### Gradient descent for linear regression(线性回归的梯度下降)

> 将梯度下降法应用到最小化平方差代价函数中。
>
> ![image-20240922161059708](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240922161059708.png)

![image-20240922162054810](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240922162054810.png)

> 注意：这里是求偏微积分，第一个式子是对theta0进行求导，进而将theta1看作常数。
>
> 第二个是对theta1进行求导，theta0进而看作常数。
>
> 1/2m的好处就在这里体现出来了，一求导，平方的2刚好和1/2m中的2相抵消。
>
> ***位置：“白话机器学习”p224***

将操作后的式子带入到梯度下降算法中：![image-20240922163355390](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240922163355390.png)

不停的调整theta0和theta1，使得函数不断的拟合数据标点

![image-20240922163957679](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240922163957679.png)

##### 正在学习的梯度下降也称为："Batch" Gradient Descent

>"Batch"（批量） Gradient Descent : Each step of gradient descent uses all the training examples.（意味着每一步梯度下降，我们都遍历了整个训练集的样本） 

----



### homework1

![image-20240923070817787](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240923070817787.png)

![image-20240923070836764](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240923070836764.png)

> 注意h(x)也可以由theta转置乘x可得！（代码中用）

在这个部分，由题目以及实际运算可得：损失函数为：![image-20240923070915763](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240923070915763.png)

> 其中有两个变量x1，x2。
>
> theta要根据矩阵相乘的要求进行转置为2*1的矩阵

位置：b站（收藏夹）作业讲解第一节课17分钟左右

## ***sum1：***

在线性回归中，梯度下降法是一种常用的参数估计方法，它通过迭代更新参数来最小化损失函数。对于多元线性回归，损失函数通常是均方误差（Mean Squared Error, MSE），其公式为：
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$
其中，\( m \) 是训练样本的数量，
$$
h_\theta(x^{(i)})
$$


 是模型关于输入 \( x^{(i)} \) 的预测值，\( y^{(i)} \) 是实际值，\( \theta \) 是模型参数。

梯度下降法的更新公式为：
$$
\theta := \theta - \alpha \frac{1}{m} X^T(X\theta - y)
$$

这里，\( \alpha \) 是学习率，\( X \) 是特征矩阵，\( y \) 是目标值向量，\( \theta \) 是参数向量。更新公式表明，参数 \( \theta \) 应该沿着梯度的负方向更新，因为这样可以减少损失函数的值。

在每次迭代中，梯度下降法会计算预测值和实际值之间的误差，然后根据这个误差来更新参数。这个过程会一直重复，直到达到一定的迭代次数或者损失函数的值不再显著减少。

需要注意的是，学习率 \( \alpha \) 的选择对梯度下降法的性能有很大影响。如果 \( \alpha \) 太大，可能会导致算法在最小值附近震荡而不是收敛；如果 \( \alpha \) 太小，算法的收敛速度会很慢。此外，特征缩放（标准化或归一化）可以帮助加速梯度下降法的收敛。

![image-20240925105624180](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240925105624180.png)

梯度下降和正规方程之间的比较：![image-20240925105743060](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240925105743060.png)

![image-20240928200905357](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240928200905357.png)

![image-20240928215445057](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240928215445057.png)

----

## ***Matrix***



### 基础知识  :

![image-20240925202543767](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240925202543767.png)

![image-20240925203903367](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240925203903367.png)

![image-20240925204518826](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240925204518826.png)

![image-20240925212319309](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240925212319309.png)

![image-20240925212608931](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240925212608931.png)

![image-20240925212850646](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240925212850646.png)

![image-20240925212833716](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240925212833716.png)

![image-20240925213928206](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240925213928206.png)

#### 小技巧：

向量化求解

![image-20240926110151881](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926110151881.png)

matrix-matrix multiplication

练习：![image-20240926110637172](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926110637172.png)

![image-20240926111315293](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926111315293.png)

多个式子使用矩阵法求解：

![image-20240926111607569](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926111607569.png)

### matrix multiplication properties

矩阵乘法不服从交换律：![image-20240926114207047](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926114207047.png)

 ![image-20240926114553821](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926114553821.png)



### 1.单位矩阵（Identity Matrix）：

也称为恒等矩阵，是一个主对角线上的元素都是1，其余位置的元素都是0的方阵。单位矩阵的行数和列数相等，即它是一个n*×*n*的矩阵，记作*
$$
I 
n
$$


单位矩阵具有以下性质：

1. **乘法单位元**：任何矩阵A*与单位矩阵I*相乘，结果仍然是矩阵A*，即<font color='red'>**A×I=I×A=A**</font>。
2. **逆矩阵**：单位矩阵是它自己的逆矩阵，即I^-1=I
3. **行列式**：单位矩阵的行列式值为1。

单位矩阵的一般形式如下：![image-20240926115305045](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926115305045.png)

其中，矩阵的第i*行第i*列的元素是1，其余元素都是0。

**注意大多数的矩阵A*B!=B*A，但是单位矩阵可以！！！**

![image-20240926215336917](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926215336917.png)

> 矩阵*逆矩阵=单位矩阵

### 2.矩阵的逆运算

矩阵的逆运算是指找到一个矩阵，使得当它与原矩阵相乘时，结果为单位矩阵。如果一个矩阵 A存在这样的逆矩阵，我们称 A是可逆的，或者称为非奇异矩阵。逆矩阵通常表示为 A^-1。

#### 矩阵逆的条件

不是所有矩阵都有逆矩阵，一个矩阵 A 有逆矩阵的必要且充分条件是它必须是方阵（即行数和列数相等），并且它的行列式（determinant）不为零。

#### 计算矩阵的逆

计算一个矩阵的逆可以通过以下步骤：

1. **计算行列式**：首先计算矩阵 A 的行列式 det(*A*)。如果det(*A*)=0，则矩阵不可逆。
2. **计算伴随矩阵**：计算矩阵A的伴随矩阵（也称为伴随矩阵或伴随矩阵）。伴随矩阵是通过计算矩阵的余子式并转置得到的。
3. **计算逆矩阵**：如果矩阵可逆，那么它的逆矩阵可以通过以下公式计算： 

![image-20240926214737696](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926214737696.png)

​             其中adj(*A*) 是 A 的伴随矩阵。

![image-20240926210347814](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926210347814.png)

E.g.假设有一个 2×2 矩阵：

![image-20240926210200512](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926210200512.png)

##### 步骤 1: 计算行列式

首先，我们计算矩阵 A的行列式 det(*A*)： det(*A*)=(3)(4)−(2)(1)=12−2=10

##### 步骤 2: 计算伴随矩阵

接下来，我们计算 A* 的伴随矩阵。对于 2×2 矩阵，伴随矩阵是原矩阵的余子式矩阵的转置：

![image-20240926210246716](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926210246716.png)

​					**伴随矩阵：主对角线交换，副对角线取反**

#### 步骤 3: 计算逆矩阵

最后，我们使用行列式和伴随矩阵来计算 *A* 的逆矩阵

![image-20240926210310698](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926210310698.png)

![image-20240926210331252](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926210331252.png)

> ![image-20240926215657740](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926215657740.png)
>
> 例如：零矩阵



> 矩阵的转置：
>
> ![image-20240926215808037](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240926215808037.png)

----

### Multiple feature

![image-20240927065727927](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240927065727927.png)

> 多个变量
>
> x1,x2,x3,x4分别为4个特征值

----



![image-20240927070057938](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240927070057938.png)

> 多元线性回归

![image-20240927070439736](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240927070439736.png)

![image-20240927070624267](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240927070624267.png)



> 减小代价函数的偏移，可以使用特征值缩放
>
> ![image-20240927071324534](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240927071324534.png)

![image-20240928170744187](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240928170744187.png)

> 均值归一化
>
> ![image-20240928171043318](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240928171043318.png)

> 如何选择学习率alpha
>
> ![image-20240928171521977](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240928171521977.png)
>
> ![image-20240928172011339](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240928172011339.png)
>
> ![image-20240928172208844](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240928172208844.png)
>
> finally:
>
> ![image-20240928172238892](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240928172238892.png)

##### 多项式回归：

![image-20240930171718098](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240930171718098.png)

### normal equation（正规方程）

![image-20240930172033894](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240930172033894.png)

![image-20241002105054779](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002105054779.png)

![image-20241002105425109](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002105425109.png)



![image-20241002111347361](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002111347361.png)

公式：![image-20241002111636936](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002111636936.png)

优缺点：![image-20241002111823000](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002111823000.png)



------

##### ***sum:***

在机器学习中，正规方程（Normal Equation）是一种用于线性回归的解析解法，它提供了一种直接计算最优参数的方法，而不需要使用梯度下降法的迭代过程。正规方程的基本思想是通过对代价函数求导，并将导数置为零来求解参数向量 θ。

正规方程的公式如下：

![image-20241002110338500](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002110338500.png)

使用正规方程求解线性回归问题的优点包括：

1. 不需要选择学习率（如梯度下降法中的 α*）。
2. 不需要迭代，可以直接计算出最优解。

然而，正规方程也有一些缺点：

1. 当特征数量 *n* 很大时，计算 (X^T)^(-1) 的代价非常高，因为矩阵求逆的时间复杂度是 O*(*n^3)。
2. <u>*正规方程不适用于大规模数据集*</u>，因为矩阵的存储和计算成本随着数据量的增加而迅速增长。

在实际应用中，如果特征数量不是特别大（通常认为小于10000个特征），正规方程是一个有效的计算参数的方法。但是，当特征数量很大时，梯度下降法或其他优化算法可能是更好的选择，因为它们在计算上更加高效。

此外，如果设计矩阵 X* 不是满秩的，即 X^T*X 不可逆，那么正规方程就不能直接应用。这种情况可能发生在特征之间存在线性依赖关系，或者特征数量大于样本数量时。在这种情况下，可能需要通过添加正则化项（如L2正则化）来解决矩阵不可逆的问题，或者使用奇异值分解（SVD）等方法来求解伪逆。

总结来说，正规方程是一种在特征数量不是特别大的情况下，快速直接求解线性回归参数的方法。但是，当特征数量增加或数据集规模变大时，需要考虑其他更高效的算法。

------

#### normal equation and non-invertibility (optional)

`non-invertibility:不可逆`

不可逆：![image-20241002171302845](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002171302845.png)

> 可以检查下特征值是否有多余，有的话删除多余的；若是特征值太多，可以删除一些不影响的或则使用正规化方式

![image-20241005210127770](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005210127770.png)

## classification

### sigmoid函数（logistic回归）

![image-20241002211531137](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002211531137.png)

![image-20241002212033682](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002212033682.png)

#### 决策边界（decision boundary）

![image-20241002212321354](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002212321354.png)

> 0 < g(z) < 1

example：![image-20241002230756692](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002230756692.png)

example：![image-20241002231144100](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002231144100.png)

#### 如何拟合logistic回归模型的参数theta：

> 用来拟合参数的优化目标或叫代价函数

![image-20241002231431416](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241002231431416.png)

![image-20241004172955734](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004172955734.png)



> 预测y=1，预测的准的话cost代价函数就小；要是不准的话，cost代价函数就比较大
>
> ![image-20241004174736270](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004174736270.png)
>
> ![image-20241004175030441](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004175030441.png)



#### simplified cost function and gradient descent

函数：

![image-20241004175255755](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004175255755.png)

![image-20241004183456929](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004183456929.png)

> 其中的cost代价函数将y=1和y=0的情况涵盖在一起了，形成的一个式子

![image-20241004222855631](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004222855631.png)



#### advanced optimization

![image-20241004224205207](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004224205207.png)

![image-20241004224320760](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004224320760.png)

> example：![image-20241004225103561](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004225103561.png)
>
> ![image-20241004225202281](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004225202281.png)



<hr>

#### one - versus - all classification

![image-20241004225629372](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241004225629372.png)

![image-20241005103556589](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005103556589.png)

![image-20241005103714042](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005103714042.png)

> - 我们为每个类别训练了一个分类器（使用所有特征）。
> - 在预测时，我们计算新样本属于每个类别的概率，并选择概率最高的类别作为预测结果。



#### the problem of overfitting(过度拟合)

> 存在多个特征，但是数据很少，或者模型函数不合理，都会出现过拟合的现象。过拟合可能对样本数能够很好的解释，但是无法正确的预测新数据。
>
> ![image-20241005105143164](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005105143164.png)

过拟合举例：

![image-20241005105048934](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005105048934.png)

![image-20241005105543005](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005105543005.png)



![image-20241005105907375](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005105907375.png)





#### Regularization && cost function

![image-20241005204056415](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005204056415.png)

![image-20241005204327544](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005204327544.png)

> 在代价函数中加入正则项，通过lambda的来平衡拟合程度和参数的大小，θ约大越容易出现过拟合的现象。
>
> lambda是正则化参数



![image-20241005204507539](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005204507539.png)

> 如果lambda过大，导致θ≈0，那么最终只剩下下theta0，图像将变成一个直线。

![image-20241005205236770](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005205236770.png)



> 补：拟合回归模型：梯度下降，正规方程

#### normal equation

![image-20241005205937042](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005205937042.png)



#### regularized logistic regression

优化梯度下降：

![image-20241005210647708](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005210647708.png)

![image-20241005210811064](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005210811064.png)





<hr>

### **sum2：**

![image-20241005231310498](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005231310498.png)

![image-20241005231609777](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005231609777.png)

![image-20241005231920060](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005231920060.png)

![image-20241005231932956](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005231932956.png)

![image-20241005231942551](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241005231942551.png)

> 注意：这里使用的theta是一开始就转置完的，所有向sigmoid函数中传入的是X*theta

> 注意：*表示点乘！！！其余的使用@表示矩阵相乘

![image-20241006164023293](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241006164023293.png)

<hr>

**线性不可分**

![image-20241006183837179](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241006183837179.png)

```markdown
在线性不可分问题中，我们面临的挑战是数据在原始特征空间中不能用一条直线（二维），一个平面（三维），或一个超平面（多维）分开。为了解决这个问题，我们可以使用一种称为特征映射的技术，将数据映射到更高维的空间中，在这个新的空间里，数据可能是线性可分的。

**特征映射**是指通过数学变换将原始数据从低维空间映射到高维空间的过程。这样做的目的是为了使数据在新的空间中更容易被处理和分类。例如，在二维平面上无法用直线分开的两组数据，可能在三维空间中可以用一个平面分开。

进行特征映射的原因通常包括：
1. **解决非线性问题**：增加维度可以帮助模型捕捉数据中的复杂关系。
2. **提高模型的分类能力**：在高维空间中可能更容易找到区分不同类别的决策边界。
3. **使数据线性可分**：在原始空间中线性不可分的数据在新空间中可能变得线性可分。

核函数是实现特征映射的关键工具之一，它允许我们在高维空间中计算点积，而无需实际地将数据映射到那个空间。这大大减少了计算的复杂性。常见的核函数包括：
- **线性核**：适用于数据线性可分的情况。
- **多项式核**：通过构造一个多项式特征空间来处理更复杂的数据关系。
- **高斯核（RBF）**：能够处理非常复杂的数据结构，并且可以将数据映射到无限维空间。
- **Sigmoid核**：类似于神经网络中的激活函数，可以捕捉数据的复杂非线性关系。

使用核函数的好处是，我们不需要显式地定义映射函数，也不需要在高维空间中直接操作数据，从而避免了高维空间中的“维度灾难”。

在实践中，支持向量机（SVM）是一个常用的分类器，它通过核函数来处理线性不可分的数据。SVM通过最大化两个类别之间的间隔来寻找最佳的决策边界。当数据集不能用线性方式分开时，SVM可以使用核函数将数据映射到高维空间，在这个空间中寻找最佳的线性分割平面。

总的来说，特征映射和核函数是处理线性不可分问题的重要工具，它们使得我们能够通过增加数据的维度来简化分类问题，同时避免了在高维空间中直接进行计算的高昂成本。 

```

![image-20241006201932477](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241006201932477.png)

![image-20241006203207459](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241006203207459.png)

![image-20241006205951887](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241006205951887.png)

正则化的梯度下降：

![image-20241006214237810](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241006214237810.png)





## neurons and the brain (非线性假设)

### 基础：

![image-20241007094945193](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007094945193.png)

![image-20241007095118837](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007095118837.png)

![image-20241007101159955](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007101159955.png)

> 大脑中的神经元结构

> theta可能会被称为weight（权重）

![image-20241007163839145](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007163839145.png)

> 机器学习中的神经网络一般包括三部分，输入层，隐藏层，输出层。
>
> 输入层是输入特征值。





![image-20241007170105120](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007170105120.png)

> 在神经网络中，“低层次有偏置单元”通常指的是在输入层或隐藏层中包含偏置单元（bias unit）。偏置单元是一种特殊的神经元，它的值固定为1，主要用于在计算过程中添加一个常数偏移，以帮助网络捕捉数据中的模式。

> theta1为什么是3*4的矩阵？x0是偏置特征，x0到x3分别映射到a1到a3上，相当于有四个a1到a3	，即三行四列

> theta上标表示第i层的特征矩阵，下标表示在这个矩阵中的位置。(下标是对应关系, 上标是第几层)
>
> 例如：theta1表示的就是从输入层到隐藏层的映射的参数矩阵



向前传播：

![image-20241007172536066](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007172536066.png)

![image-20241007173838083](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007173838083.png)

![image-20241007173945028](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007173945028.png)

> 可以有多个隐藏层



### Examples and intuitions

![image-20241007202433941](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007202433941.png)

> `xor`异或：不同1，相同0；`xnor`同或：相同1，不同0

![image-20241007202959151](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007202959151.png)

![image-20241007203123251](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241007203123251.png)

![image-20241008145418507](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008145418507.png)



![image-20241008150333294](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008150333294.png)

> XNOR（同或）是逻辑运算的一种，它结合了XOR（异或）和NOT（非）两种操作。在XNOR运算中，当两个输入位完全相同（即同时为0或同时为1）时，输出为1；如果输入位不同，则输出为0。



### 多元分类

![image-20241008151553539](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008151553539.png)

> 通过构建神经网络，每种输出就对应一个分类器。

![image-20241008151909189](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008151909189.png)

### cost function



![image-20241008152656985](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008152656985.png)

> 注意：s_l不包含第l层的偏差单元
>
> 二元分类中注意：最后s_l为1，因为最后只有一个输出（0 or 1）

![image-20241008153140075](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008153140075.png)

> 一共是m个样本，每个样本有k个输出，所以是这种二重累加；至于正则项，一共是L层，所以共有L-1个系数矩阵，每个矩阵行数是sl+1，列数是sl，要保证每个矩阵中的每个元素都尽量小，所以是三重求和



### backpropagation algorithm---反向传播算法（***BP***）



#### [什么是bp算法](https://www.bilibili.com/video/BV19K4y1L7ao/?spm_id_from=333.880.my_history.page.click&vd_source=3c58a56884ef40ab5aa99f8a00685d85)



```markdown
BP算法，即反向传播（Backpropagation）算法，是一种在多层神经网络中训练权重的监督学习算法。它通过计算损失函数相对于网络参数的梯度来更新网络的权重，以达到最小化损失函数的目的。
```

BP算法通常包括以下几个步骤：

1. **前向传播（Forward Pass）**：
   - 输入数据通过网络的每一层进行传播，每一层都会对输入数据进行一定的变换（例如，通过加权求和后应用激活函数）。
   - 这个过程一直持续到输出层，产生预测结果。

2. **计算损失（Loss Calculation）**：
   - 计算预测结果与真实标签之间的差异，这个差异通过损失函数来量化。
   - 常见的损失函数包括均方误差（MSE）和交叉熵损失（Cross-Entropy Loss）。

3. **反向传播（Backward Pass）**：
   - 从输出层开始，计算损失函数相对于网络参数（权重和偏置）的梯度。
   - 使用链式法则（Chain Rule）来递归地计算每一层的梯度。

4. **参数更新（Parameter Update）**：
   
   - 利用计算得到的梯度和学习率（Learning Rate）来更新网络的权重和偏置。
   - 权重更新的公式通常是：

   $$
   W = W - \alpha \cdot \frac{\partial L}{\partial W}
   $$
   
   ​	
   $$
   其中  W  是权重，alpha  是学习率，\frac{\partial L}{\partial W}是损失函数相对于权重的梯度。
   $$
   
   
   
5. **迭代优化（Iteration）**：
   
   - 重复上述步骤，直到满足某个终止条件，如达到最大迭代次数、损失函数下降到某个阈值以下或梯度变化非常小。

BP算法是深度学习中非常关键的技术，它使得多层神经网络的训练成为可能。通过不断地调整网络参数，BP算法能够使得神经网络学习到复杂的函数映射，从而在各种任务上取得优异的性能。



![image-20241008192523968](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008192523968.png)

![image-20241008192935472](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008192935472.png)

![image-20241009144536763](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241009144536763.png)

![image-20241009145025211](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241009145025211.png)

![image-20241009150005213](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241009150005213.png)

![image-20241009150345002](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241009150345002.png)

> 可能有问题(代价函数可能缺少负号)

![image-20241009151203950](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241009151203950.png)

> 注意：delta的值只关于隐藏单元，并不包括偏执单元





<hr/>

#### bp算法（李宏毅老师）：

![image-20241008201815118](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008201815118.png)

##### 链式法则：

![image-20241008202051362](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008202051362.png)



![image-20241008202412419](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008202412419.png)



![image-20241008202845798](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008202845798.png)

> b表示偏置项（bias）——它的作用类似于在数学方程中的常数项，用于调整神经元的激活函数的输出。

![image-20241008203013713](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008203013713.png)

![image-20241008203216631](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008203216631.png)

![image-20241008203612878](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008203612878.png)

   ![image-20241008205247941](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008205247941.png)

![image-20241008210942046](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008210942046.png)

![image-20241008211150562](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008211150562.png)

![image-20241008211344150](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008211344150.png)

![image-20241008211613675](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008211613675.png)

![image-20241008211739923](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008211739923.png)

##### summary

![image-20241008211832239](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241008211832239.png)





### Gradient checking——梯度检测



在实现反向传播算法时，如何确保梯度计算正确呢？

在数学上可以使用拉格朗日中值定理来近似的表示曲线上某一点的导数，梯度检测正是使用的这种思想。

![image-20241010211124553](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241010211124553.png)

![image-20241010211518355](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241010211518355.png)

![image-20241010211643757](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241010211643757.png)

### random initialization

在对神经网络进行训练时，theta的取值要随机取值，如果都赋值为0，就会使得每一层的输出值、误差相同，从而存在大量冗余。

![image-20241010212027906](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241010212027906.png)

在进行梯度下降或者其他一些优化之前，需要给θ向量赋初始值。

可以给初始θ设置为0，这在逻辑回归中是允许的。但是这在实际神经网络训练中是不被允许的。下面的这个例子中，如果初始的权重设置为0，那么第二层的输出也是相等的。

![image-20241010212841037](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241010212841037.png)

![image-20241010231322624](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241010231322624.png)

### summary---putting it together

![image-20241010232819283](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241010232819283.png)

<p>输入单元的数量由特向向量的维度确定，输出单元的数量由分类数量确定。样本的标签值也要向量化，例如某个10分类的神经网络，假设某个样本标签值是5，那么y就要写成向量化的形式y = [0 0 0 0 1 0 0 0 0 0]。<hr/>
对于隐藏层单元数量，如果隐藏层数量超过一层，那么每层单元数量相同。隐藏层单元越多越好，但是太多也会导致计算量较大，因此需要在两者之间均衡。
<hr/>
另外隐藏层每层的单元数量也可以和特征数相等或者是特征数的两倍、三倍，都是可以的。
</p>

<p>
    训练神经网络的步骤：
![image-20241010233232632](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241010233232632.png)

![image-20241010233424542](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241010233424542.png)

1.随机初始化权重---设置权重的初始化非常小，接近零
2.前向传播计算最终输出值，对所有样本
3.计算代价函数
4.反向传播计算代价函数的偏导数





<p>
    使用for循环对每一个样本执行前向和反向传播：


当然也有向量化的方法，不过较为复杂。

当然上面的四个步骤还不够：

使用梯度检测和前面的偏导数进行比较，两者近似相等，说明反向传播是正确的，然后将梯度检测禁用
然后使用梯度下降或者共轭梯度等算法来最小化代价函数。当然有可能收敛到局部最优，而非全局最优。



![image-20241010234024763](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20241010234024763.png)

> 图片展示了，试图找到某个最优的参数值使得神经网络的输出值与训练集中观测到的y^(i)的实际值尽可能的接近

> 
>
> 梯度下降的原理：我们从某个随机的初始点开始，它将会不停的往下下降，那么反向传播算法的目的就是算出梯度下降的方向，而梯度下降的作用就是沿着沿着这个方向一点点下降，一直到我们希望得到的点。



## 模型评估

