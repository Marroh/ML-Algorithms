# 代码说明
linear_classifier.py包含了Logistic回归，Softmax回归，感知机三个算法。运行程序时将main.py相应代码取消注释运行即可。

# 算法说明
Logistic回归，Softmax回归，感知机每个class都都包含了数据读取、预处理（可选两种类型的归一化）、训练、可视化四个部分。

每一个class的参数说明请见代码中的注释。在可视化部分黑线为分类边界，彩色线为某一类别的判别平面。

在Softmax回归和感知机中由于训练loss震荡剧烈，且增大batch size或减小lr并未有显著改善，均加入了学习率衰减机制，每x步衰减90%。

多分类感知机中，由于采用1 vs 1, 1 vs rest都会产生模糊区域，故采用与Kesler‘s construction相同的决策检验式(模式识别第四版 p.70)。即选取最大输出类别作为分类类别。这种判别条件相比优点在于不存在模糊区域。

#结果展示
![Logistic 2cls](images/logistic.png "logistic回归GD")
![Logistic 3cls](images/Logistic_Newton.png "logistic回归Newton")
![Softmax_2cls](images/Softmax-2cls.png "softmax回归二分类结果")
![Softmax 3cls](images/Softmax_3cls.png "softmax回归三分类结果")
![perceptron_2cls](images/perceptron_2cls.png "感知机二分类结果")
![perceptron 3cls](images/perceptron_3cls.png "多分类感知机三分类结果")

#结果分析
从Logistic回归的结果中可以看到，Newton迭代的速度远远快于GD。

使用SGD会造成loss不稳定，但是增大了随机性，可能跳出局部最优点，二分类softmax中测试机acc的几个尖锐突起可以证明这一点。