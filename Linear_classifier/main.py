from linear_classifier import *


X, y = load_data('./Exam/train')
X_eval, y_eval = load_data('./Exam/test')
mul_X, mul_y = load_data('./Iris/train')
mul_X_eval, mul_y_eval = load_data('./Iris/test')

# Logistic分类器二分类
# logistic_classifier = LogisticRegression(X, y, X_eval, y_eval, thresh=0.5, lr=1e-5, show=True, show_period=500)
# logistic_classifier.normalization()  # 归一化有normalization和minmax_norm两种选择
# logistic_classifier.train(optimizer='Newton')  # optimizer有'GD'和'Newton'两种选择

# Softmax分类器二分类
# softmax_classifier = SoftmaxRegression(X, y, X_eval, y_eval, batch_size=16, lr=1e-3, show=True, show_period=10, seed=10)
# softmax_classifier.normalization()
# softmax_classifier.train()

# Softmax分类器三分类
# softmax_classifier = SoftmaxRegression(mul_X, mul_y, mul_X_eval, mul_y_eval, batch_size=32, lr=1e-1, c=3, show=True, show_period=5, seed=1)
# softmax_classifier.normalization()
# softmax_classifier.train()

# 二分类感知机
# MP = MultiClsPerceptron(X, y, X_eval, y_eval, batch_size=16, lr=1e-2, c=2, show=True, show_period=10)
# MP.normalization()
# MP.train()

# 多分类感知机
# MP = MultiClsPerceptron(mul_X, mul_y, mul_X_eval, mul_y_eval, batch_size=16, lr=1e-2, c=3, seed=1, show_period=10)
# MP.normalization()
# MP.train()
