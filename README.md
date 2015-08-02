#HMM

[hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)

隐马尔科夫模型：

主要有三个问题：

1. 评估问题（概率计算问题） ：给定模型参数和一个观测序列，求得到该观测序列的最大概率值。使用前向算法和后向算法

2. 学习问题：已知观测序列，求模型的参数。使用前向-后向算法，也是Baum-Welch算法。

3. 预测问题（解码问题）：已知模型参数和观测序列，求最有可能的隐藏序列。使用Viterbi算法。