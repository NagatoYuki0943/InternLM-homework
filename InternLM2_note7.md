

# 大模型评测

## 评测促进模型发展

![1](InternLM2_note7.assets/1.jpg)

## 大模型评测中的挑战

大模型评测中存在很多挑战：

全面性：

- 大模型应用场景千变万化，要对每个场景都设计对应的评测
- 模型能力演进迅速，模型能提提升迅速，如何有效评测模型能力是急需解决的
- 设计和构造可扩展的能力维度体系，从多维度评测模型能力要设计出可以扩展的评测方式

评测成本：

- 评测数万道题，在本地运行需要大量计算资源，api评测也有很高的费用
- 基于人工打分的主观评测成本高昂，参考 https://chat.lmsys.org/

数据污染：

- 模型训练的海量语料不可避免带来数据污染
- 急需可靠的数据污染检测技术
- 如何设计可动态更新的高质量评测基准

鲁棒性：

- 大模型对提示词十分敏感，如何设计提示词让模型发挥出最大实力是需要设计的
- 对模型进行多次采样，模型性能不稳定

![2](InternLM2_note7.assets/2.jpg)



# OpenCompass

## 开源历程

![3](InternLM2_note7.assets/3.jpg)

## 机构合作

![4](InternLM2_note7.assets/4.jpg)



## 评测方式

![5](InternLM2_note7.assets/5.jpg)

### 客观评测与主观评测

![6](InternLM2_note7.assets/6.jpg)

### 提示工程

![7](InternLM2_note7.assets/7.jpg)



![8](InternLM2_note7.assets/8.jpg)

### 长文本评测

![9](InternLM2_note7.assets/9.jpg)

## 社区

![10](InternLM2_note7.assets/10.jpg)

### CompassRank

![11](InternLM2_note7.assets/11.jpg)

### CompassKit

![12](InternLM2_note7.assets/12.jpg)

#### 评测流水线

![13](InternLM2_note7.assets/13.jpg)

#### 全站工具链

![14](InternLM2_note7.assets/14.jpg)

### CompassHub

![15](InternLM2_note7.assets/15.jpg)

![16](InternLM2_note7.assets/16.jpg)

## 自研高质量大模型评测基准

![17](InternLM2_note7.assets/17.jpg)

### MathBench

![18](InternLM2_note7.assets/18.jpg)

### CIBench

![19](InternLM2_note7.assets/19.jpg)

### T-Eval

![20](InternLM2_note7.assets/20.jpg)

### 生态

![21](InternLM2_note7.assets/21.jpg)