# 论文细节补充

## 0. 最终实验设计口径（建议采用）

核心思路不是把所有航段直接按普通二分类随机切分，而是采用“模块化训练 + 统一 hold-out 测试”的设计：

- BiLSTM 模块只负责学习整段异常（segment/whole）的长时模式，因此训练集应由 `whole` 异常航段和有代表性的正常航段组成。
- 规则模块负责识别突变异常（jump），并通过红色渐变事件（red）过滤/豁免来降低 BiLSTM 对真实飞行事件的误报。
- 最终融合方法在统一测试集上评估，测试集同时包含 `whole`、`jump`、`red`、`stable/jolt`，用于证明单一方法不足，而融合方法能互补。

建议在论文中这样表述：

> 本文采用模块化训练与统一测试的实验策略。BiLSTM 模块面向整段异常识别，在整段异常航段与代表性正常航段上训练，以避免海量正常样本淹没整段异常模式；规则模块面向突变异常识别，在 jump/red 难例上进行离线归纳与阈值确定。最终，BiLSTM、规则模块及融合方案均在未参与模型训练和规则确定的 hold-out 测试航段上统一评估，测试集覆盖整段异常、突变异常、红色渐变事件以及正常/颠簸航段。

具体口径建议：

1. **BiLSTM 训练集**
   - 目标：只训练 segment/whole 异常识别能力。
   - 使用：`whole` 训练航段 + 采样的 `stable/jolt/red` 正常航段。
   - 不建议使用全部正常航段，否则正常样本过多，模型容易偏向预测正常。
   - 当前旧实验中训练集异常比例约 40%，这个比例对 BiLSTM 学习整段异常是合理的，但论文中要说明这是模块训练口径，不是真实分布模拟。

2. **BiLSTM 验证集**
   - 目标：选择 epoch、窗口长度、隐藏维度、分类阈值和后处理参数。
   - 使用：`whole` 验证航段 + 一部分 `stable/jolt/red` 正常航段。
   - TODO：当前代码还没有真正的 val loader，需要补上。

3. **规则模块开发/验证集**
   - 目标：确定 jump 规则阈值、渐变 red 过滤逻辑，以及算法 2 中规则演变过程。
   - 使用：`jump` 的 train/val 难例 + `red` 难例。
   - 输出：`R0 -> R1 -> final` 的指标变化表，说明 LLM 只参与离线规则归纳，最终在线规则是确定性的。

4. **最终测试集**
   - 目标：统一评估 `BiLSTM-only`、`rule-only`、`fusion`。
   - 使用：未参与训练/验证的 hold-out 测试航段，包含 `whole + jump + red + stable/jolt`。
   - 重点展示：
     - BiLSTM 能识别整段异常，但可能误判 jump/red。
     - 规则能补充 jump，并过滤/豁免 red。
     - 融合后整体 Precision、Recall、F1 提升。

论文中需要避免的表述：

- 不要笼统写“训练集、验证集、测试集按 6:2:2 随机划分”。
- 不要把 BiLSTM 的采样训练集说成真实业务分布。
- 不要说测试集是“训练集以外所有数据”后又声称有独立 val，需要统一。

后续落地 TODO：

- 改造 `load_whole_data()`：支持返回 `train/val/test`，同时支持 BiLSTM 模块采样训练口径。
- 输出 split 统计表：航段列表、类型、正常点数、异常点数、异常比例。
- 训练脚本增加 val：保存 `history.csv`、loss/F1 曲线、best checkpoint。
- 规则脚本增加 split 批量评估：输出 jump/red 相关指标和规则演变表。
- 挑选典型案例图：`BiLSTM 补 whole`、`规则补 jump`、`red 豁免降低误报`。

## 1. 表3.1中，超参k和e的取值。
特征工程滚动窗口：zt = xt – median(x(t-k):(t+k)) / MAD(x(t–k):(t+k)) + ε 中k=48，e取1e-6（极小）

Answer: 发现实际用的
预处理：直接StandardScaler标准化，不是中位数。
特征：原始VRTG、一阶差分、二阶差分、多步差分、滑动中位数、滑动标准差、原始VRTG与滑动中位数的差值。共7维（NO）

BiLSTM输入上下文窗口：w=100，前后各100个点，共201个。

## 2. 图3.2，复查一下，看着有点奇怪。

Answer:
1）发现这里画的不是基于中位数和MAD的稳健z分数，而是基于均值和标准差的经典z分数。
2）改成真正的稳健z分数后，图3-2（a）：

·在（未画出的）曲线末尾，vrtg将突然进入值为-3的整段异常，因此出现最后一个波形的变形。
·锯齿状可能是因为原始vrtg本身存在微小的锯齿波动（可能是由于传感器），而z分数拉大了这一波动。也就是波动的出现应该是正常的。
3）类似地，更正后的图3-2（b）



## 3. 26页，3.2.3 小节标题的上一段，关于biLSTM那些超参的消融实验结果，只report 只包含正常和整段异常的 测试集上的结果应该就ok？

Answer:
“同时，本文围绕窗口长度、滑动步长与隐藏维度等因素开展消融分析，以兼顾召回率与定位精度，相关设置与结果在第4章统一报告。”
也就是TODO：BiLSTM在整段异常+正常的测试集上的【窗口长度、滑动步长与隐藏维度】调参结果



## 4. 算法1中，三个超参的取值。

Answer:
窗口半径w=50，分类阈值θ_seg = 0.5，合并间隔g_seg = 24，最短段长ℓ_seg = 80

## 5. 算法2中，初始规则是啥？ Prompt的具体例子。

Answer:
初始规则（R₀）
在规则设计初期，本文采用简化的判定方法对突变异常进行检测。具体而言，首先设定宽松的越限区间[L,U]，当观测值xt超出该范围时，认为其处于异常候选状态；同时，引入相邻时刻差分 Δxt=∣xt−xt−1∣以刻画局部变化幅度。当某一时刻同时满足越限条件且差分幅度超过阈值τ时，即判定为突变异常点。该规则仅依赖单点及其邻域的局部变化信息，未对异常的持续结构、进入与退出模式进行刻画，因此使用LLM辅助规则设计。

[Prompt]
You are an expert in time-series anomaly detection, especially in aviation sensor data.
Your task is to improve a simple rule-based algorithm for detecting "jump anomalies" in VRTG (vertical acceleration) signals.

[Current Rule R0]
We define an anomaly point at time t if:
1.Out-of-bound condition:
x_t < L or x_t > U
2.Local jump condition:
Δx_t = |x_t - x_{t-1}| ≥ τ
Parameters:
L = 0.5
U = 1.8
τ = 0.5

[Observed Problems]
This rule produces many false positives and false negatives:
It detects short noise spikes as anomalies
It misclassifies gradual drift as jump anomalies
It sometimes misses real jump segments that last multiple timesteps

[Failure Cases]
False Positives (FP):
FP1 (gradual drift):
x = [..., 1.10, 1.18, 1.25, 1.32, 1.40, ...]
→ slowly increases but detected as jump
FP2 (noise spike):
x = [..., 1.00, 1.90, 1.00, ...]
→ single-point spike

False Negatives (FN):
FN1 (short segment jump):
x = [..., 1.00, 1.60, 1.60, 1.00, ...]
→ real jump but partially missed

[Statistics Summary]
Δx_t distribution:
True anomalies: mean ≈ 0.7
False positives: mean ≈ 0.3–0.4
Segment duration:
True jumps: usually 1–3 points
Gradual drift: usually > 5 points

[Your Task]
Based on the above information, propose improvements to the rule.
You may:
introduce new features or conditions
modify thresholds
combine multiple conditions
Constraints:
keep rules simple and interpretable
avoid complex models
prefer minimal and incremental changes

[Output Format]
1.Key problems of current rule
2.Proposed rule modifications
3.Explanation of why they help
4.(Optional) suggested parameter adjustments




## 6. 4.1.2节中，训练集、验证集、测试集 具体的航段列表，以及正常点数、异常点数、异常比例。

Answer:
训练集	测试集
——用到了训练集以外的所有数据
（整段异常+全部的跳点异常+采样部分纯正常航段）

整段异常	3422308320230907B-7561.csv	vrtg-B-5627-CDG1181-38230597.csv
	3504866520231030B-1808.csv	vrtg-B-6363-CCA1708-38553835.csv
	3505259820231030B-1808.csv	vrtg-B-6378-CSN6497-37521204.csv
	3505460520231030B-1808.csv	vrtg-B-6871-CES2412-38489902.csv
	3567331020231211B-8446.csv	vrtg-B-6871-CES2418-38497530.csv
	3568046720231211B-8446.csv	3310337520230704B-323N.csv
	vrtg-B-1613-CES6670-38752469.csv	3503687720231029B-6829.csv
	vrtg-B-1613-CES6865-38755557.csv	3542350220231124B-6567.csv
正常
（红色）	red-2094235820200803045520.csv	vrtg-B-222P-HLF9804-37630518.csv
	red-3384518120230816B-209F.csv	vrtg-B-326E-CGZ7145-38299047.csv
	1872600020191221B-1701.csv	vrtg-B-326E-CGZ7218-38448832.csv
正常
（颠簸/平稳）	3305440320230628B-6488.csv	vrtg-B-1897-CHB6295-38849440.csv
	4285373120250113023805.csv	vrtg-B-1932-CDG7660-38463083.csv
	4301349720250120220741.csv	vrtg-B-6789-GCR7620-38742564.csv
	4317018120250130045317.csv	vrtg-B-8247-CQH8755-38411708.csv
	4327195020250204222607.csv	4354525520250220125841.csv
	4371054420250302024335.csv	vrtg-B-8948-CSC8170-36660139.csv
	4388076020250312132253.csv	..... (共86个航段).
异常点数量	648,882	226,463
正常点数量	842,894	6,322,362
异常点比例	43%	3.46%


## 7. BiLSTM的额外结果： 训练集、验证集上的结果，只包含正常和整段异常的 测试集上的结果。 训练集、验证集 上的损失 vs epoch的图。







## 8. 规则识别突变的额外结果：训练集、验证集上的结果，只包含突变和整段异常的 测试集上的结果。 算法2中，随着规则演变，m指标的变化过程。






## 9. 红色识别是否准确？就算不完全准，应该也可以找几个准点的例子，说明这个识别在最终的预测中有作用。


***

这个算法能输出  红色 事件，你有没有看过这个准吗？
在想你这个工作的创新性，可以从三个方面来讲，一是用两种策略来处理不同的异常类型。二是基于大模型来演变规则。三是可以区分数据异常和“颠簸&红色事件”（如果你这个red的输出比较准）

***

## 10. 关于第一个创新点（两种方式融合），去找几个例子，比如 方法1（规则）不能识别seg，方法2（bilstm）不能识别跳点，但融合之后就可以。以及对比的那些方法没搞定，但我们搞定的情况。总之就是要说明白为啥单一方法和现有方法不行，但融合方法可以。给点典型的例子来说明。




## 11. 增加几个对比的方法（哪几个方法待确定）。





## 12. 增加对比的方法：OC-SVM，LOF，IF（参见 丁建立 航空学报的文章），丁建立 航空学报的方法，2022 & 2024 两篇IEEE TAES的文章。主要的逻辑是，传统的方法 OC-SVM，LOF，IF 增加一点， 最新的顶刊论文的方法增加一点。
