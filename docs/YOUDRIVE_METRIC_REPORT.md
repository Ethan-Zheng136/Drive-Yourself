# YouDrive Metric Report

## 1. 目标

YouDrive 的目标不是只评估“开得对不对”，而是评估“是否按照目标个性 / 风格在开”，同时不能牺牲安全和基本任务完成能力。

现有指标里：
- `PDM` 更偏向安全与任务完成。
- `SM-PDMS` 尝试把 style 融入 PDM，但仍然偏向预设的 A/N/C 语义。
- 这两类指标都不够直接回答：模型是否真的表现出不同的驾驶人格、是否支持更细的个性化风格。

因此，当前我们采用两层结构：
- 第一层：`YDPS`，作为“安全门控 + 任务门控 + 风格对齐”的总指标。
- 第二层：`fine-grained style taxonomy`，把模型行为拆成多个可解释的风格原型，而不是一个单一总分。

---

## 2. 数据与实验设置

本次分析基于 `NAVSIM mini` 的完整结果。

### 2.1 覆盖范围
- Split: `NAVSIM mini`
- Scenario 数量: `396`
- 模型数量: `8`
- 总有效记录数: `3168`

### 2.2 已评测模型
- `human`
- `diffusiondrive`
- `transfuser_seed0`
- `transfuser_seed1`
- `transfuser_seed2`
- `ltf_seed0`
- `ego_status_mlp`
- `constant_velocity`

### 2.3 完成的结果文件
- `exp/youdrive_sipa_v2_human_navmini/2026.07.01.14.20.12/2026.07.01.14.22.18.csv`
- `exp/youdrive_sipa_v2_diffusiondrive_navmini/2026.07.01.14.56.04/2026.07.01.15.05.38.csv`
- `exp/youdrive_sipa_v2_transfuser_seed0_navmini/2026.07.01.14.28.46/2026.07.01.14.41.22.csv`
- `exp/youdrive_sipa_v2_transfuser_seed1_navmini/2026.07.01.14.37.01/2026.07.01.14.42.44.csv`
- `exp/youdrive_sipa_v2_transfuser_seed2_navmini/2026.07.01.14.42.49/2026.07.01.14.46.19.csv`
- `exp/youdrive_sipa_v2_ltf_seed0_navmini/2026.07.01.14.56.04/2026.07.01.15.05.25.csv`
- `exp/youdrive_sipa_v2_ego_status_mlp_navmini/2026.07.01.14.26.26/2026.07.01.14.28.38.csv`
- `exp/youdrive_sipa_v2_constant_velocity_navmini/2026.07.01.14.24.24/2026.07.01.14.26.22.csv`

---

## 3. 为什么需要新 metric

### 3.1 PDM 的问题
PDM 主要回答：
- 是否碰撞？
- 是否离路？
- 是否保持 TTC 在阈值内？
- 是否完成路线进展？

这些都重要，但它们主要评价的是“安全与任务完成”，而不是“风格人格”。

### 3.2 SM-PDMS 的问题
SM-PDMS 比 PDM 更进一步，试图让 style-aware，但它仍然有结构性局限：
- 风格维度仍然偏 A/N/C。
- 安全容忍度与风格容易混在一起。
- 容易把“更接近风险边界”误认为“更 aggressive”。
- 没有直接评估模型对 persona / task vector 的可控性。

### 3.3 YouDrive 需要的是什么
YouDrive 需要同时满足：
- 安全不能被风格补偿。
- 任务失败不能被“风格像”掩盖。
- 风格应该是连续、多维、可解释的。
- 风格不能退化成“低速”“停着不动”“硬躲所有交互”。
- 风格应该能区分更多细粒度行为，例如：
  - 速度与推进方式
  - 舒适与平顺
  - 稳定与一致
  - 谨慎与安全裕度
  - 礼让与社交互动
  - 压迫他车的倾向
  - 是否冻结 / 爬行 / 过度保守

---

## 4. 指标总结构

我们把 metric 分成两层。

### 4.1 第一层：YDPS

YDPS 是一个总分，但它不是单独的 style score，而是：

```text
YDPS(z)=G_safe * G_task * S_task * S_nonfreeze * (0.75 + 0.25 S_social) * (0.60 A_weighted(z) + 0.40 A_maha(z))
```

#### 各项含义
- `G_safe`: 硬安全门控，安全失败直接清零。
- `G_task`: 任务门控，防止低进展、停滞、爬行刷分。
- `S_task`: 任务能力的软分数。
- `S_nonfreeze`: 防止 conservative 被误解成“停住不动”。
- `S_social`: 社交裕度和低压迫分数。
- `A_weighted(z)`, `A_maha(z)`: 对目标 persona 的风格对齐分数。

#### 这一步的意义
YDPS 解决的是“总评价”问题，但它不是最后答案。
它只是一个带强约束的上层摘要分数。

### 4.2 第二层：Fine-Grained Style Taxonomy

真正回答“能不能区分风格”的，是后面的细粒度风格 taxonomy。

我们把每个 scenario 变成一个 10 维风格向量：

```text
x = [
  pace,
  progress_persistence,
  assertiveness,
  comfort_smoothness,
  stability_consistency,
  caution_margin,
  courtesy_social,
  pressure_avoidance,
  human_likeness,
  anti_freeze
]
```

然后把这些向量再聚类成多个风格原型，从而得到：
- 模型级风格分布
- scenario 级风格归属
- 可解释的 style prototype

这一步才是真正让 benchmark 从“一个分数”变成“多种风格画像”的关键。

---

## 5. 风格向量是如何计算的

下面逐个解释 `x` 里的 10 个维度。

### 5.1 先做场景归一化

我们不是直接拿原始数值，而是先做 scenario-conditioned robust normalization。

对特征 `f`，在同一个 scenario `s` 下：

```text
z_f = clip((f - median_s(f)) / (IQR_s(f) + eps), -5, 5)
```

这里的目的：
- 同一个速度在不同场景里含义不同。
- 同一个 TTC 在高速、路口、拥堵、环岛里的语义不同。
- 所以必须做场景归一化，避免场景 bias 直接污染风格判断。

如果没有 scene z-score，就退化成全局 robust scaling。

---

### 5.2 `pace`

```text
pace_raw =
  0.34 * z(mean_speed)
+ 0.24 * z(max_speed)
+ 0.22 * z(path_length)
+ 0.20 * z(progress_per_second)
```

#### 含义
表示“整体推进是否积极、速度是否偏高、路线推进是否更快”。

#### 为什么这样算
- `mean_speed` / `max_speed` 反映运动速度。
- `path_length` 反映整体行驶量。
- `progress_per_second` 反映单位时间内的路线推进效率。

#### 为什么不是单纯速度
因为单纯速度会被场景误导，也会把“危险快”和“合理快”混在一起。

---

### 5.3 `progress_persistence`

```text
progress_persistence_raw =
  0.32 * z(progress_per_second)
+ 0.22 * z(moving_ratio)
+ 0.18 * z(displacement)
+ 0.16 * z(directness)
+ 0.12 * z(low_speed_progress_balance)
- 0.20 * z(sustained_stop_s)
- 0.15 * z(crawl_ratio)
```

#### 含义
表示“是不是持续推进任务，而不是停滞或爬行”。

#### 为什么要有它
因为 conservative 不能被误解成 static。

这就是你特别强调的点：
- 可以慢一点
- 可以谨慎一点
- 但不能一直停

所以这里显式惩罚：
- 持续停车时间
- crawling 比例

---

### 5.4 `assertiveness`

```text
assertiveness_raw =
  0.24 * z(progress_per_second)
+ 0.20 * z(mean_speed)
+ 0.14 * z(max_acceleration)
+ 0.12 * z(path_length)
+ 0.10 * z(moving_ratio)
- 0.10 * z(sustained_low_speed_s)
- 0.10 * z(pressure_event_rate)
- 0.10 * z(close_object_ratio)
```

#### 含义
表示“是否果断推进任务，但不是通过压迫别人来实现”。

#### 关键点
这里的 assertive 不是 aggressive 的粗暴版本。
它强调：
- 有推进
- 有决断
- 但不能压迫别的交通参与者

所以它显式减掉：
- `pressure_event_rate`
- `close_object_ratio`

---

### 5.5 `comfort_smoothness`

```text
comfort_smoothness_raw = -(
  0.27 * z(max_jerk)
+ 0.20 * z(mean_jerk)
+ 0.18 * z(max_acceleration)
+ 0.14 * abs(z(mean_yaw_rate))
+ 0.12 * z(speed_variance)
+ 0.09 * z(hard_brake_count)
)
```

#### 含义
表示“开起来是否平顺、是否舒适”。

#### 为什么这样算
舒适性通常由：
- jerk
- 急加减速
- 方向盘 / yaw 变化
- 速度波动
- 硬刹车次数

决定。

这是最典型的 smoothness dimension。

---

### 5.6 `stability_consistency`

```text
stability_consistency_raw = -(
  0.26 * z(heading_variance)
+ 0.22 * z(speed_variance)
+ 0.18 * abs(z(max_yaw_rate))
+ 0.16 * z(max_jerk)
+ 0.10 * z(trajectory_shift_l2)
+ 0.08 * z(hard_brake_count)
)
```

#### 含义
表示“策略是否稳定、一致、没有明显抖动”。

#### 与 comfort 的区别
- `comfort_smoothness` 更偏乘坐体验。
- `stability_consistency` 更偏 policy-level consistency。

比如：
- 一个策略可以不太舒适，但很一致；
- 也可以局部舒适，但整体抖动大。

---

### 5.7 `caution_margin`

```text
caution_margin_raw =
  0.20 * z(min_object_distance)
+ 0.18 * z(mean_nearest_object_distance)
+ 0.18 * z(min_front_gap)
+ 0.16 * z(min_headway_s)
+ 0.14 * z(min_relative_ttc_s)
+ 0.08 * safety_margin
- 0.18 * z(close_object_ratio)
- 0.12 * z(predicted_overlap_count)
```

#### 含义
表示“有没有留足安全和交互裕度”。

#### 为什么这是谨慎的核心
你提到 conservative 不应该只是低速。
所以这里不用速度定义保守，而是用：
- 距离
- headway
- TTC
- overlap

这样一个模型即使不慢，也可以是谨慎的；
一个模型即使慢，如果没有裕度，仍然不是好的 conservative。

---

### 5.8 `courtesy_social`

```text
courtesy_social_raw =
  0.24 * z(social_margin_proxy)
+ 0.22 * z(courtesy_pressure_proxy)
+ 0.16 * z(mean_front_gap)
+ 0.12 * z(mean_headway_s)
+ 0.10 * social_score
- 0.20 * z(pressure_event_rate)
- 0.14 * z(predicted_overlap_count)
- 0.08 * z(close_object_ratio)
```

#### 含义
表示“是否会压迫别人，是否在社交交互上更有礼让性”。

#### 这是在回答什么
你明确说过：
- 不能只是速度低
- 要看是否压迫其他交通参与者
- 要看与他车的距离
- 要看路口 hesitation

这个维度就是把“社交/礼让”单独拆出来。

---

### 5.9 `pressure_avoidance`

```text
pressure_avoidance_raw = -(
  0.32 * z(pressure_event_rate)
+ 0.24 * z(close_object_ratio)
+ 0.20 * z(predicted_overlap_count)
+ 0.14 * z(object_density)
- 0.10 * z(social_margin_proxy)
)
```

#### 含义
表示“是否避免产生压力式交互”。

#### 与 `courtesy_social` 的关系
- `courtesy_social` 更像是“正向礼让感”。
- `pressure_avoidance` 更像是“负向冲突避免”。

这两个维度一起，才能更好地描述社交风格。

---

### 5.10 `human_likeness`

```text
human_likeness_raw =
  0.34 * human_likeness_proxy
+ 0.24 * score
+ 0.16 * z(directness)
+ 0.12 * z(low_speed_progress_balance)
- 0.14 * z(trajectory_shift_l2)
```

#### 含义
表示“行为是否接近人类可接受的轨迹模式”。

#### 注意
它不是单纯的 `L2 to human trajectory`。
因为单纯 L2 会过度偏向模仿表面轨迹，而忽略：
- 安全
- 任务完成
- 交互风格

所以它只是一个辅助维度，不是主导维度。

---

### 5.11 `anti_freeze`

```text
anti_freeze_raw =
  0.30 * non_freeze
+ 0.20 * low_speed_progress_balance
+ 0.18 * task_score
+ 0.14 * moving_ratio
- 0.22 * stop_ratio
- 0.18 * crawl_ratio
- 0.16 * sustained_stop_s
```

#### 含义
表示“这个模型是不是在真正推进任务，而不是停着、爬着、冻着”。

#### 为什么特别重要
因为很多所谓“保守”其实是退化：
- 不动
- 超低速爬行
- 长时间停滞

这不应该被算成好风格。

因此 `anti_freeze` 是为了把“谨慎驾驶”与“失去行动能力”区分开。

---

## 6. 为什么这些维度再映射到 `[0,1]`

在算完 raw 分数后，我们又做了：

```text
style_dim = robust01(raw_dim)
```

其中 `robust01` 使用 5% 和 95% 分位数做归一化，再裁剪到 `[0,1]`。

原因：
- 方便不同维度比较。
- 方便做 persona alignment。
- 方便聚类。
- 减少极端值影响。

---

## 7. 为什么这些权重这样定

这些权重当前不是训练得到的，而是 heuristic 设计的，依据有三点：

### 7.1 语义一致性
每个维度应当尽量对应单一行为语义。

例如：
- comfort 应该主要看 jerk / acceleration / yaw
- caution 应该主要看 gap / headway / TTC
- courtesy 应该主要看 social pressure

### 7.2 防止 metric gaming
必须防止：
- 高速刷分
- 停着刷保守
- 过度小心导致任务失败

所以我们加入了：
- `progress_persistence`
- `anti_freeze`
- `pressure_avoidance`
- `courtesy_social`

### 7.3 保持人格可解释
你要给别人讲解，就必须能说清每一维是什么意思。
这也是我没有直接用黑盒 embedding 的原因。

---

## 8. proxy 是什么

proxy 就是“代理指标”。

意思是：你真正想测的东西没有直接字段，就先用一个可计算、相关性较强的替代量近似它。

例如：
- 你真正想测的是“是否压迫其他参与者”
- 但数据里没有直接的 `forced_other_agent_brake`
- 那么就用：
  - `pressure_event_rate`
  - `close_object_ratio`
  - `predicted_overlap_count`
  - `min_headway_s`
  - `min_relative_ttc_s`

来近似它。

因此：
- `pressure_event_rate` 不是“压迫”的真值
- 它只是“压迫风险”的 proxy

这很重要，因为论文里要诚实说明：
- proxy 有用
- 但它不是 ground truth
- 后面还需要更直接的交互测量

---

## 9. 细粒度 taxonomy 的意义

我们不是只想要一个总分，而是想要“像 A/N/C 但更细”的风格体系。

在 NAVSIM mini 上，聚类后得到的 prototype 说明：
- 有典型的 `calm_cautious`
- 有典型的 `social_margin_keeper`
- 有典型的 `efficient_smooth`
- 有典型的 `pressure_prone_close_interaction`
- 也有 `low_progress_or_freeze_risk*` 这种失败型诊断风格

这说明模型不是“只有好坏”之分，而是会在不同场景下落入不同风格原型。

因此，最终应该报告：
- 一个总分：`YDPS`
- 一个风格向量：`x`
- 一个风格分布：`style mixture`

这样才能真正说明“个性化驾驶”。

---

## 10. 结果解读

### 10.1 PDM 结果
PDM 依旧更偏任务完成和安全。

### 10.2 YDPS 结果
`YDPS` 和 PDM 有相关性，但不是完全一致。

这说明它没有退化成 PDM 的拷贝，而是额外引入了：
- 风格对齐
- 非冻结
- 社交裕度
- 交互压力

### 10.3 Fine-grained style 结果
taxonomy 显示：
- `human` 和 `diffusiondrive` 大量集中在 `calm_cautious`
- `transfuser` 和 `ltf` 也有较多 `social_margin_keeper`
- `constant_velocity` 明显落在 `pressure_prone` 和 `low_progress_or_freeze_risk`

这说明 taxonomy 能把：
- 正常谨慎
- 过度社交保守
- 压迫型交互
- 冻结/爬行型失败

区分开。

---

## 11. 这套 metric 的优点

### 11.1 可解释
每个维度都能讲清楚它在测什么。

### 11.2 不依赖 A/N/C
它是连续、多维 persona space，不局限于 aggressive / normal / conservative。

### 11.3 不会把“慢”当成“保守”
因为 `anti_freeze`、`progress_persistence`、`task_score` 都会压制这种退化。

### 11.4 能反映交互风格
通过 `courtesy_social`、`pressure_avoidance`、`caution_margin` 等维度，可以把“是否压迫别人”“是否留足社交空间”纳入评价。

### 11.5 能做 finer taxonomy
这点是关键：它不是一个统一 metric，而是一个可以进一步分解成多种风格原型的体系。

---

## 12. 当前局限

这套 metric 还是第一版原型，主要局限是：
- 交互项大部分还是 proxy。
- 还没有真正直接测 `forced braking of others`。
- 还没有显式的 `intersection hesitation`。
- 还没有完整的 `yield correctness` 和 `blocking` 指标。
- taxonomy 的命名还偏 heuristic。

---

## 13. 后续应该怎么升级

如果要把它做成更强的论文版本，下一步建议补这些维度：

- `intersection_hesitation`
- `forced_braking_proxy`
- `gap_acceptance`
- `over_yielding`
- `blocking_behavior`
- `yield_correctness`

这样就能更接近“真实个性化驾驶行为”的定义，而不是只靠轨迹平滑度和距离 proxy。

---

## 14. 一句话总结

YouDrive 的 metric 不是单一 score，而是一个三层系统：

1. `YDPS` 负责安全和任务门控。
2. `10` 维连续 style vector 负责表达驾驶人格。
3. `fine-grained taxonomy` 负责把不同模型分到多个可解释风格原型里。

这套设计的核心思想是：
- 安全不能被 style 补偿；
- 风格不能退化成低速或停滞；
- 个性化驾驶应该是“可解释的、多维的、场景条件化的”。
