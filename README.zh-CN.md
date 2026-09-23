<p align="center">
  <img src="docs/assets/brand/neural-bvi-logo.png" alt="Neural-BVI 标志：雷达回波、地下分层与概率曲线" width="560">
</p>

神经网络提供初始 GPR 反演，低维残差空间中的 boosting variational inference 结合物理似然，输出重建均值、空间不确定性与高介电常数事件概率。

入口与目录说明见 [英文 README](README.md)。当前版本包含可安装 Python 包、CPU 示例、数值计算模块、最终实验配置、逐记录指标、测试与 GitHub Actions。

项目采用 [Apache-2.0 许可证](LICENSE)，适用范围见 [许可证说明](LICENSE_POLICY.md)。源码仓库：[Bilybill/Neural-BVI](https://github.com/Bilybill/Neural-BVI)。

## 主要特点

- 神经网络初始化与可微物理后验更新相结合。
- 在报告的六方法测试实验中，Image NRMSE、Data RMSE 和 CRPS 均值最优。
- 同时输出介电常数重建、空间不确定性及目标事件概率。
- 提供无需下载数据的 CPU 示例、明确的实验配置和自动化测试。

## 方法框架

![Neural-BVI 方法架构](docs/assets/paper/architecture.png)

**图 1：Neural-BVI 框架。** 神经网络提供初始介电常数模型，物理似然驱动低维残差后验更新，模型空间先验约束更新幅度，后验样本同时给出重建均值、不确定性与目标事件概率。

## 快速开始

```bash
python -m pip install -e ".[dev]"
python -m neural_bvi doctor
python -m neural_bvi demo --output outputs/demo
python scripts/verify_results.py
python -m pytest
```

示例使用程序生成数据，依次执行小型网络训练、正演梯度检查和 BVI 后验更新，无需实测数据。它用于验证程序链路，不对应论文的完整实验设置。

论文结果对应 20 个地下模型、每个模型 0/5/10 dB 三个视图和六种方法。最终方法包含神经预测离散度增强、均值选择和开发集校准，详见 [方法说明](docs/method.md)。测试模型身份定义在 `configs/paper/test_set.json`，完整指标保留在 `results/paper/`。

## 论文实验图

![合成测试集上的重建与后验结果](docs/assets/paper/synthetic-results.png)

**图 2：5 dB 合成测试样例。** 展示观测 B-scan、真实模型、神经网络中心、Neural-BVI 后验均值、校准后的标准差、绝对误差和高介电常数事件概率。这是论文测试样例，不是快速示例的输出。

![Neural-BVI 与 FWI 的实际数据对比](docs/assets/paper/field-results.png)

**图 3：实际数据实验。** 对比 Neural-BVI 与全波形反演（FWI）的重建、回代波形及残差，并展示结构不确定性、总体指标改善和分时间窗 RMSE。该记录中，Neural-BVI 的源校正 RMSE 从 0.1602 降至 0.1566，包络 RMSE 从 0.1946 降至 0.1870。

三张图均直接由论文中的 PDF 原图导出，未修改数据或图内标注，详见 [图片说明](docs/assets/README.md)。Logo 为 AI 生成的项目标志，不用于科研结果。

完整合成实验复现需要原始数据、实测残差噪声库和训练权重；这些文件尚未随源码发布。获取输入后，可使用 [复现步骤](docs/reproducibility.md) 执行。当前公开执行入口覆盖合成数据训练、评估及自包含示例，实测图作为论文结果展示。
