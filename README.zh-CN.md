# Neural-BVI

神经网络提供初始 GPR 反演，低维残差空间中的 boosting variational inference 结合物理似然，输出重建均值、空间不确定性与高介电常数事件概率。

入口与目录说明见 [英文 README](README.md)。当前版本包含可安装 Python 包、CPU 示例、原始研究模块、冻结实验配置、逐记录指标、测试与 GitHub Actions。

项目采用 [Apache-2.0 许可证](LICENSE)，适用范围见 [许可证说明](LICENSE_POLICY.md)。源码仓库为 [Bilybill/Neural-BVI](https://github.com/Bilybill/Neural-BVI)，版本发布步骤见 [发布清单](docs/release_checklist.md)。

```bash
python -m pip install -e ".[dev]"
python -m neural_bvi doctor
python -m neural_bvi demo --output outputs/demo
python scripts/verify_results.py
python -m pytest
```

示例使用程序生成数据，依次执行小型网络训练、正演梯度检查和 BVI 后验更新，无需实测数据。它用于验证程序链路，不对应论文的完整实验设置。

论文结果对应 20 个地下模型、每个模型 0/5/10 dB 三个视图和六种方法。最终方法包含神经预测离散度增强、均值选择和开发集校准；这些细节在 [方法说明](docs/method.md) 中公开。早期已查看的测试模型已作为开发数据，最终测试索引和冻结配置保留在 `provenance/`。

完整复现需要原始数据、实测残差噪声库和训练权重；这些文件尚未随源码发布。获取输入后，可使用 [复现步骤](docs/reproducibility.md) 中的命令执行。原始实测数据、论文投稿材料、私人路径与内部审计文档不包含在发布包中。
