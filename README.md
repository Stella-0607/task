# 心力衰竭患者生存预测与医疗AI新手任务

---

## 结构
```text
.
├── README.md                              # 说明文档
├── Task1_API/                             # 任务1：API开发模块
│   ├── task1code.py                       # API接口实现代码
│   └── task1result.json                   # API调用结果示例
├── Task2_Paper/                           # 任务2：建模与论文模块
│   ├── Task2_code.py                      # 模型训练与评估完整代码
│   ├── Task2_essay.pdf                    # 项目论文报告
│   ├── heart_failure_clinical_records_dataset.csv  # 心力衰竭临床数据集
│   ├── heatmap.png                        # 特征相关性热力图
│   ├── importance.png                     # 模型特征重要性排序图
│   ├── roc_curve.png                      # 多模型ROC曲线对比图
│   ├── workflow.png                       # 项目分析流程图
│   ├── 前三大核心危险因子临床生存分析.png
│   ├── 四模型混淆矩阵对比图.png
│   ├── 血清肌酐偏态平滑对比图.png
│   └── 血清肌酐对数变换前后分布对比图.png
└── Task3_Guide/                           # 任务3：医疗大模型应用指南
    ├── 医疗大模型新人快速上手指南.md
    └── 医疗大模型新人快速上手指南.pdf
```

## 任务说明
1. 挑战一：基于大模型的中文病例实体抽取（测试：API调用、代码基础、GitHub）

指定一份中文病例（Case Report）`A case of portal vein recanalization and symptomatic heart failure.pdf`。你需要编写 Python 脚本，通过调用大模型 API 从中准确提取关键的医学实体。

2. 挑战二：心衰患者生存预测与 LaTeX 小型英文论文撰写（测试：ML/DL实战、AI生图、LaTeX排版）

下载心力衰竭临床记录数据集 (Heart Failure Clinical Records)`heart_failure_clinical_records_dataset.csv`（包含数百名患者的年龄、贫血、射血分数、血清肌酐等 13 个字段），完成从数据清洗到建立预测模型的全流程，并用 LaTeX 将其写成一篇小型论文。

3. 挑战三：经验沉淀与新人避坑指南（测试：复盘总结、Markdown排版、知识库建设意识）

结合完成前两个挑战的亲身经历，使用 Markdown 编写一份《医疗大模型新人快速上手指南》
