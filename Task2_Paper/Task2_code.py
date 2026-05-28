import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # 新增：逻辑回归 Baseline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# --- 【🔍 新增包：用于自动化计算偏度、峰度以及医学统计学检验 p 值】 ---
from scipy.stats import skew, kurtosis, ranksums, chi2_contingency 

# 设置绘图风格与随机种子保证结果可复现
sns.set_theme(style="whitegrid")
np.random.seed(42)
torch.manual_seed(42)

# 设置 matplotlib 支持中文显示（防止图片中的标签乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

# =====================================================================
# 1. 数据初处理 (Data Exploratory & Preprocessing)
# =====================================================================

# 1.1 加载数据 (请确保 csv 文件在当前工作目录下，或填写正确路径)
df = pd.read_csv('D:/STELLA/科研资料/新生文档/新手任务/heart_failure_clinical_records_dataset.csv')

print("--- 数据基本信息 ---")
print(df.info())
print("\n--- 描述性统计 ---")
print(df.describe().T)

# 1.2 检查缺失值
missing_values = df.isnull().sum().sum()
print(f"\n总缺失值数量: {missing_values}")

# --- 【优化方案：新增 Table 1 自动化打印】 ---
print("\n" + "="*20 + " [LaTeX Table 1: 患者基线临床特征统计表数据] " + "="*20)
baseline_stats = []
for col in df.columns:
    if col == 'DEATH_EVENT': continue
    if df[col].nunique() <= 2:  # 二分类变量统计百分比
        p_all = df[col].mean() * 100
        p_live = df[df['DEATH_EVENT']==0][col].mean() * 100
        p_dead = df[df['DEATH_EVENT']==1][col].mean() * 100
        baseline_stats.append({
            '临床特征': col,
            '总体人群 (N=299)': f"{p_all:.2f}%",
            '存活组 (n=203)': f"{p_live:.2f}%",
            '死亡组 (n=96)': f"{p_dead:.2f}%"
        })
    else:  # 连续变量统计 mean ± sd
        m_all, s_all = df[col].mean(), df[col].std()
        m_live, s_live = df[df['DEATH_EVENT']==0][col].mean(), df[df['DEATH_EVENT']==0][col].std()
        m_dead, s_dead = df[df['DEATH_EVENT']==1][col].mean(), df[df['DEATH_EVENT']==1][col].std()
        baseline_stats.append({
            '临床特征': col,
            '总体人群 (N=299)': f"{m_all:.2f} ± {s_all:.2f}",
            '存活组 (n=203)': f"{m_live:.2f} ± {s_live:.2f}",
            '死亡组 (n=96)': f"{m_dead:.2f} ± {s_dead:.2f}"
        })
print(pd.DataFrame(baseline_stats).to_string(index=False))
print("="*75 + "\n")


# --- 【🔍 新增 Table 2 代码：连续变量对数变换前后偏态与峰度对照表】 ---
print("\n" + "="*20 + " [LaTeX Table 2: 核心连续变量对数变换前后偏态与峰度对照表] " + "="*20)
table2_records = []
for col in ['serum_creatinine', 'creatinine_phosphokinase']:
    skew_before = skew(df[col])
    kurt_before = kurtosis(df[col])
    log_var = np.log1p(df[col])
    skew_after = skew(log_var)
    kurt_after = kurtosis(log_var)
    table2_records.append({
        '临床连续特征': col,
        '变换前偏度 (Skewness)': f"{skew_before:.4f}",
        '变换后偏度 (Skewness)': f"{skew_after:.4f}",
        '变换前峰度 (Kurtosis)': f"{kurt_before:.4f}",
        '变换后峰度 (Kurtosis)': f"{kurt_after:.4f}"
    })
print(pd.DataFrame(table2_records).to_string(index=False))
print("="*85 + "\n")


# 1.3 异常值检测与处理 (以血清肌酐 serum_creatinine 为例，医学上极值常见，此处通过对数变换缓解偏态)
# --- 【优化方案：新增可视化 1 - 偏态双峰分布与对数变换对比图】 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
sns.histplot(df['serum_creatinine'], kde=True, ax=axes[0], color='#4C72B0', stat="density")
axes[0].set_title('血清肌酐原始偏态分布 (右偏)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Serum Creatinine (mg/dL)')

# 执行对数变换
df['log_serum_creatinine'] = np.log1p(df['serum_creatinine']) 

sns.histplot(df['log_serum_creatinine'], kde=True, ax=axes[1], color='#55A868', stat="density")
axes[1].set_title('对数变换 $\log(1+x)$ 后的平滑分布', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Log-transformed Serum Creatinine')
plt.suptitle('血清肌酐数据清洗前后的偏态平滑对比分析', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('D:/STELLA/科研资料/新生文档/新手任务/血清肌酐对数变换前后分布对比图.png', dpi=300, bbox_inches='tight')
plt.close()

# 1.4 特征与标签分离
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

# 1.5 划分训练集与测试集 (8:2 比例，分层抽样保持标签比例一致)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1.6 数据标准化/归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 转换回DataFrame方便后续特征工程分析
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)


# =====================================================================
# 2. 因子检测 (Risk Factor Detection & Feature Engineering)
# =====================================================================

# 2.1 绘制相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Heart Failure Records')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.savefig('D:/STELLA/科研资料/新生文档/新手任务/heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.2 基于随机森林的特征重要性输出排名前三的危险因子
rf_for_importance = RandomForestClassifier(random_state=42)
rf_for_importance.fit(X_train_raw, y_train)

importances = rf_for_importance.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n--- 特征重要性排名 (Random Forest) ---")
for f in range(3):
    print(f"{f+1}. 特征: {X.columns[indices[f]]} ({importances[indices[f]]:.4f})")


# --- 【🔍 新增 Table 4 代码：特征权重与临床差异性检验统计表】 ---
print("\n" + "="*20 + " [LaTeX Table 4: 临床特征模型重要性权重与组间显著性检验表] " + "="*20)
table4_records = []
live_group = df[df['DEATH_EVENT'] == 0]
dead_group = df[df['DEATH_EVENT'] == 1]

for i in indices:
    col = X.columns[i]
    weight = importances[i]
    if df[col].nunique() <= 2:
        contingency_table = pd.crosstab(df[col], df['DEATH_EVENT'])
        _, p_val, _, _ = chi2_contingency(contingency_table)
        test_method = "卡方检验 (Chi-Square)"
    else:
        _, p_val = ranksums(live_group[col], dead_group[col])
        test_method = "秩和检验 (Wilcoxon)"
    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
    significance = "是 (Yes)" if p_val < 0.05 else "否 (No)"
    table4_records.append({
        '临床特征变量': col,
        'RF 权重 (Weight)': f"{weight:.4f}",
        '统计检验方法': test_method,
        'p-value': p_str,
        '组间显著差异 (p<0.05)': significance
    })
print(pd.DataFrame(table4_records).to_string(index=False))
print("="*85 + "\n")


# 1. 准备所有特征的数据
all_features = [X.columns[i] for i in indices]
all_importances = importances[indices]

# 2. 初始化高分辨率画布
plt.figure(figsize=(10, 6), dpi=300)

# 3. 绘制横向条形图
sns.barplot(x=all_importances, y=all_features, palette="Blues_r", edgecolor='black')

# 4. 自动为每个条形图右侧贴上权重标签
for i, val in enumerate(all_importances):
    plt.text(val + 0.003, i, f'{val:.4f}', va='center', ha='left', 
             fontsize=10, color='black', fontweight='bold' if i < 3 else 'normal')

# 5. 图表学术细节美化
plt.title('基于随机森林的随访生存预测特征重要性全局排序', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('平均基尼不纯度减少量 (Mean Decrease Gini)', fontsize=12)
plt.ylabel('临床特征变量', fontsize=12)
plt.xlim(0, max(all_importances) + 0.04)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 6. 紧凑排版并保存图片
plt.tight_layout()
plt.savefig('importance.png', bbox_inches='tight')
plt.savefig('D:/STELLA/科研资料/新生文档/新手任务/importance.png', dpi=300, bbox_inches='tight')
plt.close()


# --- 【优化方案：新增可视化 2 - 前三大危险因子的临床生存分层箱线拼图】 ---
top_3_features = [X.columns[indices[0]], X.columns[indices[1]], X.columns[indices[2]]]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
for i, col in enumerate(top_3_features):
    sns.boxplot(x=y, y=df[col], ax=axes[i], palette="coolwarm", width=0.5, linewidth=1.5)
    sns.stripplot(x=y, y=df[col], ax=axes[i], color='#707070', alpha=0.4, jitter=0.15, size=3)
    axes[i].set_title(f'核心驱动因子 {i+1}: {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('生存结局 (DEATH_EVENT)')
    axes[i].set_xticklabels(['存活组 (0)', '死亡组 (1)'])
plt.suptitle('前三大核心致病危险因子在不同预后结局中的临床分布特征', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('D:/STELLA/科研资料/新生文档/新手任务/前三大核心危险因子临床生存分层图.png', dpi=300, bbox_inches='tight')
plt.close()


# =====================================================================
# 3. 概率预测 (Modeling) 与模型对比
# =====================================================================

# 3.1 模型初始化（优化：加入 Logistic Regression 作为 Baseline 基准）
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss', random_state=42)
}

# 存储各模型的评估结果
results = {}
# --- 【🔍 新增容器：用于记录各模型在训练集上的表现以排查过拟合】 ---
train_eval_results = {} 

# 训练并评估传统模型
for name, model in models.items():
    # 优化方案逻辑：如果模型是逻辑回归，调配标准化数据；树模型采用原始真实尺度数据
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        # 记录训练集表现
        y_train_pred = model.predict(X_train_scaled)
        y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
    else:
        model.fit(X_train_raw, y_train)
        y_pred = model.predict(X_test_raw)
        y_prob = model.predict_proba(X_test_raw)[:, 1]
        # 记录训练集表现
        y_train_pred = model.predict(X_train_raw)
        y_train_prob = model.predict_proba(X_train_raw)[:, 1]
    
    results[name] = {
        'pred': y_pred,
        'prob': y_prob
    }
    train_eval_results[name] = {
        'pred': y_train_pred,
        'prob': y_train_prob
    }

# 3.2 深度学习项：搭建 MLP (多层感知机)
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class HeartFailureMLP(nn.Module):
    def __init__(self, input_dim):
        super(HeartFailureMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        return self.network(x)

mlp_model = HeartFailureMLP(X_train_scaled.shape[1])
criterion = nn.BCELoss() 
optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)

mlp_model.train()
for epoch in range(50): 
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = mlp_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

mlp_model.eval()
with torch.no_grad():
    mlp_prob = mlp_model(X_test_tensor).numpy().flatten()
    mlp_pred = (mlp_prob >= 0.5).astype(int)
    # 记录 MLP 训练集表现
    mlp_train_prob = mlp_model(X_train_tensor).numpy().flatten()
    mlp_train_pred = (mlp_train_prob >= 0.5).astype(int)

results['MLP (Deep Learning)'] = {
    'pred': mlp_pred,
    'prob': mlp_prob
}
train_eval_results['MLP (Deep Learning)'] = {
    'pred': mlp_train_pred,
    'prob': mlp_train_prob
}


# =====================================================================
# 4. 模型评估与可视化 (Evaluation)
# =====================================================================

print("\n--- 模型性能评估对比 ---")
plt.figure(figsize=(8, 6), dpi=300)

# 用于存储最终 DataFrame 表格数据的字典
table2_data = []
# --- 【🔍 新增容器：用于存储 Table 6 训练与测试过拟合对比表数据】 ---
table6_data = [] 

# --- 【优化方案：新增可视化 3 - 四模型混淆矩阵对比拼图画布准备】 ---
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(10, 9), dpi=300)
axes_cm = axes_cm.ravel()

for idx, (name, metrics) in enumerate(results.items()):
    acc = accuracy_score(y_test, metrics['pred'])
    prec = precision_score(y_test, metrics['pred'])
    rec = recall_score(y_test, metrics['pred'])
    f1 = f1_score(y_test, metrics['pred'])
    
    fpr, tpr, _ = roc_curve(y_test, metrics['prob'])
    roc_auc = auc(fpr, tpr)
    
    # 计算对应的训练集指标
    acc_tr = accuracy_score(y_train, train_eval_results[name]['pred'])
    fpr_tr, tpr_tr, _ = roc_curve(y_train, train_eval_results[name]['prob'])
    roc_auc_tr = auc(fpr_tr, tpr_tr)
    
    print(f"\n[{name}]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {roc_auc:.4f}")
    
    # 填充原有的 Table 2 字典
    table2_data.append({
        'Model': name,
        'Accuracy': f"{acc*100:.2f}%",
        'Precision': f"{prec:.4f}",
        'Recall': f"{rec:.4f}",
        'F1-Score': f"{f1:.4f}",
        'AUC': f"{roc_auc:.2f}"
    })
    
    # --- 【🔍 填充新增的 Table 6 字典：过拟合风险对比】 ---
    table6_data.append({
        '评估模型 (Model)': name,
        '训练集准确率 (Train Acc)': f"{acc_tr*100:.2f}%",
        '测试集准确率 (Test Acc)': f"{acc*100:.2f}%",
        '训练集曲线面积 (Train AUC)': f"{roc_auc_tr:.4f}",
        '测试集曲线面积 (Test AUC)': f"{roc_auc:.4f}",
        'AUC泛化衰减率 (Drop)': f"{(roc_auc_tr - roc_auc):.4f}"
    })
    
   # =====================================================================
# 4. 模型评估与可视化 (Evaluation)
# =====================================================================

print("\n--- 模型性能评估对比 ---")
plt.figure(figsize=(8, 6))

for name, metrics in results.items():
    acc = accuracy_score(y_test, metrics['pred'])
    prec = precision_score(y_test, metrics['pred'])
    rec = recall_score(y_test, metrics['pred'])
    f1 = f1_score(y_test, metrics['pred'])
    
    # 计算 ROC 和 AUC
    fpr, tpr, _ = roc_curve(y_test, metrics['prob'])
    roc_auc = auc(fpr, tpr)
    
    print(f"\n[{name}]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {roc_auc:.4f}")
    
    # 绘制 ROC 曲线
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# 绘制对角随机线
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.savefig('D:/STELLA/科研资料/新生文档/新手任务/roc_curve.png', dpi=300, bbox_inches='tight')

# =====================================================================
# 4.3 新增：独立的四模型混淆矩阵对比图绘制 (置于 ROC 曲线代码下方)
# =====================================================================
print("\n--- 正在独立绘制四模型混淆矩阵对比图... ---")

# 1. 初始化 2x2 的田字格高分辨率画布，彻底与 ROC 画布隔离
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(10, 9.5), dpi=300)
axes_flat = axes_cm.flatten()  # 拉平成一维数组方便通过索引 [0, 1, 2, 3] 遍历

# 2. 遍历结果字典，自动化绘制 2x2 拼图
for idx, (name, metrics) in enumerate(results.items()):
    # 计算当前模型的混淆矩阵
    cm = confusion_matrix(y_test, metrics['pred'])
    current_ax = axes_flat[idx]  # 锁定当前子图格子
    
    # 绘制热力图：
    # - annot=True 填入数字，fmt='d' 强制整数
    # - square=True 强制格子为正方形，不拉伸变形
    # - cbar=False 移除右侧独立的颜色条，彻底杜绝排版错位和数字乱飞
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=current_ax, cbar=False,
                square=True, edgecolor='black', linewidths=1.2, 
                annot_kws={'size': 14, 'weight': 'bold'})
    
    # 精细微调单个子图格子的文本与学术排版
    current_ax.set_title(f'{name} 混淆矩阵', fontsize=12, fontweight='bold', pad=10)
    current_ax.set_xlabel('模型预测类别', fontsize=10, labelpad=6)
    current_ax.set_ylabel('真实临床结局', fontsize=10, labelpad=6)
    current_ax.set_xticklabels(['存活 (0)', '死亡 (1)'], fontsize=9)
    current_ax.set_yticklabels(['存活 (0)', '死亡 (1)'], fontsize=9, va='center')

# 3. 为总大标题留出顶部空间，并精细拉开四个格子之间的上下左右间距（防止标签重叠）
fig_cm.suptitle('四种分类预测模型在独立测试集上的混淆矩阵对照图谱', fontsize=14, fontweight='bold', y=0.98)
fig_cm.subplots_adjust(wspace=0.35, hspace=0.35) 

# 4. 规范化保存并即时销毁画布，给下一段程序或内存让出完全干净的空间
fig_cm.savefig('四模型混淆矩阵对比图.png', dpi=300, bbox_inches='tight')
fig_cm.savefig('D:/STELLA/科研资料/新生文档/新手任务/四模型混淆矩阵对比图.png', dpi=300, bbox_inches='tight')
plt.close(fig_cm)

print("--- 混淆矩阵对比图 [四模型混淆矩阵对比图.png] 已完美导出！ ---")

# --- 【优化方案：新增 Table 2 性能性能对比规整 DataFrame 控制台输出】 ---
print("\n" + "="*20 + " [LaTeX Table 3: 多模型测试集生存预测泛化性能对比表] " + "="*20)
df_table2 = pd.DataFrame(table2_data)
print(df_table2.to_string(index=False))
print("="*85 + "\n")



# --- 【🔍 新增 Table 6 代码：自动化控制台打印过拟合风险分析表】 ---
print("\n" + "="*20 + " [LaTeX Table 6: 四种模型训练集 vs 测试集过拟合风险分析表] " + "="*20)
df_table6 = pd.DataFrame(table6_data)
print(df_table6.to_string(index=False))
print("="*85 + "\n")