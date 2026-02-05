from datasets import load_dataset, DatasetDict  # Hugging Face数据集库，用于加载和处理数据集
import numpy as np  # 数值计算库，用于数值操作
import pandas as pd  # 数据处理库，用于DataFrame操作
from sklearn.model_selection import train_test_split  # 数据集划分工具，用于划分训练集和验证集

'''
用AI生成的代码，生成训练集，验证集和测试集
这段代码从Hugging Face加载AG News数据集，选择三个类别，并划分为训练集、验证集和测试集
'''

# 加载AG News数据集
# load_dataset: 从Hugging Face数据集库加载预定义的数据集
# "ag_news": AG News新闻分类数据集，包含4个类别：World, Sports, Business, Sci/Tech
dataset = load_dataset("ag_news")

# 定义要选择的三个类别及其新标签映射关系
# 原始AG News有4个类别，这里选择其中3个进行三分类任务
selected_classes = {
    1: 0,  # 原始标签1（Sports体育）映射为新标签0
    2: 1,  # 原始标签2（Business商业）映射为新标签1  
    3: 2   # 原始标签3（Sci/Tech科技）映射为新标签2
    # 注意：原始标签0（World世界新闻）被排除
}

# 处理数据集：筛选出选择的三个类别
# filter: 过滤数据集，只保留指定类别的样本
train_data = dataset["train"].filter(lambda x: x["label"] in selected_classes)  # 过滤训练集
test_data = dataset["test"].filter(lambda x: x["label"] in selected_classes)    # 过滤测试集

# 映射标签：将原始标签转换为新的标签编号（0,1,2）
# map: 对数据集中的每个样本应用函数转换
train_data = train_data.map(lambda x: {"label": selected_classes[x["label"]]})  # 训练集标签映射
test_data = test_data.map(lambda x: {"label": selected_classes[x["label"]]})    # 测试集标签映射

# 转换为Pandas DataFrame，便于进一步处理和分析
# DataFrame: Pandas的二维表格数据结构
train_df = pd.DataFrame(train_data)  # 将训练集转换为DataFrame
test_df = pd.DataFrame(test_data)    # 将测试集转换为DataFrame

# 从训练集中划分验证集（80%训练，20%验证）
# train_test_split: 将数据集划分为训练集和验证集
train_df, val_df = train_test_split(
    train_df,              # 要划分的原始训练数据
    test_size=0.2,         # 验证集比例：20%
    random_state=42,       # 随机种子，确保每次划分结果一致
    stratify=train_df['label']  # 分层抽样，保持训练集和验证集的类别比例一致
)

# 添加类别名称列，便于理解和可视化
# 定义标签ID到标签名称的映射字典
label_names = {0: "sports", 1: "business", 2: "sci_tech"}  # 标签ID对应的可读名称
# map: 将标签ID映射为标签名称
train_df['label_name'] = train_df['label'].map(label_names)  # 训练集添加标签名称
val_df['label_name'] = val_df['label'].map(label_names)      # 验证集添加标签名称
test_df['label_name'] = test_df['label'].map(label_names)    # 测试集添加标签名称

# 重新排序列顺序，使数据结构更清晰
column_order = ['text', 'label', 'label_name']  # 定义列的顺序：文本、标签ID、标签名称
train_df = train_df[column_order]  # 重新排列训练集列顺序
val_df = val_df[column_order]      # 重新排列验证集列顺序
test_df = test_df[column_order]    # 重新排列测试集列顺序

# 保存为CSV文件，便于后续使用
# to_csv: 将DataFrame保存为CSV文件
# index=False: 不保存行索引
# encoding='utf-8': 使用UTF-8编码，支持中文等特殊字符
train_df.to_csv('./ag_news_3class/train.csv', index=False, encoding='utf-8')
val_df.to_csv('./ag_news_3class/val.csv', index=False, encoding='utf-8')
test_df.to_csv('./ag_news_3class/test.csv', index=False, encoding='utf-8')

# 打印数据集统计信息
print("=" * 50)  # 打印分隔线
print("AG News 三分类数据集统计")  # 标题
print("=" * 50)

# 创建数据集字典，便于循环处理
datasets = {
    "训练集": train_df,  # 训练集DataFrame
    "验证集": val_df,    # 验证集DataFrame
    "测试集": test_df    # 测试集DataFrame
}

# 循环打印每个数据集的统计信息
for name, df in datasets.items():  # 遍历数据集字典
    print(f"\n{name}: {len(df)} 个样本")  # 打印数据集名称和样本数量
    print("-" * 30)  # 打印分隔线
    # 遍历每个类别
    for label_id, label_name in label_names.items():  # 遍历标签映射
        count = len(df[df['label'] == label_id])  # 统计当前类别的样本数量
        percentage = count / len(df) * 100  # 计算当前类别的百分比
        print(f"  {label_name}({label_id}): {count}个 ({percentage:.1f}%)")  # 打印类别统计

# 打印文件保存信息
print(f"\n文件已保存到 ./ag_news_3class/ 目录:")
print(f"  train.csv: {len(train_df)} 行")  # 训练集行数
print(f"  val.csv: {len(val_df)} 行")      # 验证集行数
print(f"  test.csv: {len(test_df)} 行")    # 测试集行数

# 显示样本示例，便于了解数据格式
print("\n样本示例（训练集前2行）:")
print(train_df.head(2).to_string(index=False))  # 显示训练集前2行，不显示索引