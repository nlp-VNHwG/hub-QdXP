# 导入PIL库中的Image模块，用于处理图像
from PIL import Image
# 导入requests库，用于网络请求（虽然代码中未使用，但保留以备需要）
import requests
# 从transformers库导入ChineseCLIPProcessor和ChineseCLIPModel
# ChineseCLIPProcessor: 用于处理文本和图像的预处理器
# ChineseCLIPModel: 中文CLIP模型，用于计算图像和文本的相似度
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
# 导入PyTorch库，用于深度学习计算
import torch
# 导入numpy库，用于数值计算
import numpy as np
# 导入matplotlib.pyplot，用于数据可视化
import matplotlib.pyplot as plt

# 设置matplotlib使用黑体字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
# 解决负号显示为方块的问题，确保负号正常显示
plt.rcParams['axes.unicode_minus'] = False
# ==================================

# 设置设备：如果有GPU则使用CUDA，否则使用CPU
# torch.cuda.is_available()检查是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义Zero-Shot分类函数
def zero_shot_classification(image, labels, model, processor, device):
    """
    使用 CLIP 进行 Zero-Shot 分类
    
    参数:
        image: PIL Image对象，待分类的图片
        labels: list，候选类别标签列表
        model: CLIP模型，用于提取特征
        processor: 预处理器，用于将图像和文本转换为模型输入
        device: str，计算设备（cuda或cpu）
    
    返回:
        probs: numpy数组，每个类别的概率
    """
    # 步骤1：编码图片
    # processor将图片转换为模型可接受的张量格式
    # return_tensors="pt"表示返回PyTorch张量
    # padding=True表示自动填充文本到相同长度
    # .to(device)将数据移动到指定设备（GPU或CPU）
    image_inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    
    # 步骤2：编码文本
    # 将候选标签列表转换为模型输入
    text_inputs = processor(text=labels, return_tensors="pt", padding=True).to(device)
    
    # 步骤3：推理（不计算梯度，节省内存）
    with torch.no_grad():
        # 获取图像特征
        # get_image_features()方法返回图像的特征向量
        image_outputs = model.get_image_features(**image_inputs)
        
        # 处理返回结果：如果是BaseModelOutputWithPooling对象，提取pooler_output
        # 有些版本的CLIP返回的是包含pooler_output的对象，需要提取
        if hasattr(image_outputs, 'pooler_output'):
            # 如果有pooler_output属性，则使用池化后的特征
            image_features = image_outputs.pooler_output
        else:
            # 否则直接使用返回的特征
            image_features = image_outputs
        
        # 获取文本特征
        text_outputs = model.get_text_features(**text_inputs)
        # 同样处理文本特征
        if hasattr(text_outputs, 'pooler_output'):
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs
        
        # 特征归一化：将特征向量除以它的模长
        # 归一化后所有向量的模长为1，便于计算余弦相似度
        # dim=-1表示在最后一个维度上进行归一化
        # keepdim=True保持维度不变
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度矩阵
        # @ 是矩阵乘法运算符，计算图像特征和文本特征的相似度
        # 乘以100作为缩放因子，放大差异，使softmax结果更明显
        similarity = (image_features @ text_features.T) * 100
        
        # 使用softmax将相似度转换为概率分布
        # softmax函数将数值转换为概率（所有概率之和为1）
        # dim=-1表示在最后一个维度（文本类别维度）上计算softmax
        probs = similarity.softmax(dim=-1)
    
    # 返回概率数组，并转换为numpy数组，移到CPU
    # [0]表示取第一个（也是唯一一个）图像的结果
    return probs[0].cpu().numpy()

def display_results(image, labels, probs, top_k=5):
    """
    显示分类结果，包括图片和柱状图
    
    参数:
        image: PIL Image对象，原始图片
        labels: list，候选类别标签列表
        probs: numpy数组，每个类别的概率
        top_k: int，显示前k个结果，默认为5
    """
    # 获取概率最高的top_k个结果的索引
    # np.argsort(probs)返回排序后的索引，[::-1]反转得到降序
    # [:top_k]取前top_k个
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    # 获取top_k对应的标签和概率
    top_labels = [labels[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]
    
    # 在控制台打印分类结果
    print("\n" + "="*50)
    print("Zero-Shot 图像分类结果:")
    print("="*50)
    # 遍历top_k个结果并打印
    for i, (label, prob) in enumerate(zip(top_labels, top_probs)):
        print(f"{i+1}. {label}: {prob:.4f} ({prob*100:.2f}%)")
    
    # 创建图形窗口，设置大小为10x5英寸
    plt.figure(figsize=(10, 5))
    
    # 创建第一个子图（1行2列的第1个），用于显示原始图片
    plt.subplot(1, 2, 1)
    # 显示图片
    plt.imshow(image)
    # 关闭坐标轴显示
    plt.axis('off')
    # 设置子图标题
    plt.title("输入图片", fontsize=14)
    
    # 创建第二个子图（1行2列的第2个），用于显示结果柱状图
    plt.subplot(1, 2, 2)
    
    
    # 创建颜色映射：使用RdYlGn_r（红-黄-绿）颜色方案
    # np.linspace(0.2, 0.8, len(top_labels))生成等间距的数值
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_labels)))
    # 创建水平柱状图
    # barh(y位置, 宽度, 颜色)
    bars = plt.barh(range(len(top_labels)), top_probs, color=colors)
    # 设置y轴刻度标签为类别标签
    plt.yticks(range(len(top_labels)), top_labels)
    # 设置x轴标签
    plt.xlabel("概率", fontsize=12)
    # 设置图表标题
    plt.title("Zero-Shot 分类结果", fontsize=14)
    
    # 在每个柱子上显示具体的概率数值
    # enumerate(bars, top_probs)同时遍历柱子对象和概率值
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        # 在柱子右侧添加文本
        # prob是x坐标，bar.get_y() + bar.get_height()/2是y坐标（柱子中间）
        # f'{prob:.3f}'格式化概率值为3位小数
        # ha='left'水平左对齐，va='center'垂直居中
        plt.text(prob, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', ha='left', va='center')
    
    # 自动调整子图参数，使之填充整个图像区域
    plt.tight_layout()
    # 显示图形
    plt.show()
    
    
        
def main():
    """
    主函数：程序入口
    """
    # 设置图片路径
    image_path = "dog.jpg"  # 小狗图片的路径
    
    # 加载中文CLIP模型
    # from_pretrained从HuggingFace下载或加载预训练模型
    # "OFA-Sys/chinese-clip-vit-base-patch16"是模型在HuggingFace上的名称
    model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    # 加载对应的处理器，用于预处理图像和文本
    processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    
    # 加载小狗图片
    # Image.open()打开图片文件
    # .convert('RGB')将图片转换为RGB色彩模式，确保格式统一
    image = Image.open("D:/AI/AI work/Week 10/week10/dog.jpg").convert('RGB')
    
    # 准备候选描述标签
    # 这些是模型将要分类的类别
    candidate_texts = [
        "猫", "狗", "鸟", "鱼", "汽车", "飞机", "火车", "船",  # 动物和交通工具
        "花", "树", "山", "海", "城市", "乡村", "食物", "人物",  # 自然和场景
        "风景", "动物", "建筑", "交通工具"  # 更广泛的类别
    ]
    
    # 进行Zero-Shot分类
    print("\n正在进行 Zero-Shot 分类...")
    # 调用分类函数，传入图片、标签、模型、处理器和设备
    probs = zero_shot_classification(image, candidate_texts, model, processor, device)
    
    # 显示分类结果
    # top_k=5表示显示前5个最可能的类别
    display_results(image, candidate_texts, probs, top_k=5)
    
# 程序入口点
# 当直接运行此脚本时（而不是作为模块导入），执行main函数
if __name__ == "__main__":
    main()