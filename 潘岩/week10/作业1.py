import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

# 加载 中文CLIP 模型和处理器
model = ChineseCLIPModel.from_pretrained('models/chinese-clip-vit-base-patch16/')
processor = ChineseCLIPProcessor.from_pretrained('models/chinese-clip-vit-base-patch16/')

# 加载本地图片
image_path = "images.jpg"
img = [Image.open(image_path).convert("RGB")]

# --------------------- 1. 提取图像特征 ---------------------
input = processor(images=img, return_tensors='pt')
img_image_feat = []
with torch.no_grad():
    image_feature = model.get_image_features(**input)
    image_feature = image_feature.data.numpy()
    img_image_feat.append(image_feature)

img_image_feat = np.vstack(img_image_feat)
img_image_feat = normalize(img_image_feat)

# --------------------- 2. 提取文本特征 ---------------------
img_texts_feat = []
# 修改：把“树”改成正确的动物类
texts = ['这是一只：小狗', '这是一只：小猫', '这是一只：小鸟', '这是一只：鱼', '这是一只：兔子']

inputs = processor(text=texts, return_tensors='pt', padding=True)
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    text_features = text_features.data.numpy()
    img_texts_feat.append(text_features)

img_texts_feat = np.vstack(img_texts_feat)
img_texts_feat = normalize(img_texts_feat)

# --------------------- 3. 计算相似度---------------------
sim_result = np.dot(img_image_feat[0], img_texts_feat.T)  # 图片 vs 所有文本 相似度
sim_idx = sim_result.argsort()[::-1][0]  # 相似度从高到低排序，取第1个

# --------------------- 4. 输出最终结果 ---------------------
print("所有类别相似度：")
for i, score in enumerate(sim_result):
    print(f"{texts[i]} → 相似度：{score:.4f}")

print("\n🎉 最终分类结果：", texts[sim_idx])
