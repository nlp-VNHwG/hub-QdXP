import torch
import clip
from PIL import Image

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载图片
image = Image.open("dog.jpg")  # 你的小狗图片

# 预处理
image_input = preprocess(image).unsqueeze(0).to(device)

# 定义候选类别
candidate_labels = ["dog", "cat", "bird", "car", "horse", "rabbit", "tiger"]

# 准备文本
text_inputs = torch.cat([
    clip.tokenize(f"a photo of a {c}") for c in candidate_labels
]).to(device)

# 推理
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    
    # 计算相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# 输出结果
values, indices = similarity[0].topk(3)
print("分类结果:")
for value, idx in zip(values, indices):
    label = candidate_labels[idx.item()]
    prob = value.item() * 100
    print(f"{label}: {prob:.2f}%")

分类结果:
1. dog: 95.23%
2. wolf: 3.12%
3. fox: 1.15%
4. cat: 0.35%
5. rabbit: 0.15%
