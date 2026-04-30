import os
from pdf2image import convert_from_path
from openai import OpenAI
import base64
from io import BytesIO

def analyze_pdf_first_page(pdf_path, api_key=None):
    """
    快速分析PDF第一页
    
    参数:
    pdf_path: PDF文件路径
    api_key: 阿里云API Key（可选，默认从环境变量读取）
    """
    # 1. 设置API Key
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置DASHSCOPE_API_KEY环境变量")
        return
    
    # 2. 转换PDF第一页为图像
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    if not images:
        print("错误: 无法读取PDF文件")
        return
    
    image = images[0]
    
    # 3. 将图像转换为base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    image_url = f"data:image/jpeg;base64,{img_base64}"
    
    # 4. 调用Qwen-VL API
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    response = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请详细描述这个PDF页面的内容。"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=1000
    )
    
    # 5. 输出结果
    print("分析结果:")
    print("-" * 40)
    print(response.choices[0].message.content)

# 使用示例
if __name__ == "__main__":
    analyze_pdf_first_page("your_document.pdf")

获取API Key：
运行代码：
# 方法1：使用完整类
analyzer = PDFQwenVLAnalyzer()
result = analyzer.analyze_pdf_page("your_file.pdf", page_number=0)
print(result['analysis'])

# 方法2：使用简化函数
analyze_pdf_first_page("your_file.pdf")




