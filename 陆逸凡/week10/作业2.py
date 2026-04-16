import os
import base64
from openai import OpenAI
import fitz  # PyMuPDF，用于处理PDF

# PDF转成png图片
def pdf_first_page_to_base64(pdf_path):
    """
    将PDF的第一页转换为base64编码的图片
    
    参数:
        pdf_path: PDF文件路径
    
    返回:
        base64编码的图片字符串
    """
    try:
        # 打开PDF文件，创建一个文档对象
        doc = fitz.open(pdf_path)
        
        # 获取第一页（索引从0开始，所以0表示第一页）
        page = doc.load_page(0)
        
        # 设置缩放倍数（越高越清晰，但也会增加token消耗）
        # 可以使用1.5-2.0之间，2.0表示放大2倍，提高图片清晰度
        zoom = 2.0
        # 创建变换矩阵，用于控制图片的缩放比例
        matrix = fitz.Matrix(zoom, zoom)
        
        # 将页面渲染为图片（像素图对象）
        # 使用变换矩阵将PDF页面转换为像素图
        pix = page.get_pixmap(matrix=matrix)
        
        # 转换为PNG格式的字节流（二进制数据）
        img_data = pix.tobytes("png")
        
        # 关闭PDF文档，释放资源
        doc.close()
        
        # 将二进制图片数据转换为base64编码的字符串
        # base64编码用于在文本协议中传输二进制数据
        base64_image = base64.b64encode(img_data).decode('utf-8')
        
        print(f"PDF第一页转换成功！")
        
        return base64_image
        
    except Exception as e:
        print(f"PDF转换失败: {e}")
        return None


# 创建OpenAI客户端实例，用于调用阿里云百炼平台的大模型API
# 客户端对象封装了API请求的配置信息
client = OpenAI(
    # 请替换为自己的百炼API Key，用于身份认证
    api_key="自己的api key",
    # 北京地域的base_url，指定API服务的访问地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# 指定要处理的PDF文件路径
pdf_path = "应阔浩-2025自如企业级AI架构落地的思考与实践.pdf"  # 替换为你的PDF路径

# 检查文件是否存在，避免后续操作出错
if not os.path.exists(pdf_path):
    print(f"找不到PDF文件 - {pdf_path}")
    print("请检查文件路径是否正确")
    exit(1)  # 退出程序，返回错误码1

# 转换PDF第一页为base64图片 
print("正在处理PDF文件")
base64_image = pdf_first_page_to_base64(pdf_path)

# 检查图片转换是否成功
if base64_image is None:
    print("PDF转换失败，程序退出")
    exit(1)  # 退出程序，返回错误码1


# 初始化变量用于存储模型输出
reasoning_content = ""  # 定义完整思考过程（部分模型支持，如qvq-max）
answer_content = ""     # 定义完整回复内容
is_answering = False    # 判断是否结束思考过程并开始回复（用于控制输出格式）
enable_thinking = True  # 是否开启思考过程（仅部分模型支持）

print("\n" + "=" * 50)
print("开始调用 Qwen-VL 模型...")
print("=" * 50 + "\n")

# 创建聊天完成请求，调用大模型API
completion = client.chat.completions.create(
    model="qwen3-vl-plus",  # 可选: qwen3-vl-plus, qwen3-vl-flash, qvq-max
    # 消息列表，包含用户输入的内容
    messages=[
        {
            "role": "user",  # 角色：用户
            "content": [  # 内容可以是多模态的，这里包含图片和文本
                {
                    "type": "image_url",  # 内容类型：图片URL
                    "image_url": {
                        # 使用base64编码的图片，格式：data:image/png;base64,{base64字符串}
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
                {
                    "type": "text",  # 内容类型：文本
                    "text": "请详细提取并描述这个PDF第一页的全部内容，包括标题、正文、图表等所有信息。"
                },
            ],
        },
    ],
    stream=True,  # 使用流式输出，可以实时接收模型生成的响应
)


# 遍历流式响应的每个数据块
for chunk in completion:
    # 如果chunk.choices为空，说明这是最后一个数据块，包含usage信息
    if not chunk.choices:
        print("\n" + "=" * 20 + "Token使用情况" + "=" * 20)
        print(chunk.usage)  # 打印token使用统计
    else:
        # 获取当前数据块的增量内容
        delta = chunk.choices[0].delta
        
        # 处理思考过程（如果开启了思考模式）
        # 部分模型（如qvq-max）会先输出推理过程，再输出最终答案
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            # 实时打印思考内容，不换行
            print(delta.reasoning_content, end='', flush=True)
            # 将思考内容累加到变量中
            reasoning_content += delta.reasoning_content
        else:
            # 处理正式回复内容
            # 如果之前没有输出过回复标题，现在输出标题
            if delta.content and delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True  # 标记已经开始输出回复
            # 打印回复内容，实时显示
            if delta.content:
                print(delta.content, end='', flush=True)
                # 将回复内容累加到变量中
                answer_content += delta.content

print("处理完成！")

def save_result_to_file(content, filename):
    """将结果保存到文件"""
    try:
        # 以写入模式打开文件，编码为utf-8，支持中文
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)  # 写入内容
        print(f"结果已保存到: {filename}")
    except Exception as e:
        print(f"保存文件失败: {e}")

# 如果成功获取了回复内容，则保存到文本文件
if answer_content:
    save_result_to_file(answer_content, "作业2提取的内容.txt")