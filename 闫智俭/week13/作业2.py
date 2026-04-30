下载Qwen3-0.6B模型
# 安装ModelScope
pip install modelscope

# 下载Qwen3-0.6B模型
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen3-0.6B', cache_dir='~/.cache/modelscope')

运行01_Qwen3.ipynb文件
# 安装Jupyter Notebook
pip install notebook

# 启动Jupyter Notebook
jupyter notebook
