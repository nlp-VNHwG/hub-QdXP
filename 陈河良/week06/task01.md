D:\AILearning\Miniconda\envs\py312\python.exe D:\AILearning\八斗学院大模型课程\第6周：RAG工程化实现\Week06\04_ES测试.py 
--- 正在测试 Elasticsearch 连接 ---
连接成功！
{
  "name": "DESKTOP-TMTFMAP",
  "cluster_name": "elasticsearch",
  "cluster_uuid": "ZwSVZatXQmuOILxBYnMheg",
  "version": {
    "number": "9.3.1",
    "build_flavor": "default",
    "build_type": "zip",
    "build_hash": "0dd66e52ba3aa076cf498264e46339dbb71f0269",
    "build_date": "2026-02-23T23:37:38.684779921Z",
    "build_snapshot": false,
    "lucene_version": "10.3.2",
    "minimum_wire_compatibility_version": "8.19.0",
    "minimum_index_compatibility_version": "8.0.0"
  },
  "tagline": "You Know, for Search"
}

==================================================

--- 正在测试常见的 Elasticsearch 内置分词器 ---

使用分词器：standard
原始文本: 'Hello, world! This is a test.'
分词结果: ['hello', 'world', 'this', 'is', 'a', 'test']

使用分词器：simple
原始文本: 'Hello, world! This is a test.'
分词结果: ['hello', 'world', 'this', 'is', 'a', 'test']

使用分词器：whitespace
原始文本: 'Hello, world! This is a test.'
分词结果: ['Hello,', 'world!', 'This', 'is', 'a', 'test.']

使用分词器：english
原始文本: 'Hello, world! This is a test.'
分词结果: ['hello', 'world', 'test']

==================================================

--- 正在测试 IK 分词器 ---

使用 IK 分词器：ik_smart
原始文本: '我在使用Elasticsearch，这是我的测试。'
分词结果: ['我', '在', '使用', 'elasticsearch', '这是', '我', '的', '测试']

使用 IK 分词器：ik_max_word
原始文本: '我在使用Elasticsearch，这是我的测试。'
分词结果: ['我', '在', '使用', 'elasticsearch', '这是', '我', '的', '测试']

Process finished with exit code 0
