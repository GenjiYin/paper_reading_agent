# paper_reading_agent
## 项目简介
这是一款论文阅读助手，采用 RAG 架构构建的智能体。输入提示词后，智能体将提示词与论文内容进行匹配，并将最优匹配结果作为资料与提示词一并传入大模型中。通过这种方式，大模型可以根据提供的资料更准确地解答你的疑问，从而大大降低大模型的幻觉问题。

### 环境依赖
以下是项目运行所需的 Python 库及版本：
```
transformers==4.47.1
lxml==5.3.0
numpy==2.0.1
pandas==2.2.3
langchain==0.3.10
langchain_text_splitters==0.3.2
sentence_transforers==3.0.1
gradio==5.11.0
```
- **Python 版本要求**：python>=3.10
- **支持私有大模型**：如果使用自己的私有大模型，请确保使用 transformers 库进行封装。
- **开源大模型资源**：你可以在魔搭社区搜索适合的开源大模型，以满足项目需求: https://www.modelscope.cn/models

### 模型选择建议
1. **语言大模型**：优先选择带有“instruct”字样的指令模型，便于后续智能体任务的开发;
2. **嵌入模型**: 推荐使用m3e嵌入模型: https://modelscope.cn/models/AI-ModelScope/m3e-large;
3. **翻译模型**: 推荐使用opus-mt-zh-en项目: https://www.modelscope.cn/models/moxying/opus-mt-zh-en.

请根据自己的计算机配置合理选择模型，以确保项目运行的流畅性。

## 效果展示
<div style="display: flex; justify-content: space-between;", align=center>
    <img src="/figure/2.jpg" alt="Image 1" style="width: 45%;"/>
    <img src="/figure/1.jpg" alt="Image 2" style="width: 45%;"/>
</div>

## 使用手册
1. 将pdf放入static文件夹下;
2. 打开gradio_UI.py文件, 重点关注14行到24行代码:
<div align=center>
<img src="/figure/3.png" />
</div>
3. 运行gradio_UI.py文件即可出发对话网页. 

## 项目原理

<div align=center>
<img src="/figure/4.png" />
</div>
我们将提示词进行向量编码化
