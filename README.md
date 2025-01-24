# paper_reading_agent
## 项目简介
这是一款论文阅读助手, 采用RAG架构构建的智能体, 输入提示词, 智能体将提示词与论文内容进行匹配, 将最优匹配结果当作资料与提示词一并传入大模型中, 让大模型根据资料解答你的疑问, 大大降低了大模型的幻觉。

需要安装的库如下: 
transformers==4.47.1
lxml==5.3.0
numpy==2.0.1
pandas==2.2.3
langchain==0.3.10
langchain_text_splitters==0.3.2
sentence_transforers==3.0.1
gradio==5.11.0
