# paper_reading_agent
## 项目简介
这是一款论文阅读助手, 采用RAG架构构建的智能体, 输入提示词, 智能体将提示词与论文内容进行匹配, 将最优匹配结果当作资料与提示词一并传入大模型中, 让大模型根据资料解答你的疑问, 大大降低了大模型的幻觉。

需要安装的库如下: 
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
python>=3.10, 模型可在魔搭社区搜索
https://www.modelscope.cn/models
请根据自己的计算机配置, 合理选择模型(优先搜索指令模型: 带有instruct字样的模型, 便于其他智能体任务的开发工作)。
关于嵌入模型, 大家可以选择m3e (https://modelscope.cn/models/AI-ModelScope/m3e-large).
关于翻译模型, 大家可以使用opus-mt-zh-en (https://www.modelscope.cn/models/moxying/opus-mt-zh-en).
## 效果展示
<div style="display: flex; justify-content: space-between;">
    <img src="/figure/2.jpg" alt="Image 1" style="width: 45%;"/>
    <img src="/figure/1.jpg" alt="Image 2" style="width: 45%;"/>
</div>

## 使用手册
1. 将pdf放入static文件夹下;
2. 打开gradio_UI.py文件, 重点关注14行到24行代码:
![](/figure/3.png)
3. 运行gradio_UI.py文件即可出发对话网页. 

## 项目原理
