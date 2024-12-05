from openai import OpenAI
from chat_cache import Dataset
from langchain_community.tools.tavily_search import TavilySearchResults

deepseek_api = 'sk-7713acb9f24c40608015328e6a20af14'

    
class LLM:
    def __init__(self):
        self.client = OpenAI(api_key=deepseek_api, base_url="https://api.deepseek.com").chat.completions.create
        self.his = []
        self.dataset = Dataset()
    
    def __call__(self, text, system=None):
        """
        集成了多轮对话功能
        """
        if system is not None:
            self.his.append({"role": "system", "content": system})
        
        self.his.append({'role': 'user', 'content': text})
        response = self.client(
            model="deepseek-chat", 
            messages=self.his, 
            stream=False
        )
        self.his.append({"role": "assistant", "content": response.choices[0].message.content})
        
        # 数据保存
        data = self.dataset.parser(self.his)
        self.dataset.save(data)
        return response.choices[0].message.content
        