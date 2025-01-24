import os
import numpy as np
import pandas as pd

class tools:
    def __init__(self, name, model=None):
        # 新建一个文件用来存放文本和嵌入矩阵
        self.data_path = os.path.join(os.getcwd(), 'data_save', name)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        # else:
        #     self.load(self.data_path)
        
        self.model = model
    
    def save(self, string_list, embedding_array):
        # 保存嵌入矩阵
        np.save(os.path.join(self.data_path, 'embedding.npy'), embedding_array)
        self.embedding = embedding_array

        # 解析文本数据并保存到csv文件
        pd.DataFrame(
            {
                "text": string_list
            }
        ).to_csv(os.path.join(self.data_path, 'text.csv'), index=False)
        self.string_list = string_list
    

    def load(self, path):
        self.embedding = np.load(os.path.join(path, 'embedding.npy'))
        self.string_list = pd.read_csv(os.path.join(path, 'text.csv'))['text'].to_list()
    
    def query(self, question, topk:int=2):
        if self.model is None:
            raise TypeError('请传入嵌入模型')
        
        embedding = self.model.encode([question])
        distance = np.sum((embedding - self.embedding)**2, axis=-1)
        max_index = distance.argsort()[:topk]
        return [self.string_list[i] for i in max_index]

class RAG:
    def __init__(self):
        pass



if __name__ == '__main__':
    t = tools(id='13')
    # print(os.path.join(os.getcwd(), '我的s'))