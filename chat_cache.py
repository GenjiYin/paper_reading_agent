"""
对话存储, 数据储存的格式为: 
{
    "system": "", 
    "user": "", 
    "assistant": "", 
    "history": [
        {
            "user": "", 
            "assistant": ""
        }, 
        {
            "user": "", 
            "assistant": ""
        }, 
        ...
    ]
}
"""

import os

class Dataset:
    def __init__(self, data_folder_path='./data'):
        self.data_folder_path = data_folder_path
        json_path_list = os.listdir(data_folder_path)
        if len(json_path_list) == 0:
            self.name_id = 1
        else:
            name_id_list = [*map(lambda x: int(x.split('.')[0]), json_path_list)]
            name_id_list.sort()
            self.name_id = name_id_list[-1]
    
    def parser(self, history):
        """
        将数据解析成存储格式
        """
        pass
    
    def save(self, data):
        # 获取最新的数据json文件的路径
        self.data_path = self.data_folder_path + '/' + str(self.name_id) + '.json'
        if os.path.getsize(self.data_path) / 1024 >= 1:
            # 文件大小大于等于1M时, 另新建一个文件来储存
            self.name_id += 1