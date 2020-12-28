# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
from albert_zh import AlbertConfig,AlbertTokenizer,AlbertModel
import os
basedir = os.path.dirname(__file__)
print(basedir)

class Config(object):

    """配置参数"""
    def __init__(self):

        self.save_path = basedir+'/models/albert.ckpt'        # 模型训练结果
        self.class_list = [x.strip() for x in open(basedir+'/models/class.txt').readlines()]  # 类别名单
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.num_classes = len(self.class_list)                         # 类别数
        self.bert_path = './albert_tiny'
        self.tokenizer = AlbertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 312
        self.pad_size = 100                                              # 每句话处理成的长度(短填长切)
        # self.learning_rate = 1e-5                                       # 学习率
        # self.adam_epsilon = 1e-6                                       # 学习率
        self.PAD, self.CLS = '[PAD]', '[CLS]'

config = Config()

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        model = AlbertModel.from_pretrained(config.bert_path)
        self.bert = model
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.m = nn.Softmax()

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask)
        out = self.fc(pooled)
        sorce = self.m(out)
        return sorce


class PredictModel():
    def __init__(self):
        model = Model().to(config.device)
        model.load_state_dict(torch.load(config.save_path,map_location=torch.device('cpu')))
        model.eval()
        self.config = config
        self.model = model


    def __encodeContext(self,s):
        token = config.tokenizer.tokenize(s)
        token = [self.config.CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)
        pad_size = self.config.pad_size
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents =(torch.LongTensor([token_ids]).to(config.device),
                         torch.LongTensor([seq_len]).to(config.device),
                         torch.LongTensor([mask]).to(config.device))
        return contents

    def __evaluateTest(self, data_iter):
        predict_all = []
        scores = []
        self.model.eval()
        with torch.no_grad():
            for texts in data_iter:
                outputs = self.model(texts)
                outputs = outputs.squeeze(0).data.cpu().numpy().tolist()
                scores.append(outputs)
                rs = outputs
                rs_temp = rs.copy()
                rs_temp.sort()
                l = len(rs_temp)
                rs_temp = rs_temp[l - 10:l]
                rs_temp.reverse()
                predcArr = []
                for v in rs_temp:
                    if v in rs:
                        i = rs.index(v)
                        if i not in predcArr:
                            predcArr.append(i)
                predict_all.append(predcArr)
        return predict_all,scores
    ###################################################33333

    def dis(self,x,y):
        return np.dot(x,y) /(np.linalg.norm(x) * np.linalg.norm(y))

    def predict(self,testArr):
        dataIter = []
        for s in testArr:
            encode = self.__encodeContext(s)
            dataIter.append(encode)
        predcArr = self.__evaluateTest(dataIter)
        return predcArr

    def getEncoder(self,testArr):
        dataIter = []
        for s in testArr:
            encode = self.__encodeContext(s)
            dataIter.append(encode)

        return dataIter

