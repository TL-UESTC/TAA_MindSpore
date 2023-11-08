import mindspore.nn as nn
import numpy as np
from .layers import EmbeddingLayer, MultiLayerPerceptron
import mindspore.ops as ops

class AITMModel(nn.Cell):
    def  __init__(self,categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super(AITMModel,self).__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = nn.Dense(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims)+1)*embed_dim
        self.task_num = task_num
        self.hidden_dim = bottom_mlp_dims[-1]
        self.hidden_dim_sqrt = np.sqrt(self.hidden_dim)

        self.g = nn.CellList([nn.Dense(bottom_mlp_dims[-1], bottom_mlp_dims[-1]) for i in range(task_num - 1)])
        self.h1 = nn.Dense(bottom_mlp_dims[-1], bottom_mlp_dims[-1])
        self.h2 = nn.Dense(bottom_mlp_dims[-1], bottom_mlp_dims[-1])
        self.h3 = nn.Dense(bottom_mlp_dims[-1], bottom_mlp_dims[-1])
        
        # self.testlayer = MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False)
        
        self.bottom = nn.CellList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(task_num)])
        self.tower = nn.CellList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
    
    def construct(self,categorical_x,numerical_x):
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = ops.concat([categorical_emb, numerical_emb],axis=1).view(-1, self.embed_output_dim)
        fea = [self.bottom[i](emb) for i in range(self.task_num)]

        for i in range(1,self.task_num):
            p = self.g[i-1](fea[i-1]).unsqueeze(1)
            # p = P.ExpandDims()(p, 1)
            q = fea[i].unsqueeze(1)
            # q = P.ExpandDims()(q, 1)
            x = ops.concat([p,q],axis = 1)
            V = self.h1(x)
            K = self.h2(x)
            Q = self.h3(x)
            fea[i] = ops.sum(ops.softmax(ops.sum(K * Q, 2, True) / self.hidden_dim_sqrt, axis = 1) * V, 1)
            # fea[i] = ops.sum(ops.softmax(ops.sum(K * Q, 2, True) / np.sqrt(self.hidden_dim),axis = 1) * V, 1)
        
        result = [ops.sigmoid(self.tower[i](fea[i]).squeeze(1)) for i in range(self.task_num)]
        return result, fea