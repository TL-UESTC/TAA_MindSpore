import mindspore.nn as nn
import numpy as np
from .layers import EmbeddingLayer, MultiLayerPerceptron
import mindspore.ops as ops


class OMoEModel(nn.Cell):
    """
    A pytorch implementation of one-gate MoE Model.

    Reference:
        Jacobs, Robert A., et al. "Adaptive mixtures of local experts." Neural computation 3.1 (1991): 79-87.
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = nn.Dense(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.expert_num = expert_num

        self.expert = nn.CellList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        self.tower = nn.CellList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.gate = nn.SequentialCell(nn.Dense(self.embed_output_dim, expert_num), nn.Softmax(axis=1))

    def construct(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = ops.concat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        gate_value = self.gate(emb).unsqueeze(1)
        fea = ops.concat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], axis = 1)
        fea = ops.bmm(gate_value, fea).squeeze(1)
        
        results = [ops.sigmoid(self.tower[i](fea).squeeze(1)) for i in range(self.task_num)]
        # print(results)
        return results,fea