import mindspore.nn as nn
import numpy as np
from .layers import EmbeddingLayer, MultiLayerPerceptron
import mindspore.ops as ops


class SharedBottomModel(nn.Cell):
    """
    A pytorch implementation of Shared-Bottom Model.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = nn.Dense(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num

        self.bottom = MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False)
        self.tower = nn.CellList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])

    def construct(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = ops.concat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        fea = self.bottom(emb)

        results = [ops.sigmoid(self.tower[i](fea).squeeze(1)) for i in range(self.task_num)]
        return results,fea