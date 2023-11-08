from mindspore import nn,Tensor
from mindspore.common.initializer import XavierUniform
import numpy as np
from mindspore import Tensor
from mindspore.common.initializer import initializer
import mindspore.ops.operations as P

class EmbeddingLayer(nn.Cell):
    def __init__(self,field_dims,embed_dim):
        super(EmbeddingLayer,self).__init__()
        self.embedding = nn.Embedding(sum(field_dims),embed_dim,embedding_table = 'xavier_uniform')
        self.offsets = Tensor(np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long))
        # initializer  = XavierUniform(gain = 1.0)
        # self.embedding.embedding_table.set_data(Tensor(initializer(self.embedding.embedding_table.data)))
        # initializer_obj = initializer('XavierUniform')
        # weight = Tensor(initializer(self.embedding.embedding_table.data))
        # self.embedding.embedding_table.set_data(weight)

    def construct(self,x):
        x = (x + self.offsets.unsqueeze(0))
        return self.embedding(x)
    
class MultiLayerPerceptron(nn.Cell):
    def __init__(self,input_dim, embed_dims, dropout, output_layer=True):
        super(MultiLayerPerceptron,self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Dense(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Dense(input_dim,1))
        self.mlp = nn.SequentialCell(layers)
    
    def construct(self,x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)



