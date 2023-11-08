import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal

class Discriminator(nn.Cell):
    def __init__(self, input_dims=256, hidden_dims=64, output_dims=1):
        super(Discriminator,self).__init__()

        # self.layer = nn.SequentialCell(
        #     nn.Dense(input_dims, hidden_dims)),
        #     nn.LeakyReLU(0.2)
        # )
        self.FC1 = nn.Dense(input_dims, hidden_dims,weight_init=Normal(0.0, 0.02), bias_init='zeros')
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.FC2 = nn.Dense(hidden_dims,16,weight_init=Normal(0.0, 0.02), bias_init='zeros')
        self.FC3 = nn.Dense(16,output_dims,weight_init=Normal(0.0, 0.02), bias_init='zeros')
        self.log = ops.Log()
        self.softmax = nn.Softmax(axis=1)

    def construct(self,input):
        out = self.FC1(input)
        out = self.leakyrelu(out)
        out = self.FC2(out)
        out = self.leakyrelu(out)
        out = self.FC3(out)
        out = self.softmax(out)
        out = self.log(out)

        return out
        

        