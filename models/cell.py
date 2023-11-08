from mindspore import ops,nn
import mindspore.numpy as mnp
from mindspore import jit

class GenWithLossCell(nn.Cell):
    def __init__(self,tgt_model,dis_model0,dis_model1,criterion):
        super(GenWithLossCell,self).__init__()
        self.tgt_model = tgt_model
        self.dis_model0 = dis_model0
        self.dis_model1 = dis_model1
        self.criterion = criterion

    @jit
    def construct(self, t_categorical_fields,t_numerical_fields,t_labels,task_num,learning_e,learning_e2,learning_c,learning_p):

        tgt_y, tgt_feat = self.tgt_model(t_categorical_fields, t_numerical_fields)
        if not isinstance(tgt_feat, list):
            tgt_feat = [tgt_feat]*task_num
        criticG_fake = self.dis_model0(tgt_feat[0])
        criticG_fake = ops.reduce_mean(criticG_fake)
        G_cost = (-criticG_fake)*learning_e

        criticG_fake1 = self.dis_model1(tgt_feat[1])
        criticG_fake1 = ops.reduce_mean(criticG_fake1)
        G_cost += (-criticG_fake1)*learning_e2

        loss_list = [self.criterion(tgt_y[i], t_labels[:, i].float()) for i in range(t_labels.shape[1])]
        loss = 0

        loss += loss_list[0]*learning_c
        loss += loss_list[1]*learning_p
        loss /= len(loss_list)

        loss += G_cost

        return loss
    
class DisWithLossCell0(nn.Cell):
    def __init__(self,dis_model0):
        super(DisWithLossCell0,self).__init__()
        # self.tgt_model = tgt_model
        self.dis_model0 = dis_model0
        # self.dis_model1 = dis_model1
        self.grad_op = ops.GradOperation()
        self.uniform = ops.UniformReal()

    def calc_gradient_penalty(self,netD,real_data, fake_data):
        alpha = self.uniform((real_data.shape[0],1))
        alpha = ops.BroadcastTo(shape = real_data.shape)(alpha)
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data))
        gradient_function = self.grad_op(netD)
        gradients = gradient_function(interpolates)
        gradient_penalty = ops.reduce_mean(((mnp.norm(gradients, 2, axis=1) - 1) ** 2)) * 10

        return gradient_penalty

    @jit 
    def construct(self,src_feat,tgt_feat):

        criticD_real0 = self.dis_model0(src_feat[0])
        criticD_real0 = ops.reduce_mean(criticD_real0)

        criticD_fake0 = self.dis_model0(tgt_feat[0])
        criticD_fake0 = ops.reduce_mean(criticD_fake0)

        gradient_penalty0 = self.calc_gradient_penalty(self.dis_model0, src_feat[0], tgt_feat[0])

        D0_cost = criticD_fake0 - criticD_real0 + gradient_penalty0

        return D0_cost

class DisWithLossCell1(nn.Cell):
    def __init__(self,dis_model1):
        super(DisWithLossCell1,self).__init__()
        # self.tgt_model = tgt_model
        self.dis_model1 = dis_model1
        self.grad_op = ops.GradOperation()
        self.uniform = ops.UniformReal()

    def calc_gradient_penalty(self,netD,real_data, fake_data):
        alpha = self.uniform((real_data.shape[0],1))
        alpha = ops.BroadcastTo(shape = real_data.shape)(alpha)
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data))
        gradient_function = self.grad_op(netD)
        gradients = gradient_function(interpolates)
        gradient_penalty = ops.reduce_mean(((mnp.norm(gradients, 2, axis=1) - 1) ** 2)) * 10

        return gradient_penalty
        
    @jit
    def construct(self,src_feat,tgt_feat):
        # _, src_feat1 = self.tgt_model(s_categorical_fields, s_numerical_fields)
        # if not isinstance(src_feat1, list):
        #     src_feat1 = [src_feat1]*task_num

        # src_feat = src_feat1

        criticD_real1 = self.dis_model1(src_feat[1])
        criticD_real1 = ops.reduce_mean(criticD_real1)

        # _, tgt_feat = self.tgt_model(t_categorical_fields, t_numerical_fields)
        # if not isinstance(tgt_feat, list):
        #     tgt_feat = [tgt_feat]*task_num
        
        criticD_fake1 = self.dis_model1(tgt_feat[1])
        criticD_fake1 = ops.reduce_mean(criticD_fake1)

        gradient_penalty1 = self.calc_gradient_penalty(self.dis_model1, src_feat[1], tgt_feat[1])
        D1_cost = criticD_fake1 - criticD_real1 + gradient_penalty1

        return D1_cost




        








