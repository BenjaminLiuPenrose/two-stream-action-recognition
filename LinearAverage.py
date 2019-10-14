import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn
import math
import copy
from pdb import set_trace as st


class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)
        momentum = params[1].item()

        # memory = F.normalize(memory, p = 2, dim = 1)
        # memory = Normalize(2)(memory)

        # inner product
        # out = torch.mm(x.data, memory.t().cuda() )
        x = x.cuda()
        y = y.cuda()
        memory = memory.cuda()
        out = torch.mm(x.data, memory.t())
        out.div_(T) # batchSize * N

        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()

        # print("grad: {}; x: {}; mem :{}".format(gradOutput.shape, x.shape, memory.shape))

        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data, comment here for w not equals v
        # weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        # weight_pos.mul_(momentum)
        # weight_pos.add_(torch.mul(x.data, 1-momentum))
        # w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        # updated_weight = weight_pos.div(w_norm)
        # memory.index_copy_(0, y, updated_weight)

        return gradInput, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',torch.tensor([T, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        ### stop w = v process to make w stand alone
        # self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))
        self.register_parameter('memory', None)
        self.memory = nn.Parameter(F.normalize(torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv)) )
        # self.l2norm = Normalize(2)


    def forward(self, x, y):
        # print(self.memory.requires_grad)
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out

# =======================================================================================
# ================== 0801 workable version ==================================================
class LinearAverageWithWeights(nn.Module):
    def __init__(self, inputSize, outputSize, T = 0.07, momentum = 0.5, sample_duration = 1, n_samples_for_each_video = 1, concat = 1):
        super(LinearAverageWithWeights, self).__init__()
        stdv = 1. / math.sqrt(inputSize/3)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(42)
        # input low_dim, output ndata
        self.weights =  nn.Parameter(
                        F.normalize(torch.rand(outputSize * n_samples_for_each_video, inputSize * concat).mul_(2*stdv).add_(-stdv)) ,
                        requires_grad = True
                        ).cuda()
        self.memory =  nn.Parameter(
                        F.normalize(torch.rand(outputSize * n_samples_for_each_video, inputSize * concat).mul_(2*stdv).add_(-stdv)) ,
                        requires_grad = False
                        ).cuda()

        ### modify 0813
        self.vectorBank = nn.Parameter(
                        F.normalize(torch.rand(outputSize * sample_duration, inputSize).mul_(2*stdv).add_(-stdv)) ,
                        requires_grad = True
                        )
        self.register_buffer('memory2', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))
        # self.memory = F.normalize(self.memory_learnt).cuda()
        # self.l2norm = Normalize(2)
        self.params = nn.Parameter(torch.tensor([T, momentum]), requires_grad = False)
        self.n_samples_for_each_video = n_samples_for_each_video
        self.sample_duration = sample_duration

    def forward(self, x, y, y2 = None):
        T = self.params[0].item()
        momentum = self.params[1].item()

        # self.memory2 = self.memory.data
        # out = torch.mm(x, self.memory2.t())

        out = torch.mm(x, F.normalize(self.weights).t() )
        # out = torch.mm(x, self.weights.t() )

        out.div_(T)

        log_probabilities = nn.LogSoftmax(x)
        # NLLLoss(x, class) = -weights[class] * x[class]
        # -self.weights.index_select(0, y.data) * log_probabilities.index_select(-1, y.data).diag()

        with torch.no_grad():
            # st()
            weight_pos = self.memory.index_select(0, y.data.view(-1)) #.resize_as_(x)
            weight_pos.mul_(momentum)
            # weight_pos.add_(torch.mul(x.data, 1-momentum))
            weight_pos.add_(torch.mul(
                    F.normalize(
                        self.weights).index_select(0, y.data.view(-1)
                    ).data,
                    1-momentum)
                )
            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            # updated_weight = F.normalize(weight_pos) # TOODO
            self.memory.index_copy_(0, y.data.view(-1), updated_weight )
            # self.memory = nn.Parameter(self.weights, requires_grad = False)



            ### modify 0813
            if y2 is not None:
                print("updating weights...")
                vector_pos = self.vectorBank.index_select(0, y2.data.view(-1))
                vector_pos.mul_(momentum)
                vector_pos.add_(torch.mul(
                    x,
                    1 - momentum
                    )
                )
                v_norm = vector_pos.pow(2).sum(1, keepdim = True).pow(0.5)
                updated_vector = vector_pos.div(v_norm)
                # st()
                # self.vectorBank.index_copy_(0, y2.data.view(-1), updated_vector)
                self.vectorBank[y2.data.view(-1), :] = updated_vector

        # loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j]))) = -x[class] + log(\sum_j exp(x[j]))

        return out
    def parameters(self, recurse=True):
        for name, param in self.named_parameters():
            if name in ['weights']:
                yield param

# ========================================================================================
# ============================ For saimese, defined loss function, equivalent to original ========================
class LinearAverageWithoutWeights(nn.Module):
    def __init__(self, inputSize, outputSize, T = 0.07, momentum = 0.5):
        super(LinearAverageWithoutWeights, self).__init__()
        stdv = 1. / math.sqrt(inputSize/3)
        self.weights =  nn.Parameter(
                        F.normalize(torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv)) ,
                        requires_grad = True
                        )
        self.memory =  nn.Parameter(
                        F.normalize(torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv)) ,
                        requires_grad = False
                        )
        self.register_buffer('memory2', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))
        self.params = nn.Parameter(torch.tensor([T, momentum]), requires_grad = False)

    def forward(self, x, x2, y):
        T = self.params[0].item()
        momentum = self.params[1].item()

        # self.memory2 = self.memory.data
        # out = torch.mm(x, self.memory2.t())

        # out = torch.mm(x, F.normalize(self.weights).t() )
        # out = torch.mm(x, self.weights.t() )

        out = (x, x2)

        with torch.no_grad():
            weight_pos = self.memory.index_select(0, y.data.view(-1))
            weight_pos.mul_(momentum)
            # weight_pos.add_(torch.mul(F.normalize(x.data), 1-momentum))
            weight_pos.add_(torch.mul(x.data, 1-momentum))
            # weight_pos.add_(torch.mul(F.normalize( self.weights).index_select(0, y.data.view(-1)).data, 1-momentum))
            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            # updated_weight = F.normalize(weight_pos) # TOODO
            self.memory.index_copy_(0, y.data.view(-1), updated_weight )

        return out
