import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)  # Note: this MaxPool operation wasn't mention in the PC-Darts paper.
    self.k = 4  # Proportion of selected channels is 1/k
    for primitive in PRIMITIVES:  # There are 8 operations in PRIMITIVES for PC-Darts.
      op = OPS[primitive](C //self.k, stride, False)  # Get operation from name. Nb of channels: floor division C by k
      if 'pool' in primitive:  # Add a batch normalization to the pooling operation
        op = nn.Sequential(op, nn.BatchNorm2d(C //self.k, affine=False))
      self._ops.append(op)  # Put these ops in the pre-defined module list


  def forward(self, x, weights):
    # channel proportion k=4
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//self.k, :, :]
    xtemp2 = x[ : ,  dim_2//self.k:, :, :]
    temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
    # reduction cell needs pooling before concat
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
    ans = channel_shuffle(ans,self.k)
    # ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
    # except channe shuffle, channel shift also works
    return ans


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction
    # The structure of input nodes is fixed and does not participate in the search

    # Decide the structure of the first input nodes
    if reduction_prev:  # If the previous cell is a reduction
        # The operation feature map, with C_prev_prev input channels, will be down-sampled
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:  # If not, we simply connect
        # The first input_nodes is the output of cell k-2, the number of output channels of cell k-2 is C_prev_prev,
        # so the number of input channels for operation here is C_prev_prev
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

    # The structure of the second input nodes.
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)  # The second input_nodes is the output of cell k-1
    self._steps = steps  # steps = 4. The connection status of 4 nodes in each cell is to be determined
    self._multiplier = multiplier

    self._ops = nn.ModuleList()  # Build the module's list of operations
    self._bns = nn.ModuleList()

    for i in range(self._steps):  # Go through 4 intermediate nodes to build a mixed operation
      for j in range(2+i):   # For the i-th node, it has j predecessor nodes
        # (the input of each node is composed of the output of the first two cells
        # and the node in front of the current cell)
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)  # op is to build a mix between two nodes
        self._ops.append(op)  # Add the mixed operation of all edges to ops. Len of the list is 2+3+4+5=14[[],[],...,[]]

  # Cell in the calculation process, automatically called during forward propagation
  def forward(self, s0, s1, weights, weights2):
    """
    Implement the equation (1) from Darts paper
    x_j = \sum_{i<j} o^{(i,j)}(x^{i})
    :param s0: Input node from C_prev_prev
    :param s1: Input node from C_prev
    :param weights: Used by operation
    :return:
    """
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]  # Predecessor node of current node
    offset = 0
    # Loop over all 4 intermediates nodes
    for i in range(self._steps): # i and j are reversed in comparison to the math equation in pc-darts paper
        # s is the output of the current node i, find the operation corresponding to i in ops,
        # and then do the corresponding operation on all the predecessor nodes of i (called the forward of MixedOp),
        # and then add the results

        # Difference with darts : we use an additional weights2 attribute in front of the operation.
      s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)  # We handle a 1D (offset + j) array instead of a 2D (j, i) array.
      states.append(s)  # Use the output of the current node i as the input of the next node.
    # states is [s0,s1,b1,b2,b3,b4] b1,b2,b3,b4 are the output of the four intermediate outputs

    # Concat the intermediate output as the output of the current cell
    # dim=1 refers to the channel dimension concat, so the number of output channels becomes 4 times the original
    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      s0, s1 = s1, cell(s0, s1, weights,weights2)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.betas_normal = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self.betas_reduce = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
      self.betas_normal,
      self.betas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights,weights2):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
          W[j,:]=W[j,:]*W2[j]
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        
        #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene
    n = 3
    start = 2
    weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
    weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
    for i in range(self._steps-1):
      end = start + n
      tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
      tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
      start = end
      n += 1
      weightsr2 = torch.cat([weightsr2,tw2],dim=0)
      weightsn2 = torch.cat([weightsn2,tn2],dim=0)
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),weightsn2.data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),weightsr2.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

