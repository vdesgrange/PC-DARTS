from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)

AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


PC_DARTS_cifar = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

PC_DARTS_cifar2 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])

cifar_ep1 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])

PCDARTS = PC_DARTS_cifar

GEN_0  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_1  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_2  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_3  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_4  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_5  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_6  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_7  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_8  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_9  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_10 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_11 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_12 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_13 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_14 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_15 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
GEN_16 = Genotype(normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('dil_conv_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_17 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_18 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 4), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_19 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
GEN_20 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_21 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_22 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4), ('sep_conv_5x5', 0)], reduce_concat=[2, 3, 4, 5])
GEN_23 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_24 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4), ('sep_conv_5x5', 0)], reduce_concat=[2, 3, 4, 5])
GEN_25 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_26 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_27 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_28 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_29 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_30 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_31 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_32 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_33 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_34 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_35 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_36 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
GEN_37 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_38 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_39 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_40 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_41 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_42 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_43 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_44 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_45 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_46 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_47 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_48 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
GEN_49 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
