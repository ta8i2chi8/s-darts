from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    # 'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1),
            ('skip_connect', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1)],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 1), ('sep_conv_7x7', 0), ('max_pool_3x3', 1), ('sep_conv_7x7', 0), ('avg_pool_3x3', 1),
            ('sep_conv_5x5', 0), ('skip_connect', 3), ('avg_pool_3x3', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 1)],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 1)],
    normal_concat=[4, 5, 6],
    reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_7x7', 2), ('sep_conv_7x7', 0),
            ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('conv_7x1_1x7', 0), ('sep_conv_3x3', 5)],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5]
)
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5]
)

# by morimoto
# noneあり
DARTS_o2 = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1)],
    reduce_concat=range(2, 6)
)
DARTS_o1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1)],
    reduce_concat=range(2, 6)
)
DARTS_o2_s1 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6)
)
DARTS_o1_s1 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 1), ('skip_connect', 2)],
    reduce_concat=range(2, 6)
)
DARTS_o2_s3 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6)
)
DARTS_o1_s3 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1)],
    reduce_concat=range(2, 6)
)
DARTS_o2_s4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0)],
    reduce_concat=range(2, 6)
)
DARTS_o1_s4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=range(2, 6)
)

# noneなし
DARTS_o1_s1_without_none = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 3), ('skip_connect', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 3)], 
    normal_concat=range(2, 6), 
    reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0)], 
    reduce_concat=range(2, 6)
)
DARTS_o1_s2_without_none = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('skip_connect', 2), ('dil_conv_5x5', 4), ('sep_conv_3x3', 2)], 
    normal_concat=range(2, 6), 
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_3x3', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], 
    reduce_concat=range(2, 6)
)
DARTS_o1_s3_without_none = Genotype(
    normal=[('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], 
    normal_concat=range(2, 6), 
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 4)], 
    reduce_concat=range(2, 6)
)
DARTS_o1_s4_without_none = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], 
    normal_concat=range(2, 6), 
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], 
    reduce_concat=range(2, 6)
)

# s-darts
SDARTS_o1_s1 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 4), ('skip_connect', 0)], 
    normal_concat=range(2, 6), 
    reduce=[('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 3)], 
    reduce_concat=range(2, 6)
)
SDARTS_o1_s2 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('dil_conv_5x5', 4)], 
    normal_concat=range(2, 6), 
    reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 4)], 
    reduce_concat=range(2, 6)
)
SDARTS_o1_s3 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0)], 
    normal_concat=range(2, 6), 
    reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 4)], 
    reduce_concat=range(2, 6)
)
SDARTS_o1_s4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 1)], 
    normal_concat=range(2, 6), 
    reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('skip_connect', 3)], 
    reduce_concat=range(2, 6)
)

# 4/23 epoch16
sample = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_5x5', 4), ('sep_conv_5x5', 0)], 
    normal_concat=range(2, 6), 
    reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 4)], 
    reduce_concat=range(2, 6)
)

# αチューニング
SDARTS_s2_a0006 = Genotype(normal=[('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 1), ('skip_connect', 4)], reduce_concat=range(2, 6))
SDARTS_s2_a0009 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
SDARTS_s2_a0012 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
SDARTS_s2_a0015 = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_s2_a0002 = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 3), ('skip_connect', 3), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('skip_connect', 3), ('skip_connect', 1), ('skip_connect', 4), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
SDARTS_s2_a0003 = Genotype(normal=[('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('skip_connect', 3), ('skip_connect', 1), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
SDARTS_s2_a0004 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
SDARTS_s2_a0005 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))
SDARTS_s2_a0006 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_s2_a0007 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_s2_a0008 = Genotype(normal=[('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 3), ('dil_conv_3x3', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 4)], reduce_concat=range(2, 6))
SDARTS_s2_a0009 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_s2_a001  = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
SDARTS_s2_a0012 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 1)], reduce_concat=range(2, 6))
SDARTS_s2_a0015 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))
SDARTS_s2_a0017 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_5x5', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
SDARTS_s2_a002  = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 4), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
SDARTS_s2_a0021 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('skip_connect', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
SDARTS_s2_a0022 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('skip_connect', 1), ('dil_conv_3x3', 4), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
SDARTS_s2_a0023 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))
SDARTS_s2_a0024 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))
SDARTS_s2_a0025 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_s2_a0026 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
SDARTS_s2_a0027 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 4), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_s2_a0028 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
SDARTS_s2_a0029 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 3), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
SDARTS_s2_a003  = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))
SDARTS_s2_a004  = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))
SDARTS_s2_a005  = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
SDARTS_s2_a0020_lr0025 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 3), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))
SDARTS_s2_a0025_lr0025 = Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
SDARTS_s2_a0030_lr0025 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_s2_a0020_lr01 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 1), ('dil_conv_5x5', 2), ('skip_connect', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_s2_a0025_lr01 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 3), ('sep_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
SDARTS_s2_a0030_lr01 = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_5x5', 3), ('dil_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# use sparsemax and auxiliary skip-connect
SDARTS_asc_a0003_lr005 = Genotype(normal=[('skip_connect', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 4), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_asc_a0004_lr005 = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))
SDARTS_asc_a0004_lr01  = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_asc_a0004_lr02  = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))
SDARTS_asc_a0004_lr03  = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 4), ('skip_connect', 0)], reduce_concat=range(2, 6))
SDARTS_asc_a0004_lr04  = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('dil_conv_3x3', 1), ('skip_connect', 4), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_asc_a0004_lr05  = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 3), ('dil_conv_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_asc_a0005_lr005 = Genotype(normal=[('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
SDARTS_asc_a0006_lr005 = Genotype(normal=[('skip_connect', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 1), ('skip_connect', 4)], reduce_concat=range(2, 6))
SDARTS_asc_a0007_lr005 = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))
