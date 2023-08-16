import torch
from torch import nn
from torch import optim
from torch.utils import data as Data
import numpy as np
import ipdb
import torchinfo


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        # Q, K, V 对应embedded: 1 * 4 * 6, mask 1 * 4 * 4
        # residual: 1 * 4 * 6, batch: 1
        residual, batch = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q)  # 将1 * 4 * 6 -> 1 * 4 * 6
        Q = Q.view(batch, -1, n_heads, d_k).transpose(1, 2)  # Q 最终变成 1 * 4 * 2 * 3 -> 1 * 2 * 4 * 3;
        # K, V 都是 [1, 2, 4, 3]
        K = self.W_K(input_K).view(batch, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch, -1, n_heads, d_v).transpose(1, 2)

        # attn_mask首先变为 [1, 1, 4, 4], 然后变成[1, 2, 4, 4]
        # print("attn_mask shape: ")
        # print(attn_mask.unsqueeze(1).shape)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # print(attn_mask.shape)
        # print("attn_mask shape end!")

        # Q, K, V: [1, 2, 4, 3], attn_mask: [1, 2, 4, 4]
        # 得到prob: [1, 2, 4, 3]
        prob = ScaledDotProductAttention()(Q, K, V, attn_mask)

        # prob: [1, 4, 2, 3]
        prob = prob.transpose(1, 2).contiguous()
        # prob: [1, 4, 6]
        prob = prob.view(batch, -1, n_heads * d_v).contiguous()
        # [1, 4, 6]
        output = self.fc(prob)
        # return [1, 4, 6]
        return self.layer_norm(residual + prob)


class ScaledDotProductAttention(nn.Module):
    def __int__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # input: Q, K, V: [1, 2, 4, 3], attn_mask: [1, 2, 4, 4]
        # scores 经处理后变为：[1, 2, 4, 4]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 用value, 填充attn_mask中, 与mask中值为1位置相对应的元素
        print('Q shape: ', Q.shape)
        print('K shape: ', K.shape)
        print('V shape: ', V.shape)
        print('socres shape: ', scores.shape)
        scores.masked_fill(attn_mask, -1e9)
        # softmax 最后一维做 softmax归一
        # 变为[1, 2, 4, 4]
        attn = nn.Softmax(dim=-1)(scores)
        # 输出 [1, 2, 4, 4] * [1, 2, 4, 3] = [1, 2, 4, 3]
        prob = torch.matmul(attn, V)
        return prob


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc = nn.Linear(d_model, len(target_vocab), bias=False)

    def forward(self, encoder_input, decoder_input):
        # 入 1*4, 出 1*4*6, 作用是将"我吃肉E"embedding, 并带上3词间的关注力信息
        encoder_output = self.encoder(encoder_input)

        # 入： [1, 4], [1, 4], [1, 4, 6]，作用是将"S I eat meat"编码带上3词之间的注意力，并且将
        # "我吃肉E + S I eat meat"的词间关注力带给"我吃肉E"
        # 出：decoder_output: [1, 4, 6]
        decoder_output = self.decoder(decoder_input, encoder_input, encoder_output)
        # 出：decoder_output: [1, 4, 5]
        decoder_logits = self.fc(decoder_output)


        return decoder_logits.view(-1, decoder_logits.size(-1))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.source_embedding = nn.Embedding(len(source_vocab), d_model)
        self.attention = MultiHeadAttention()

    def forward(self, encoder_input): #multi-heads-self-attention
        # encoder_input：1 * 4, embedded: 1 * 4 * 6
        embedded = self.source_embedding(encoder_input)
        # 得到的mask为：1 * 4 * 4
        mask = get_attn_pad_mask(encoder_input, encoder_input)
        # embedded: 1 * 4 * 6, mask 1 * 4 * 4
        # 最终得到的encoder_output: [1, 4, 6]
        encoder_output = self.attention(embedded, embedded, embedded, mask)
        return encoder_output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.target_embedding = nn.Embedding(len(target_vocab), d_model)
        self.attention = MultiHeadAttention()

    def forward(self, decoder_input, encoder_input, encoder_output):
        # 入： decoder_input: [1, 4], encoder_input: [1, 4], encoder_output: [1, 4, 6]

        # decoder_embedded: [1, 4, 6]
        decoder_embedded = self.target_embedding(decoder_input)  #

        # 得到的decoder_self_attn_mask：1 * 4 * 4
        # 最后一位为 1, 其余为0
        decoder_self_attn_mask = get_attn_pad_mask(decoder_input, decoder_input)  #
        # print("得到的decoder_self_attn_mask: ")
        # print(decoder_self_attn_mask.shape)
        # print("得到的decoder_self_attn_mask end!")
        # [1, 4, 4], 右上对角线为1，k=1; 其余为0
        decoder_subsequent_mask = get_attn_subsequent_mask(decoder_input)
        # 得到 padding mask and 掩码mask
        # [1, 4, 4]
        decoder_self_mask = torch.gt(decoder_self_attn_mask + decoder_subsequent_mask, 0)
        # decoder_embedded: [1, 4, 6],decoder_self_mask: [1, 4, 4], decoder_output: [1， 4， 6]
        # important: Masked-Multi-Head-Self-Attention
        decoder_output = self.attention(decoder_embedded, decoder_embedded, decoder_embedded, decoder_self_mask)


        # decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input, encoder_input)
        # decoder_output = self.attention(decoder_embedded, decoder_embedded, decoder_embedded, decoder_self_mask)

        # 第二层的 Attention, 首先进行decoder_input 的mask机制
        # decoder_input: q, [1, 4]; encoder_input: k, [1, 4]; decoder_encoder_attn_mask: [1, 4],
        # important: Multi-Head-Cross-Attention
        print('decoder_input shape: ', decoder_input.shape)
        print('encoder_input shape: ', encoder_input.shape)
        decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input, encoder_input) # 代表q和k;
        print('decoder_encoder_attn_mask shape: ', decoder_encoder_attn_mask.shape)
        print('decoder_encoder_attn_mask value: ', decoder_encoder_attn_mask)
        # print(decoder_encoder_attn_mask)
        # 输入均为1*4*6，Q表示"S I eat meat" K表示"我吃肉E" V表示 "我吃肉E"，整体表示"我吃肉E"，并带上和"我吃肉E"、"S I eat meat"的关注力矩阵
        # decoder_output: [1, 4, 6]
        decoder_output = self.attention(decoder_output, encoder_output, encoder_output, decoder_encoder_attn_mask)

        return decoder_output


def get_attn_pad_mask(seq_q, seq_k):
    # input encoder_input, encoder_input
    # compute q, k attention value
    batch, len_q = seq_q.size()  # 1 * 4, seq_q 用于升维，为了做attention，mask score矩阵用的
    batch, len_k = seq_k.size()  # 1 * 4
    # seq_k 等于 0 的坐标为true, 其它false, 并中间插入一维；# [batch_size, 1, len_k], True is masked
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 返回 1 * 1 * 4
    print('pad_attn_mask: ', pad_attn_mask)

    # 将 pad_attn_mask 扩展为: 1 * 4 * 4
    # [[[False, False, False, True]], [[False, False, False, True]],
    # [[False, False, False, True]], [[False, False, False, True]]]
    return pad_attn_mask.expand(batch, len_q, len_k)


def get_attn_subsequent_mask(seq):
    # seq: [1, 4]
    # attn_shape: [1, 4, 4]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # np.triu(a, k = 1)，得到主对角线向上平移一个距离的对角线
    # subsequent_mask: [1, 4, 4], 右上对角线为1
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask)
    # print("subsequent_mask")
    # print(subsequent_mask.shape)
    # print("subsequent_mask end!")

    return subsequent_mask


# params init
d_model = 6  # embedding size
d_ff = 12  # feedforward nerual network dimension
d_k = d_v = 3  # dimension of k(same as q) and v
n_heads = 2  # number of heads in multihead attention
p_drop = 0.1  # propability of dropout
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

source_vocab = {'E': 0, '我': 1, '吃': 2, '肉': 3}
target_vocab = {'E': 0, 'I': 1, 'eat': 2, 'meat': 3, 'S': 4}
print(len(target_vocab))

encoder_input = torch.LongTensor([[1, 2, 3, 0]]).to(device)  # 我 吃 肉 E, E代表结束词
decoder_input = torch.LongTensor([[4, 1, 2, 3]]).to(device)  # S I eat meat, S代表开始词，并右移一位，用于并行训练
target = torch.LongTensor([[1, 2, 3, 0]]).to(device)  # I eat meat E, 翻译目标

model = Transformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-1)

for epoch in range(500):
    output = model(encoder_input, decoder_input)  ## inference
    loss = criterion(output, target.view(-1))     ## 求解loss
    # print(output.shape)
    # print(target.shape)
    # print(target.view(-1).shape)
    print('Epoch: ', '%04d' % (epoch + 1), 'loss = ', '{:.6f}'.format(loss))

    optimizer.zero_grad()                         ## 梯度清零
    loss.backward()                               ## 反向传播求解梯度
    optimizer.step()                              ## 更新权重参数

# torch.save(model, 'model_name.pth')
# model = torch.load('model_name.pth')

print("------------模型参数---------------")
for parameters in model.parameters():
    print(parameters)


print("------------以下是具体参数---------------")
params = list(model.named_parameters())
for parameters in params:
    print(parameters)

print("------------模型参数end!---------------")

print('------------打印模型结构----------------')
print(model)

torchinfo.summary(model=model)
print('------------打印模型结构end!----------------')

# use model
target_len = len(target_vocab)
encoder_output = model.encoder(encoder_input)
decoder_input = torch.zeros(1, target_len).type_as(encoder_input.data)
# print('*********************')
# print(decoder_input)
# print(encoder_input)
# print('---------------------')

next_symbol = 4

for i in range(target_len):
    decoder_input[0][i] = next_symbol
    print('\n ***********我是分割线*************')
    print("i = %d" % i)
    print('decoder_input[0][', i,  ']:', decoder_input[0][i])
    print('decoder_input: ', decoder_input)
    decoder_output = model.decoder(decoder_input, encoder_input, encoder_output)

    # 去掉向量中维度为1的维度；
    logits = model.fc(decoder_output).squeeze(0)
    print('logits shape: ', logits.shape)
    print('logits: ', logits)
    # dim 是索引的维度，dim=0寻找每一列的最大值，dim=1寻找每一行的最大值
    prob = logits.max(dim=1, keepdim=False)[1]
    # 取每行logits 最大值对应的索引
    print('prob: ', prob)
    next_symbol = prob.data[i].item()
    print('next_symbol: ', next_symbol)
    print('下一个加入到decoder_input，用于mask的字符是: ', next_symbol)

    for k, v in target_vocab.items():
        if v == next_symbol:
            print('第', i, '轮', k)
            break

    if next_symbol == 0:
        break

print("the decoder process done!")





# ------------模型参数---------------
# Parameter containing:
# tensor([[-2.0614, -2.4971, -0.0720,  0.6516,  0.0433, -0.6987],
#         [-0.1586,  0.8119, -0.8583, -0.2807,  0.6333, -2.8893],
#         [-0.4948,  2.6407,  1.2675, -1.0057,  1.0647,  0.7283],
#         [-0.5721,  0.2308, -0.3830,  1.1738, -2.5100, -0.3893]],
#        requires_grad=True)
# Parameter containing:
# tensor([[ 0.4030,  0.2986,  0.6696, -0.3627,  0.3720, -0.1106],
#         [-0.9530, -0.1127,  0.2536, -0.1103,  0.1379, -0.5720],
#         [ 0.0276, -0.5391, -0.3254,  0.9012, -0.4352,  0.7696],
#         [-0.6708,  0.0888,  0.3680,  0.1582,  0.1778, -0.1789],
#         [ 0.3418, -0.3084, -0.6963,  0.0448,  0.0188,  0.7849],
#         [ 0.3089,  0.0163, -0.1392, -0.0465, -0.5179, -0.6807]],
#        requires_grad=True)
# Parameter containing:
# tensor([[-0.7717, -0.3048, -0.3232,  0.3434, -0.9865, -0.4759],
#         [-0.6396, -0.5969, -0.6129,  0.7422, -1.0582,  0.1792],
#         [-0.5007, -0.3219,  0.8389, -0.3577,  0.7668,  0.3944],
#         [-0.2418,  0.0491,  0.8047, -0.7218,  0.6491,  0.3054],
#         [ 0.2387,  0.0973,  0.2041,  0.7789, -0.7474,  0.1312],
#         [-0.5730, -0.4506, -0.2720,  0.1537, -0.8467,  0.1557]],
#        requires_grad=True)
# Parameter containing:
# tensor([[-0.2730, -0.5006, -0.5668,  0.3525, -0.3326,  0.3294],
#         [-0.9684,  0.2825,  0.7409, -0.9483,  0.2950, -0.9549],
#         [ 1.1825, -0.0040, -1.1101, -0.4960, -0.1245,  0.5283],
#         [-0.6177, -0.5764,  0.0855,  0.4617, -0.8376, -0.6069],
#         [ 0.1114, -0.1245,  0.0262,  0.1408,  0.6621,  0.5493],
#         [ 1.0650, -0.0352, -0.0533, -0.1710, -0.1689,  0.1182]],
#        requires_grad=True)
# Parameter containing:
# tensor([[ 0.2058, -0.1382, -0.3736,  0.1344,  0.2747, -0.2494],
#         [ 0.2372, -0.3845, -0.1098, -0.2310,  0.3457, -0.1735],
#         [ 0.1057, -0.3030, -0.2927, -0.3634, -0.1657, -0.2733],
#         [ 0.1213,  0.0603,  0.1959, -0.0537, -0.1996, -0.0214],
#         [ 0.0705, -0.3829, -0.1101,  0.3734, -0.2429, -0.3483],
#         [ 0.3730, -0.0424,  0.3880, -0.0777,  0.0526, -0.2360]],
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0083,  0.8059, -0.1127,  0.1680,  0.2405,  1.3399],
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0272,  0.7225, -0.5957,  0.3401,  0.3899, -0.2515],
#        requires_grad=True)
# Parameter containing:
# tensor([[-1.8143,  1.3149, -1.5722,  0.9693,  0.1788,  0.7243],
#         [-1.3294, -1.6204,  0.2983,  1.0502,  1.3442,  0.2204],
#         [-0.4315, -0.8338,  0.1530,  0.2638, -1.7035,  0.0598],
#         [ 0.3586,  0.4070,  1.2086, -1.2895,  0.0908,  0.4160],
#         [ 1.3007, -0.6714, -1.2184, -1.1673,  3.5792,  0.4212]],
#        requires_grad=True)
# Parameter containing:
# tensor([[-0.9411, -0.2042, -0.1424,  0.3211,  0.6084, -0.1584],
#         [ 0.7292, -0.5070, -0.5540, -1.0351, -0.0107, -0.2256],
#         [-0.8050, -0.5645,  0.3941,  0.1836,  1.2539, -0.6751],
#         [ 1.1878,  1.0107, -0.2106, -0.1894, -0.3117, -0.7057],
#         [-0.4978, -0.5110,  0.3244,  0.5910,  0.1375,  1.0711],
#         [ 0.3947, -0.9471, -0.0887,  0.3808,  0.3026,  0.8910]],
#        requires_grad=True)
# Parameter containing:
# tensor([[-0.4837,  0.0798,  0.5840, -0.4583,  0.1515, -0.7426],
#         [-0.4639, -0.6601, -0.0352,  0.7017,  0.9192,  0.6118],
#         [-0.6695,  0.9547,  1.2573,  0.0580,  0.5070, -1.6545],
#         [ 0.9373, -0.8814, -1.0067,  0.4215,  0.0401, -0.1309],
#         [-0.8003,  0.2385,  0.9274, -0.0348, -0.2675, -0.7149],
#         [ 0.0431,  1.2798,  0.3328, -1.1399,  0.1640, -0.3541]],
#        requires_grad=True)
# Parameter containing:
# tensor([[ 0.0838, -0.4046,  0.2303,  0.2415,  0.1065,  0.4170],
#         [-0.5541,  0.0827,  0.9875, -0.2319,  0.8339, -0.0917],
#         [-0.1666, -0.2950,  0.3635,  0.0826,  0.0372,  0.4122],
#         [-0.3913, -0.0887,  0.3263,  0.1408,  0.0314, -0.1499],
#         [ 0.0905,  0.0547, -0.3082,  0.4565, -0.3686, -0.5928],
#         [ 0.9430,  0.0273, -0.5687,  0.6824, -0.5681, -0.7916]],
#        requires_grad=True)
# Parameter containing:
# tensor([[ 0.1807,  0.2044,  0.1197,  0.0931,  0.1330, -0.1813],
#         [ 0.0948, -0.3784, -0.0107,  0.3713,  0.0727,  0.2682],
#         [-0.3939,  0.3274, -0.3927, -0.3262,  0.2878,  0.1366],
#         [ 0.2369,  0.2770,  0.1475, -0.2217,  0.3561, -0.3569],
#         [-0.4009, -0.0333,  0.2490,  0.2743, -0.1447, -0.1467],
#         [-0.0194, -0.3425, -0.0834, -0.0646, -0.1830,  0.3503]],
#        requires_grad=True)
# Parameter containing:
# tensor([1.2950, 1.5150, 2.4657, 2.4267, 2.5576, 0.4979], requires_grad=True)
# Parameter containing:
# tensor([ 0.3521, -0.5231,  0.2395,  0.0458,  0.5032, -0.7870],
#        requires_grad=True)
# Parameter containing:
# tensor([[-0.4191,  1.2961,  1.4981, -1.6469, -0.2372, -0.4616],
#         [ 0.5364, -0.6312, -1.6549, -1.4957,  1.6273, -0.8331],
#         [-1.1248, -0.4583, -0.2047,  1.8944,  1.5216, -0.7608],
#         [ 1.1685, -0.6388,  1.1252,  1.4231, -1.1651,  0.7420],
#         [ 0.6375,  1.5033, -1.5076,  0.0968, -0.1601, -0.0689]],
#        requires_grad=True)
# ------------以下是具体参数---------------
# ('encoder.source_embedding.weight', Parameter containing:
# tensor([[-2.0614, -2.4971, -0.0720,  0.6516,  0.0433, -0.6987],
#         [-0.1586,  0.8119, -0.8583, -0.2807,  0.6333, -2.8893],
#         [-0.4948,  2.6407,  1.2675, -1.0057,  1.0647,  0.7283],
#         [-0.5721,  0.2308, -0.3830,  1.1738, -2.5100, -0.3893]],
#        requires_grad=True))
# ('encoder.attention.W_Q.weight', Parameter containing:
# tensor([[ 0.4030,  0.2986,  0.6696, -0.3627,  0.3720, -0.1106],
#         [-0.9530, -0.1127,  0.2536, -0.1103,  0.1379, -0.5720],
#         [ 0.0276, -0.5391, -0.3254,  0.9012, -0.4352,  0.7696],
#         [-0.6708,  0.0888,  0.3680,  0.1582,  0.1778, -0.1789],
#         [ 0.3418, -0.3084, -0.6963,  0.0448,  0.0188,  0.7849],
#         [ 0.3089,  0.0163, -0.1392, -0.0465, -0.5179, -0.6807]],
#        requires_grad=True))
# ('encoder.attention.W_K.weight', Parameter containing:
# tensor([[-0.7717, -0.3048, -0.3232,  0.3434, -0.9865, -0.4759],
#         [-0.6396, -0.5969, -0.6129,  0.7422, -1.0582,  0.1792],
#         [-0.5007, -0.3219,  0.8389, -0.3577,  0.7668,  0.3944],
#         [-0.2418,  0.0491,  0.8047, -0.7218,  0.6491,  0.3054],
#         [ 0.2387,  0.0973,  0.2041,  0.7789, -0.7474,  0.1312],
#         [-0.5730, -0.4506, -0.2720,  0.1537, -0.8467,  0.1557]],
#        requires_grad=True))
# ('encoder.attention.W_V.weight', Parameter containing:
# tensor([[-0.2730, -0.5006, -0.5668,  0.3525, -0.3326,  0.3294],
#         [-0.9684,  0.2825,  0.7409, -0.9483,  0.2950, -0.9549],
#         [ 1.1825, -0.0040, -1.1101, -0.4960, -0.1245,  0.5283],
#         [-0.6177, -0.5764,  0.0855,  0.4617, -0.8376, -0.6069],
#         [ 0.1114, -0.1245,  0.0262,  0.1408,  0.6621,  0.5493],
#         [ 1.0650, -0.0352, -0.0533, -0.1710, -0.1689,  0.1182]],
#        requires_grad=True))
# ('encoder.attention.fc.weight', Parameter containing:
# tensor([[ 0.2058, -0.1382, -0.3736,  0.1344,  0.2747, -0.2494],
#         [ 0.2372, -0.3845, -0.1098, -0.2310,  0.3457, -0.1735],
#         [ 0.1057, -0.3030, -0.2927, -0.3634, -0.1657, -0.2733],
#         [ 0.1213,  0.0603,  0.1959, -0.0537, -0.1996, -0.0214],
#         [ 0.0705, -0.3829, -0.1101,  0.3734, -0.2429, -0.3483],
#         [ 0.3730, -0.0424,  0.3880, -0.0777,  0.0526, -0.2360]],
#        requires_grad=True))
# ('encoder.attention.layer_norm.weight', Parameter containing:
# tensor([-0.0083,  0.8059, -0.1127,  0.1680,  0.2405,  1.3399],
#        requires_grad=True))
# ('encoder.attention.layer_norm.bias', Parameter containing:
# tensor([-0.0272,  0.7225, -0.5957,  0.3401,  0.3899, -0.2515],
#        requires_grad=True))
# ('decoder.target_embedding.weight', Parameter containing:
# tensor([[-1.8143,  1.3149, -1.5722,  0.9693,  0.1788,  0.7243],
#         [-1.3294, -1.6204,  0.2983,  1.0502,  1.3442,  0.2204],
#         [-0.4315, -0.8338,  0.1530,  0.2638, -1.7035,  0.0598],
#         [ 0.3586,  0.4070,  1.2086, -1.2895,  0.0908,  0.4160],
#         [ 1.3007, -0.6714, -1.2184, -1.1673,  3.5792,  0.4212]],
#        requires_grad=True))
# ('decoder.attention.W_Q.weight', Parameter containing:
# tensor([[-0.9411, -0.2042, -0.1424,  0.3211,  0.6084, -0.1584],
#         [ 0.7292, -0.5070, -0.5540, -1.0351, -0.0107, -0.2256],
#         [-0.8050, -0.5645,  0.3941,  0.1836,  1.2539, -0.6751],
#         [ 1.1878,  1.0107, -0.2106, -0.1894, -0.3117, -0.7057],
#         [-0.4978, -0.5110,  0.3244,  0.5910,  0.1375,  1.0711],
#         [ 0.3947, -0.9471, -0.0887,  0.3808,  0.3026,  0.8910]],
#        requires_grad=True))
# ('decoder.attention.W_K.weight', Parameter containing:
# tensor([[-0.4837,  0.0798,  0.5840, -0.4583,  0.1515, -0.7426],
#         [-0.4639, -0.6601, -0.0352,  0.7017,  0.9192,  0.6118],
#         [-0.6695,  0.9547,  1.2573,  0.0580,  0.5070, -1.6545],
#         [ 0.9373, -0.8814, -1.0067,  0.4215,  0.0401, -0.1309],
#         [-0.8003,  0.2385,  0.9274, -0.0348, -0.2675, -0.7149],
#         [ 0.0431,  1.2798,  0.3328, -1.1399,  0.1640, -0.3541]],
#        requires_grad=True))
# ('decoder.attention.W_V.weight', Parameter containing:
# tensor([[ 0.0838, -0.4046,  0.2303,  0.2415,  0.1065,  0.4170],
#         [-0.5541,  0.0827,  0.9875, -0.2319,  0.8339, -0.0917],
#         [-0.1666, -0.2950,  0.3635,  0.0826,  0.0372,  0.4122],
#         [-0.3913, -0.0887,  0.3263,  0.1408,  0.0314, -0.1499],
#         [ 0.0905,  0.0547, -0.3082,  0.4565, -0.3686, -0.5928],
#         [ 0.9430,  0.0273, -0.5687,  0.6824, -0.5681, -0.7916]],
#        requires_grad=True))
# ('decoder.attention.fc.weight', Parameter containing:
# tensor([[ 0.1807,  0.2044,  0.1197,  0.0931,  0.1330, -0.1813],
#         [ 0.0948, -0.3784, -0.0107,  0.3713,  0.0727,  0.2682],
#         [-0.3939,  0.3274, -0.3927, -0.3262,  0.2878,  0.1366],
#         [ 0.2369,  0.2770,  0.1475, -0.2217,  0.3561, -0.3569],
#         [-0.4009, -0.0333,  0.2490,  0.2743, -0.1447, -0.1467],
#         [-0.0194, -0.3425, -0.0834, -0.0646, -0.1830,  0.3503]],
#        requires_grad=True))
# ('decoder.attention.layer_norm.weight', Parameter containing:
# tensor([1.2950, 1.5150, 2.4657, 2.4267, 2.5576, 0.4979], requires_grad=True))
# ('decoder.attention.layer_norm.bias', Parameter containing:
# tensor([ 0.3521, -0.5231,  0.2395,  0.0458,  0.5032, -0.7870],
#        requires_grad=True))
# ('fc.weight', Parameter containing:
# tensor([[-0.4191,  1.2961,  1.4981, -1.6469, -0.2372, -0.4616],
#         [ 0.5364, -0.6312, -1.6549, -1.4957,  1.6273, -0.8331],
#         [-1.1248, -0.4583, -0.2047,  1.8944,  1.5216, -0.7608],
#         [ 1.1685, -0.6388,  1.1252,  1.4231, -1.1651,  0.7420],
#         [ 0.6375,  1.5033, -1.5076,  0.0968, -0.1601, -0.0689]],
#        requires_grad=True))
# ------------模型参数end!---------------
# ------------打印模型结构----------------
# Transformer(
#   (encoder): Encoder(
#     (source_embedding): Embedding(4, 6)
#     (attention): MultiHeadAttention(
#       (W_Q): Linear(in_features=6, out_features=6, bias=False)
#       (W_K): Linear(in_features=6, out_features=6, bias=False)
#       (W_V): Linear(in_features=6, out_features=6, bias=False)
#       (fc): Linear(in_features=6, out_features=6, bias=False)
#       (layer_norm): LayerNorm((6,), eps=1e-05, elementwise_affine=True)
#     )
#   )
#   (decoder): Decoder(
#     (target_embedding): Embedding(5, 6)
#     (attention): MultiHeadAttention(
#       (W_Q): Linear(in_features=6, out_features=6, bias=False)
#       (W_K): Linear(in_features=6, out_features=6, bias=False)
#       (W_V): Linear(in_features=6, out_features=6, bias=False)
#       (fc): Linear(in_features=6, out_features=6, bias=False)
#       (layer_norm): LayerNorm((6,), eps=1e-05, elementwise_affine=True)
#     )
#   )
#   (fc): Linear(in_features=6, out_features=5, bias=False)
# )
# =================================================================
# Layer (type:depth-idx)                   Param #
# =================================================================
# Transformer                              --
# ├─Encoder: 1-1                           --
# │    └─Embedding: 2-1                    24
# │    └─MultiHeadAttention: 2-2           --
# │    │    └─Linear: 3-1                  36
# │    │    └─Linear: 3-2                  36
# │    │    └─Linear: 3-3                  36
# │    │    └─Linear: 3-4                  36
# │    │    └─LayerNorm: 3-5               12
# ├─Decoder: 1-2                           --
# │    └─Embedding: 2-3                    30
# │    └─MultiHeadAttention: 2-4           --
# │    │    └─Linear: 3-6                  36
# │    │    └─Linear: 3-7                  36
# │    │    └─Linear: 3-8                  36
# │    │    └─Linear: 3-9                  36
# │    │    └─LayerNorm: 3-10              12
# ├─Linear: 1-3                            30
# =================================================================
# Total params: 396
# Trainable params: 396
# Non-trainable params: 0
# =================================================================
# ------------打印模型结构end!----------------


# ***********我是分割线*************
# i = 0
# decoder_input[0][ 0 ]: tensor(4)
# decoder_input:  tensor([[4, 0, 0, 0, 0]])
# pad_attn_mask:  tensor([[[False,  True,  True,  True,  True]]])
# Q shape:  torch.Size([1, 2, 5, 3])
# K shape:  torch.Size([1, 2, 5, 3])
# V shape:  torch.Size([1, 2, 5, 3])
# socres shape:  torch.Size([1, 2, 5, 5])
# decoder_input shape:  torch.Size([1, 5])
# encoder_input shape:  torch.Size([1, 4])
# pad_attn_mask:  tensor([[[False, False, False,  True]]])
# decoder_encoder_attn_mask shape:  torch.Size([1, 5, 4])
# decoder_encoder_attn_mask value:  tensor([[[False, False, False,  True],
#          [False, False, False,  True],
#          [False, False, False,  True],
#          [False, False, False,  True],
#          [False, False, False,  True]]])
# Q shape:  torch.Size([1, 2, 5, 3])
# K shape:  torch.Size([1, 2, 4, 3])
# V shape:  torch.Size([1, 2, 4, 3])
# socres shape:  torch.Size([1, 2, 5, 4])
# logits shape:  torch.Size([5, 5])
# logits:  tensor([[ 3.1167, 14.3907,  4.0073, -8.3052,  1.9785],
#         [ 9.8962, 13.2096, -0.5747, -9.7799,  3.5187],
#         [ 9.8962, 13.2096, -0.5747, -9.7799,  3.5187],
#         [ 9.8962, 13.2096, -0.5747, -9.7799,  3.5187],
#         [ 9.8962, 13.2096, -0.5747, -9.7799,  3.5187]],
#        grad_fn=<SqueezeBackward1>)
# prob:  tensor([1, 1, 1, 1, 1])
# next_symbol:  1
# 下一个加入到decoder_input，用于mask的字符是:  1
# 第 0 轮 I
