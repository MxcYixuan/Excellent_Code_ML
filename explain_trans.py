import torch
from torch import nn
from torch import optim
from torch.utils import data as Data
import numpy as np
import ipdb


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
