!pip install ipdb
import torch
from torch import nn
from torch import optim
from torch.utils import data as Data
import numpy as np
import ipdb

d_model = 6 # embedding size
d_ff = 12 # feedforward nerual network  dimension
d_k = d_v = 3 # dimension of k(same as q) and v
n_heads = 2 # number of heads in multihead attention
p_drop = 0.1 # propability of dropout
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


source_vocab = {'E' : 0, '我' : 1, '吃' : 2, '肉' : 3}
target_vocab = {'E' : 0, 'I' : 1, 'eat' : 2, 'meat' : 3, 'S' : 4}

encoder_input = torch.LongTensor([[1,2,3,0]]).to(device) # 我 吃 肉 E, E代表结束词
decoder_input = torch.LongTensor([[4,1,2,3]]).to(device) # S I eat meat, S代表开始词, 并右移一位，用于并行训练
target = torch.LongTensor([[1,2,3,0]]).to(device) # I eat meat E, 翻译目标

def get_attn_pad_mask(seq_q, seq_k): # 本质是结尾E做注意力遮盖，返回 1*4*4，最后一列为True
  batch, len_q = seq_q.size() # 1, 4
  batch, len_k = seq_k.size() # 1, 4
  pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # 为0则为true，变为f,f,f,true，意思是把0这个结尾标志为true, : 1,1,4
  return pad_attn_mask.expand(batch, len_q, len_k) # 扩展为1*4*4，最后一列为true，表示抹掉结尾对应的注意力

def get_attn_subsequent_mask(seq): # decoder的自我顺序注意力遮盖，右上三角形区为true的遮盖
  attn_shape = [seq.size(0), seq.size(1), seq.size(1)] # 1, 4, 4
  subsequent_mask = np.triu(np.ones(attn_shape), k=1)
  subsequent_mask = torch.from_numpy(subsequent_mask)
  return subsequent_mask

class ScaledDotProductAttention(nn.Module):
  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()

  def forward(self, Q, K, V, attn_mask):
    # Q 1*2*4*3  乘以 K的倒置 1*2*3*4 得到 score 1*2*4*4 4*4表示4个词和词间的注意力矩阵
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
    scores.masked_fill_(attn_mask, -1e9) # 遮盖区的值设为近0，表示E结尾 or decoder的自我顺序遮盖，注意力丢弃

    attn = nn.Softmax(dim=-1)(scores) # softmax后，遮盖区变为0
    prob = torch.matmul(attn, V) # 4*4 乘以 V 4*3 变为 4*3，本质上prob的形状==V，乘积意义是将V带上了注意力信息
    return prob

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
    # input_Q 1*4*6 每批1句 * 每句4个词 * 每词6长度编码
    residual, batch = input_Q, input_Q.size(0)

    Q = self.W_Q(input_Q) # 乘以 W(6*6) 变为 1*4*6
    Q = Q.view(batch, -1, n_heads, d_k).transpose(1, 2) # 切开为2个Head 变为 1*2*4*3 1批 2个Head 4词 3编码
    K = self.W_K(input_K).view(batch, -1, n_heads, d_k).transpose(1, 2)
    V = self.W_V(input_V).view(batch, -1, n_heads, d_v).transpose(1, 2)

    attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # 1*2*4*4 2个Head的4*4，最后一列为true

    prob = ScaledDotProductAttention()(Q, K, V, attn_mask) #返回1*2*4*3 2个头，4*3为带上关注关系的4词

    # 把2头重新拼接起来，变为 1*4*6
    prob = prob.transpose(1, 2).contiguous()
    prob = prob.view(batch, -1, n_heads * d_v).contiguous()

    output = self.fc(prob)
    return self.layer_norm(residual + prob) # return 1*4*6

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.source_embedding = nn.Embedding(len(source_vocab), d_model)
    self.attention = MultiHeadAttention()

  def forward(self, encoder_input): # input 1 * 4 4个单词的编码
    embedded = self.source_embedding(encoder_input) # 1 * 4 * 6 将每个单词的整数字编码扩展到6个浮点数编码
    mask = get_attn_pad_mask(encoder_input, encoder_input) # 1 * 4 * 4的矩阵，最后一列为true，表示忽略结尾词的注意力机制
    encoder_output = self.attention(embedded, embedded, embedded, mask) # 1*4*6，带上关注力的4个词矩阵
    return encoder_output

class Decoder(nn.Module):

  def __init__(self):
    super(Decoder, self).__init__()
    self.target_embedding = nn.Embedding(len(target_vocab), d_model)
    self.attention = MultiHeadAttention()
  # 三入参形状分别为 1*4, 1*4, 1*4*6，前两者未被embedding
  def forward(self, decoder_input, encoder_input, encoder_output):
    decoder_embedded = self.target_embedding(decoder_input) # 编码为1*4*6

    decoder_self_attn_mask = get_attn_pad_mask(decoder_input, decoder_input) # 1*4*4 全为false，表示没有结尾词
    decoder_subsequent_mask = get_attn_subsequent_mask(decoder_input) # 1*4*4 右上三角区为1，其余为0
    decoder_self_mask = torch.gt(decoder_self_attn_mask + decoder_subsequent_mask, 0) #1*4*4 右上三角区为true，其余为false
    decoder_output = self.attention(decoder_embedded, decoder_embedded, decoder_embedded, decoder_self_mask) # 1*4*6 带上注意力的4词矩阵

    decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input, encoder_input) #1*4*4 最后一列为true，表示E结尾词
    # 输入均为1*4*6，Q表示"S I eat meat" K表示"我吃肉E" V表示 "我吃肉E"，整体表示"我吃肉E"，并带上和"我吃肉E"、"S I eat meat"的关注力矩阵
    decoder_output = self.attention(decoder_output, encoder_output, encoder_output, decoder_encoder_attn_mask)
    return decoder_output

class Transformer(nn.Module):
  def __init__(self):
    super(Transformer, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.fc = nn.Linear(d_model, len(target_vocab), bias=False)

  def forward(self, encoder_input, decoder_input):
    # 入 1*4，出 1*4*6，作用是将"我吃肉E"embedding，并带上三词间的关注力信息
    encoder_output = self.encoder(encoder_input)
    # 入 1*4,1*4,1*4*6， 作用是将"S I eat meat"编码并带上三词间关注力，并且将"我吃肉E + S I eat meat"的词间关注力带给"我吃肉E"
    decoder_output = self.decoder(decoder_input, encoder_input, encoder_output)
    # 将带上各种关注力的 "我吃肉E" 4*6编码矩阵，变形为4*5，表示预测出4个词，每个词对应到词典中5个词的概率
    decoder_logits = self.fc(decoder_output)

    return decoder_logits.view(-1, decoder_logits.size(-1))

model = Transformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-1)

for epoch in range(50):
  output = model(encoder_input, decoder_input) # 输出4*5，代表预测出4个词，每个词对应到词典中5个词的概率

  loss = criterion(output, target.view(-1)) # 和目标词 I eat meat E做差异计算
  print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# 使用模型
target_len = len(target_vocab) # 预测目标是5个单词
encoder_output = model.encoder(encoder_input) # 1*4*6 输入"我吃肉E"，进行embedding，带上自我关注力
decoder_input = torch.zeros(1, target_len).type_as(encoder_input.data) # 1*5 全是0，表示EEEEE
next_symbol = 4 # 表示S开始字符

for i in range(target_len): # 5个单词逐个预测
  #ipdb.set_trace()
  decoder_input[0][i] = next_symbol # 譬如i=0第一轮，decoder输入为SEEEE，第二轮为S I EEE，第三轮S I eat EEE
  # 1*5, 1*4, 1*4*6 => 1*5*6
  decoder_output = model.decoder(decoder_input, encoder_input, encoder_output)
  logits = model.fc(decoder_output).squeeze(0) # 5*5 表示预测出5个词，每个词在词典中的概率
  prob = logits.max(dim=1, keepdim=False)[1] # 取出概率最大的五个词的下标，譬如[1, 3, 3, 3, 3] 表示 i,meat,meat,meat,meat
  next_symbol = prob.data[i].item() #譬如i=0第一轮，data[0]=1 表示 i。并重新组装给下一轮的decoder

  for k,v in target_vocab.items():
    if v == next_symbol:
      print('第',i,'轮:',k)
      break

  if next_symbol == 0: # 遇到结尾了，那就完成翻译
    break



