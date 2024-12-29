import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import chardet
import numpy as np

class TextPreprocessor:
    def __init__(self, window_size=2, min_freq=2, stop_words=None):
        self.window_size = window_size
        self.min_freq = min_freq
        self.vocab = {}
        self.count_vocab = {}
        self.index2word = {}
        self.word2index = {}
        self.stop_words = stop_words if stop_words else set()

    def build_vocab(self, corpus):
        tokens = [word for sentence in corpus for word in re.sub(r'[^\w\s]', '', sentence).split() if
                  word.lower() not in self.stop_words]
        word_counts = Counter(tokens)
        self.count_vocab = {word.lower(): count for word, count in word_counts.items() if count >= self.min_freq}
        self.index2word = {i: word for i, (word, _) in enumerate(self.count_vocab.items())}
        self.word2index = {word: i for i, word in self.index2word.items()}
        self.vocab = list(self.count_vocab.keys())

    def generate_training_data(self, corpus):
        pairs = []
        for sentence in corpus:
            words = re.sub(r'[^\w\s]', '', sentence).lower().split()
            for i, word in enumerate(words):
                if word in self.stop_words or word not in self.count_vocab:
                    continue
                target = self.word2index[word]
                start = max(0, i - self.window_size)
                end = min(len(words), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j and words[j] not in self.stop_words and words[j] in self.count_vocab:
                        context = self.word2index[words[j]]
                        pairs.append((target, context))
        return pairs


def subsample_data(vocab, word_counts,target_samples, t=0.001):
    total_words = sum(word_counts.values())  # 总词数
    # print('Total words: {}'.format(total_words))
    word_probabilities = {}  # 每个词被采样的概率

    # 计算每个词的采样概率
    for word, freq in word_counts.items():
        prob = 1 - np.sqrt(t / (freq / total_words))
        word_probabilities[word] = prob if prob > 0 else 0  # 确保采样概率不小于0

    # 基于采样概率选择词
    subsampled_words = []
    while len(subsampled_words) < target_samples:
        word = random.choice(vocab)  # 从词汇表中随机选择一个词
        if random.random() < word_probabilities.get(word, 0):  # 根据概率决定是否采样
            subsampled_words.append(word)

    return subsampled_words

class Skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Skipgram, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)  # 目标词嵌入
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)  # 上下文词嵌入

    def forward(self, target, context, neg_samples):
        target_emb = self.in_embed(target)  # 目标词嵌入
        context_emb = self.out_embed(context)  # 上下文词嵌入
        neg_emb = self.out_embed(neg_samples)  # 负样本词嵌入

        # 正样本损失
        pos_score = torch.bmm(target_emb.unsqueeze(1), context_emb.unsqueeze(2)).squeeze()
        pos_loss = torch.log(torch.sigmoid(pos_score))  # 正样本损失

        # 负样本损失
        neg_score = torch.bmm(target_emb.unsqueeze(1), neg_emb.transpose(1, 2))
        neg_loss = torch.log(torch.sigmoid(-neg_score)).sum(dim=2)  # 负样本损失

        # 总损失
        loss = - (pos_loss + neg_loss).mean()  # 总损失
        return loss


class SkipgramDataset(Dataset):
    def __init__(self, training_pairs, vocab_size, num_neg_samples,vocab,word_count,word2idx):
        self.training_pairs = training_pairs
        self.vocab_size = vocab_size
        self.num_neg_samples = num_neg_samples
        self.vocab = vocab
        self.word_count = word_count
        self.word2idx = word2idx

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        target, context = self.training_pairs[idx]

        # 负采样
        neg_samples = subsample_data(self.vocab,self.word_count,self.num_neg_samples)
        neg_samples = [self.word2idx.get(word,-1) for word in neg_samples]

        return torch.LongTensor([target]), torch.LongTensor([context]), torch.LongTensor(neg_samples)

def get_word_vector(model, word_idx):
    return model.in_embed(torch.tensor([word_idx]))

# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()
    return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))

# 获取与给定词最相似的词
def find_most_similar_words(model, word_idx, top_k=5):
    word_vector = get_word_vector(model, word_idx)
    similarities = []
    for idx in range(len(model.in_embed.weight)):  # 遍历词汇表
        other_word_vector = get_word_vector(model, idx)
        sim = cosine_similarity(word_vector, other_word_vector)
        similarities.append((idx, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    most_similar_words = similarities[:top_k]
    most_similar_words = [(preprocessor.index2word[idx], sim) for idx, sim in most_similar_words]
    return most_similar_words


if __name__ == "__main__":
    stop_words = set(stopwords.words('english'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # print("Using device:", device)

    with open('novel.txt', 'rb') as file:  # 以二进制方式读取
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']  # 获取检测到的编码格式

    with open("novel.txt", 'r', encoding=encoding) as file:
        corpus = [line.strip() for line in file.readlines() if line.strip()]

        #实例化文本处理器
        preprocessor = TextPreprocessor(window_size=2, min_freq=2, stop_words=stop_words)
        preprocessor.build_vocab(corpus)

        training_data = preprocessor.generate_training_data(corpus)

        num_neg_samples = 5
        batch_size = 128

        # print(preprocessor.vocab)
        # print(preprocessor.index2word)
        # print(preprocessor.word2index)

        train_dataset = SkipgramDataset(training_data, len(preprocessor.vocab), num_neg_samples,preprocessor.vocab,preprocessor.count_vocab,preprocessor.word2index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        vocab_size = len(preprocessor.vocab)
        embed_size = 50  # 嵌入维度

        # 实例化模型
        model = Skipgram(vocab_size, embed_size).to(device)
        model.load_state_dict(torch.load('own.pth'))
        # criterion = nn.BCEWithLogitsLoss().to(device)  # 使用二分类交叉熵损失
        # optimizer = optim.Adam(model.parameters(), lr=0.01)

        # num_epochs = 50

        # for epoch in range(num_epochs):
        #     total_loss = 0
        #     for target, context, neg_samples in train_loader:
        #         optimizer.zero_grad()  # 清零梯度
        #         # 前向传播
        #         loss = model(target.squeeze(), context.squeeze(), neg_samples)
        #         # 反向传播
        #         loss.backward()
        #         # 更新权重
        #         optimizer.step()
        #
        #         total_loss += loss.item()
        #
        #     torch.save(model.state_dict(), "own.pth")
        #
        #     print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

        word_idx = preprocessor.word2index.get("juliet", None)
        if word_idx is not None:
            similar_words = find_most_similar_words(model, word_idx, top_k=6)[1:]
            print(similar_words)


