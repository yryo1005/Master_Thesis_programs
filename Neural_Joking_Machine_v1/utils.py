import os
import json
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import subprocess
if not os.path.exists("Japanese_BPEEncoder_V2"):
    subprocess.run(["git", "clone", "https://github.com/tanreinama/Japanese-BPEEncoder_V2.git", "Japanese_BPEEncoder_V2"])
from Japanese_BPEEncoder_V2.encode_swe import SWEEncoder_ja

# ハイパーパラメータ（実験ごとに変更しない）
EPOCH = 25
BATCH_SIZE = 32
LEARNING_RATO = 0.0001

DATA_DIR = "../../datas/boke_data_assemble/"
IMAGE_DIR = "../../datas/boke_image/"

IMAGE_FEATURE_DIR = "../../datas/encoded/resnet152_image_feature/"
if not os.path.exists(IMAGE_FEATURE_DIR):
    os.mkdir(IMAGE_FEATURE_DIR)

# 画像のデータローダを作る関数
def make_image_dataloader(image_numbers, num_workers = 4):
    # 画像の前処理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    class LoadImageDataset(Dataset):
        def __init__(self, image_numbers):
            """
                image_numbers: 画像の番号からなるリスト
            """
            self.image_numbers = image_numbers

        def __len__(self):
            return len(self.image_numbers)

        def __getitem__(self, idx):
            image = Image.open(f"{IMAGE_DIR}{self.image_numbers[idx]}.jpg").convert("RGB")

            return image, self.image_numbers[idx]
    
    def collate_fn_tf(batch):
        images = torch.stack([transform(B[0]) for B in batch])
        image_numbers = [B[1] for B in batch]

        return images, image_numbers

    print(f"num data: {len(image_numbers)}")

    dataset = LoadImageDataset(image_numbers)
    dataloader = DataLoader(
        dataset, 
        batch_size = BATCH_SIZE, 
        num_workers = num_workers, 
        collate_fn = collate_fn_tf
    )

    return dataloader

# 大喜利生成AIの学習用データローダを作る関数
def make_dataloader(boke_datas, max_sentence_length, num_workers = 4):
    """
        boke_datas: {"image_number":画像のお題番号 ,"tokenized_boke":トークナイズされた大喜利}からなるリスト
        max_sentence_length: 学習データの最大単語数(<START>, <END>トークンを含まない)
        num_workers: データローダが使用するCPUのスレッド数
    """
    class SentenceGeneratorDataset(Dataset):
        def __init__(self, image_file_numbers, sentences, teacher_signals):
            """
                image_file_numbers: 画像の番号からなるリスト
                sentences: 入力文章からなるリスト
                teacher_signals: 教師信号からなるリスト
            """
            if len(image_file_numbers) != len(sentences) and len(teacher_signals) != len(sentences):
                raise ValueError("データリストの長さが一致しません")

            self.image_file_numbers = image_file_numbers
            self.sentences = sentences
            self.teacher_signals = teacher_signals

        def __len__(self):
            return len(self.teacher_signals)

        def __getitem__(self, idx):
            image_feature = np.load(f"{IMAGE_FEATURE_DIR}{self.image_file_numbers[idx]}.npy")
            sentence = self.sentences[idx]
            teacher_signal = self.teacher_signals[idx]

            return image_feature, sentence, teacher_signal

    def collate_fn_tf(batch):
        image_features = torch.tensor(np.array([B[0] for B in batch]))
        sentences = torch.tensor(np.array([B[1] for B in batch]))
        teacher_signals = torch.tensor(np.array([B[2] for B in batch]))

        return image_features, sentences, teacher_signals

    image_file_numbers = list()
    sentences = list()
    teacher_signals = list()

    for D in tqdm(boke_datas):
        image_file_numbers.append(D["image_number"])
        tmp = D["tokenized_boke"] + [0] * (2 + max_sentence_length - len(D["tokenized_boke"]))
        sentences.append(tmp[:-1])
        teacher_signals.append(tmp[1:])

    dataset = SentenceGeneratorDataset(image_file_numbers, sentences, teacher_signals)
    dataloader = DataLoader(
        dataset, 
        batch_size = BATCH_SIZE, 
        num_workers = num_workers, 
        collate_fn = collate_fn_tf
    )

    print(f"num data: {len(teacher_signals)}")

    return dataloader

# 大喜利生成モデルのクラス
# 大喜利生成モデルのクラス
class BokeGeneratorModel(nn.Module):
    def __init__(self, num_word, image_feature_dim, sentence_length, embedding_dim = 512):
        """
            num_word: 学習に用いる単語の総数
            image_feature_dim: 画像の特徴量の次元数
            sentence_length: 入力する文章の単語数
            embedding_dim: 単語の埋め込み次元数
        """
        super(BokeGeneratorModel, self).__init__()
        self.num_word = num_word
        self.image_feature_dim = image_feature_dim
        self.sentence_length = sentence_length
        self.embedding_dim = embedding_dim
        
        self.fc1 = nn.Linear(image_feature_dim, embedding_dim)
        self.embedding = nn.Embedding(num_word, embedding_dim, padding_idx = 0)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = embedding_dim, 
                            batch_first = True)
        self.fc2 = nn.Linear(embedding_dim + embedding_dim, 2 * embedding_dim)
        self.fc3 = nn.Linear(2 * embedding_dim, 2 * embedding_dim)
        self.fc4 = nn.Linear(2 * embedding_dim, num_word)
    
    # LSTMの初期値は0で，画像の特徴量と文章の特徴量を全結合層の前で結合する
    def forward(self, image_features, sentences):
        """
            image_features: 画像の特徴量
            sentences: 入力する文章
        """
        x1 = F.leaky_relu(self.fc1(image_features))
        x1 = x1.unsqueeze(1).repeat(1, self.sentence_length, 1)

        x2 = self.embedding(sentences)
        x2, _ = self.lstm(x2)

        x = torch.cat((x1, x2), dim = -1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        return self.fc4(x)

# 文章生成の精度を計算する関数
def calculate_accuracy(teacher_signals, outputs):
    """
        teacher_signals: 教師信号
        outputs: モデルの出力
    """
    _, predicted_words = outputs.max(dim = -1)
    # パディングに対して精度を計算しない
    mask = (teacher_signals != 0)
    correct = ((predicted_words == teacher_signals) & mask).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

# 1イテレーション学習する関数
def train_step(model, optimizer, batch_data, batch_labels):
    optimizer.zero_grad()
    outputs = model(*batch_data)
    # パディングに対して損失を計算しない
    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch_labels.view(-1),
                           ignore_index = 0)
    accuracy = calculate_accuracy(batch_labels, F.softmax(outputs, dim = -1))
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy

# 1イテレーション検証する関数
def evaluate(model, batch_data, batch_labels):
    with torch.no_grad():
        outputs = model(*batch_data)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch_labels.view(-1),
                               ignore_index = 0)
        accuracy = calculate_accuracy(batch_labels, F.softmax(outputs, dim = -1))
    return loss.item(), accuracy

# 大喜利生成AI
class NeuralJokingMachine:
    def __init__(self, weight_path, index_to_word, sentence_length, embedding_dim = 512):
        """
            weight_path: 大喜利適合判定モデルの学習済みの重みのパス
            index_to_word: 単語のID: 単語の辞書(0:<PAD>, 1:<START>, 2:<END>)
            sentence_length: 入力する文章の単語数
            embedding_dim: 単語の埋め込み次元数
        """
        self.index_to_word = index_to_word
        self.sentence_length = sentence_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.boke_generate_model = BokeGeneratorModel(
                                        num_word = len(index_to_word), 
                                        image_feature_dim = 2048, 
                                        sentence_length = sentence_length, 
                                        embedding_dim = embedding_dim)
        self.boke_generate_model.load_state_dict(torch.load(weight_path))
        self.boke_generate_model.to(self.device)
        self.boke_generate_model.eval()

        self.resnet152 = models.resnet152(pretrained = True)
        self.resnet152 = torch.nn.Sequential(*list(self.resnet152.children())[:-1] + [nn.Flatten()])
        self.resnet152 = self.resnet152.to(self.device)
        self.resnet152.eval()

        # 画像の前処理
        self.image_preprocesser = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image_path, argmax = False, top_k = 5):
        """
            image_path: 大喜利を生成したい画像のパス
            argmax: Trueなら最大確率の単語を選ぶ, FalseならTop-Kサンプリングを行う
            top_k: Top-Kサンプリング時に考慮する単語の数
        """
        image = Image.open(image_path)
        preprocessed_image = self.image_preprocesser(image).to(self.device)
        image_feature = self.resnet152( preprocessed_image.unsqueeze(0) ) # (1, 2048)
        
        generated_text = [1] # <START>トークン
        for i in range(1, self.sentence_length):
            tmp = generated_text + [0] * (self.sentence_length - i) # Padding
            tmp = torch.Tensor(np.array(tmp)).unsqueeze(0).to(self.device).to(dtype=torch.int32) # (1, sentence_length)
            pred = self.boke_generate_model(image_feature, tmp) # (1, sentence_length, num_word)
            target_pred = pred[0][i - 1]

            if argmax:
                # 最大確率の単語を選ぶ
                chosen_id = torch.argmax(target_pred).item()
            else:
                # Top-Kサンプリング
                top_k_probs, top_k_indices = torch.topk(target_pred, top_k)
                top_k_probs = torch.nn.functional.softmax(top_k_probs, dim = -1)
                chosen_id = np.random.choice(top_k_indices.detach().cpu().numpy(), 
                                             p = top_k_probs.detach().cpu().numpy())
            
            generated_text.append(chosen_id)
            if chosen_id == 2:
                break
        
        generated_sentence = ""
        for I in generated_text[1:-1]:
            generated_sentence += self.index_to_word[I]
        return generated_sentence
