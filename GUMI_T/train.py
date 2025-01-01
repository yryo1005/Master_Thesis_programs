import os
import json
import numpy as np
import shutil
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
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

###
# コマンドライン引数の処理
parser = argparse.ArgumentParser(description="設定用プログラム")

parser.add_argument("--num_workers", type = int, default = 16, help = "データローダが使用するCPUのスレッド数(GPUの総スレッド数の8割が推奨)")
parser.add_argument("--reset_data", action = "store_true", help = "データセットを再作成するか")
parser.add_argument("--use_unreal_image", action = "store_true", help = "現実写真以外を使用する")
parser.add_argument("--use_word_image", action = "store_true", help = "文字を含む画像を使用する")
parser.add_argument("--use_unique_noun_boke", action = "store_true", help = "固有名詞を含む大喜利を使用する")
parser.add_argument("--min_star", type = int, default = 0, help = "大喜利の最小の星の数")
parser.add_argument("--min_apper_word", type = int, default = 32, help = "単語の最小出現回数")
parser.add_argument("--min_sentence_length", type = int, default = 4, help = "大喜利の最小単語数")
parser.add_argument("--max_sentence_length", type = int, default = 31, help = "大喜利の最大単語数")
parser.add_argument("--epoch", type = int, default = 25, help = "学習反復回数")
parser.add_argument("--batch_size", type = int, default = 64, help = "バッチサイズ")
parser.add_argument("--learning_rate", type = float, default = 0.0001, help = "学習率")
parser.add_argument("--feature_dim", type = int, default = 1024, help = "モデルの特徴量次元数")
parser.add_argument("--num_heads", type = int, default = 8, help = "Multi Head Self Attentionのヘッド数")

args = parser.parse_args()

###
NUM_WORKERS = args.num_workers
RESET_DATA = args.reset_data

USE_UNREAL_IMAGE = args.use_unreal_image
USE_WORD_IMAGE = args.use_word_image
USE_UNIQUE_NOUN_BOKE = args.use_unique_noun_boke

MIN_STAR = args.min_star
MIN_APPER_WORD = args.min_apper_word
MIN_SENTENCE_LENGTH = args.min_sentence_length
MAX_SENTENCE_LENGTH = args.max_sentence_length

EPOCH = args.epoch
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
FEATURE_DIM = args.feature_dim
NUM_HEADS = args.num_heads

RESULT_DIR = f"../../results/GUMI_T/{USE_UNREAL_IMAGE}_{USE_WORD_IMAGE}_{USE_UNIQUE_NOUN_BOKE}_{MIN_STAR}_{MIN_APPER_WORD}_{MIN_SENTENCE_LENGTH}_{MAX_SENTENCE_LENGTH}_{EPOCH}_{BATCH_SIZE}_{LEARNING_RATE}_{FEATURE_DIM}_{NUM_HEADS}/"

if not os.path.exists("../../results/GUMI_T/"):
    os.mkdir("../../results/GUMI_T/")
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)
print(f"result directory: {RESULT_DIR}")
with open(f"{RESULT_DIR}config.json", "w") as f:
    json.dump({
        "USE_UNREAL_IMAGE": USE_UNREAL_IMAGE,
        "USE_WORD_IMAGE": USE_WORD_IMAGE,
        "USE_UNIQUE_NOUN_BOKE": USE_UNIQUE_NOUN_BOKE,
        "MIN_STAR": MIN_STAR,
        "MIN_APPER_WORD": MIN_APPER_WORD,
        "MIN_SENTENCE_LENGTH": MIN_SENTENCE_LENGTH,
        "MAX_SENTENCE_LENGTH": MAX_SENTENCE_LENGTH,
        "EPOCH": EPOCH,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "FEATURE_DIM": FEATURE_DIM,
        "NUM_HEADS": NUM_HEADS
    }, f)

DATA_DIR = "../../datas/boke_data_assemble/"
IMAGE_DIR = "../../datas/boke_image/"

IMAGE_FEATURE_DIR = "../../datas/encoded/vit_image_feature/"
if not os.path.exists(IMAGE_FEATURE_DIR):
    os.mkdir(IMAGE_FEATURE_DIR)

###
# データセットの作成
if not os.path.exists(f"{RESULT_DIR}index_to_word.json") or RESET_DATA:
    # tokenizer
    with open('Japanese_BPEEncoder_V2/ja-swe32kfix.txt') as f:
        bpe = f.read().split('\n')

    with open('Japanese_BPEEncoder_V2/emoji.json') as f:
        emoji = json.loads(f.read())

    tokenizer = SWEEncoder_ja(bpe, emoji)

    tmp = list()
    word_count_dict = dict()

    for JP in tqdm(os.listdir(DATA_DIR)):
        
        N = int(JP.split(".")[0])

        with open(f"{DATA_DIR}{JP}", "r") as f:
            a = json.load(f)
        
        image_information = a["image_information"]
        is_photographic_probability = image_information["is_photographic_probability"]
        ocr = image_information["ocr"]

        # 現実写真以外を除去
        if not USE_UNREAL_IMAGE:
            if is_photographic_probability < 0.8: continue
            
        # 文字のある画像を除去
        if not USE_WORD_IMAGE:
            if len(ocr) != 0: continue
        
        bokes = a["bokes"]

        for B in bokes:
            # 星が既定の数以下の大喜利を除去
            if B["star"] < MIN_STAR:
                continue

            # 固有名詞を含む大喜利を除去
            if not USE_UNIQUE_NOUN_BOKE:
                if len(B["unique_nouns"]) != 0: continue

            tokenized_boke = tokenizer.encode(B["boke"])
            # 単語数が既定の数でない大喜利を除去
            if not MIN_SENTENCE_LENGTH <= len(tokenized_boke) < MAX_SENTENCE_LENGTH:
                continue

            for W in tokenized_boke:
                try:
                    word_count_dict[W] += 1
                except:
                    word_count_dict[W] = 1
            
            tmp.append({
                "image_number": N,
                "tokenized_boke": tokenized_boke
            })

    # 単語の最小出現回数を満たさない大喜利を除去
    boke_datas = list()
    words = list()

    for D in tqdm(tmp):
        flag = False
        for W in D["tokenized_boke"]:
            if word_count_dict[W] < MIN_APPER_WORD:
                flag = True
                break
        if flag: 
            continue
        
        boke_datas.append({
            "image_number": D["image_number"],
            "tokenized_boke": D["tokenized_boke"]
        })
        words += D["tokenized_boke"]
    words = set(words)
    image_numbers = list(set([D["image_number"] for D in boke_datas]))
    del tmp

    # tokenize
    index_to_index = dict()

    c = 3
    for D in tqdm(boke_datas):
        tmp = list()
        for W in D["tokenized_boke"]:
            try:
                index_to_index[W]
            except:
                index_to_index[W] = c
                c += 1
            tmp.append(index_to_index[W])
        D["tokenized_boke"] = [1] + tmp + [2]

    index_to_word = {
        V: tokenizer.decode([K]) for K, V in index_to_index.items()
    }
    index_to_word[0] = "<PAD>"
    index_to_word[1] = "<START>"
    index_to_word[2] = "<END>"

    #
    train_boke_datas, test_boke_datas = train_test_split(boke_datas, test_size = 0.01)

    with open(f"{RESULT_DIR}train_boke_datas.json", "w") as f:
        json.dump(train_boke_datas, f)
    with open(f"{RESULT_DIR}test_boke_datas.json", "w") as f:
        json.dump(test_boke_datas, f)
    with open(f"{RESULT_DIR}index_to_word.json", "w") as f:
        json.dump(index_to_word, f)

else:
    with open(f"{RESULT_DIR}train_boke_datas.json", "r") as f:
        train_boke_datas = json.load(f)
    with open(f"{RESULT_DIR}test_boke_datas.json", "r") as f:
        test_boke_datas = json.load(f)
    with open(f"{RESULT_DIR}index_to_word.json", "r") as f:
        index_to_word = json.load(f)

    image_numbers = [D["image_number"] for D in train_boke_datas] + [D["image_number"] for D in test_boke_datas]
    image_numbers = list(set(image_numbers))

print(f"学習に用いる大喜利の数: {len(train_boke_datas)}\n", 
      f"検証に用いる大喜利の数: {len(test_boke_datas)}\n",
      f"使用する画像の数: {len(image_numbers)}\n",
      f"単語の種類: {len(index_to_word)}")

###
# 画像のデータローダを作る関数
def make_image_dataloader(image_paths, batch_size, num_workers = 4):
    # 画像の前処理
    image_preprocess =  ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    class LoadImageDataset(Dataset):
        def __init__(self, image_paths):
            """
                image_paths: 画像のパスからなるリスト
            """
            self.image_paths = image_paths

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(image_paths[idx]).convert("RGB")

            return image, self.image_paths[idx]
    
    def collate_fn_tf(batch):
        images = image_preprocess(images = [B[0] for B in batch], 
                                  return_tensors = "pt")
        image_numbers = [B[1] for B in batch]

        return images, image_numbers

    print(f"num data: {len(image_paths)}")

    dataset = LoadImageDataset(image_paths)
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        num_workers = num_workers, 
        collate_fn = collate_fn_tf
    )

    return dataloader

###
# 画像を特徴量に変換する
tmp = list()
for IN in tqdm(image_numbers):
    if os.path.exists(f"{IMAGE_FEATURE_DIR}{IN}.npy"):
        continue
    tmp.append(f"{IMAGE_DIR}{IN}.jpg")

if len(tmp) != 0:
    image_dataloader = make_image_dataloader(tmp, batch_size = 8, num_workers = NUM_WORKERS)

    # vision transformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
    model = model.to(device)
    model.eval()

    for Is, IPs in tqdm(image_dataloader):
        Is = Is.to(device)
        outputs = model(**Is)
        features = outputs.last_hidden_state.detach().cpu().numpy()

        for f, IP in zip(features, IPs):
            N = IP.split("/")[-1].split(".")[0]
            np.save(f"{IMAGE_FEATURE_DIR}{N}", f)

###
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, feature_dim):
        """
            max_len: 
            feature_dim: 
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, feature_dim)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / feature_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadSelfAttentionBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(embed_dim = feature_dim, 
                                                    num_heads = num_heads, batch_first = True)
        self.layer_norm1 = nn.LayerNorm(feature_dim)

        self.fc = nn.Linear(feature_dim, feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
    
    def forward(self, x, attn_mask = None):
        # self attention
        x_dash, _ = self.self_attention(x, x, x, attn_mask = attn_mask)
        x = self.layer_norm1(x_dash) + x

        # feed forward
        x_dash = F.leaky_relu( self.fc(x) )
        return self.layer_norm2(x_dash) + x

class MultiHeadCrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadCrossAttentionBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(embed_dim = feature_dim, 
                                                    num_heads = num_heads, batch_first = True)
        self.layer_norm1 = nn.LayerNorm(feature_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim = feature_dim, 
                                                     num_heads = num_heads, batch_first = True)
        self.layer_norm2 = nn.LayerNorm(feature_dim)

        self.fc = nn.Linear(feature_dim, feature_dim)
        self.layer_norm3 = nn.LayerNorm(feature_dim)
    
    def forward(self, src, tgt, attn_mask = None):
        # self attention
        tgt_dash, _ = self.self_attention(tgt, tgt, tgt, attn_mask = attn_mask)
        tgt = self.layer_norm1(tgt_dash) + tgt
        
        # cross attention
        src_dash, _ = self.cross_attention(tgt, src, src)
        tgt = self.layer_norm2(src_dash) + tgt

        # feed forward
        tgt_dash = F.leaky_relu( self.fc(tgt) )
        return self.layer_norm3(tgt_dash) + tgt

# 大喜利生成モデルのクラス
class TransformerBokeGeneratorModel(nn.Module):
    def __init__(self, num_image_patch, image_feature_dim, num_word, sentence_length, feature_dim = 1024, num_heads = 4):
        """
            num_image_patch
            num_word: 学習に用いる単語の総数
            image_feature_dim: 画像の特徴量の次元数
            sentence_length: 入力する文章の単語数
            feature_dim: 単語の埋め込み次元数
            num_heads: 
        """
        super(TransformerBokeGeneratorModel, self).__init__()
        self.num_word = num_word
        self.image_feature_dim = image_feature_dim
        self.sentence_length = sentence_length
        self.feature_dim = feature_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # image encoder
        self.fc1 = nn.Linear(image_feature_dim, feature_dim)
        self.pe1 = PositionalEncoding(num_image_patch, feature_dim)
        self.self_attention = MultiHeadSelfAttentionBlock(feature_dim = feature_dim, num_heads = num_heads)
        
        # sentence decoder
        self.embedding = nn.Embedding(num_word, feature_dim, padding_idx = 0)
        self.pe2 = PositionalEncoding(sentence_length, feature_dim)

        self.cross_attention1 = MultiHeadCrossAttentionBlock(feature_dim = feature_dim, num_heads = num_heads)
        self.cross_attention2 = MultiHeadCrossAttentionBlock(feature_dim = feature_dim, num_heads = num_heads)
        self.fc2 = nn.Linear(feature_dim, num_word)

    def forward(self, image_features, sentences):
        """
            image_features: 画像の特徴量(batch_size, num_patch, image_feature_dim)
            sentences: 入力する文章(batch_size, sentence_length)
        """
        # encode image 
        src = F.leaky_relu( self.fc1( image_features ) )
        src = self.pe1( src )
        src = self.self_attention( src )

        # decode sentence
        tgt = self.embedding( sentences )
        tgt = self.pe2( tgt )

        attn_mask = torch.triu(torch.ones(self.sentence_length, self.sentence_length), diagonal = 1)
        attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf')).to(self.device)

        tgt = self.cross_attention1( tgt = tgt, src = src, attn_mask = attn_mask )
        tgt = self.cross_attention2( tgt = tgt, src = src, attn_mask = attn_mask  )
        return self.fc2( tgt )

###
# 大喜利生成AIの学習用データローダを作る関数
def make_dataloader(boke_datas, max_sentence_length, batch_size, num_workers = 4):
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
        batch_size = batch_size, 
        num_workers = num_workers, 
        collate_fn = collate_fn_tf
    )

    print(f"num data: {len(teacher_signals)}")

    return dataloader

###
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

###
train_dataloader = make_dataloader(train_boke_datas, max_sentence_length = MAX_SENTENCE_LENGTH, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)
test_dataloader = make_dataloader(test_boke_datas, max_sentence_length = MAX_SENTENCE_LENGTH, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)

model = TransformerBokeGeneratorModel(num_image_patch = 197, 
                                      image_feature_dim = 1024, 
                                      num_word = len(index_to_word), 
                                      sentence_length = MAX_SENTENCE_LENGTH + 1, 
                                      feature_dim = FEATURE_DIM, 
                                      num_heads = NUM_HEADS)

# 学習履歴がある場合，途中から再開する
if os.path.exists(f"{RESULT_DIR}history.json"):
    with open(f"{RESULT_DIR}history.json", "r") as f:
        a = json.load(f)
        train_loss_history = a["train_loss"]
        train_accuracy_history = a["train_accuracy"]
        test_loss_history = a["test_loss"]
        test_accuracy_history = a["test_accuracy"]
    model.load_state_dict(torch.load(f"{RESULT_DIR}model_{len(train_loss_history):03}.pth"))
    START_EPOCH = len(train_loss_history)
else:
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []
    START_EPOCH = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE)

for epoch in range(START_EPOCH, EPOCH):

    # train
    train_loss_obj = 0.0
    train_accuracy_obj = 0.0
    model.train()
    pb = tqdm(train_dataloader, desc = f"Epoch {epoch+1}/{EPOCH}")
    
    for image_features, sentences, teacher_signals in pb:
        image_features = image_features.to(device)
        sentences = sentences.to(device)
        teacher_signals = teacher_signals.to(device)
        
        loss, accuracy = train_step(model, optimizer, (image_features, sentences), teacher_signals)
        train_loss_obj += loss
        train_accuracy_obj += accuracy
        pb.set_postfix({"train_loss": train_loss_obj / (pb.n + 1), "train_accuracy": train_accuracy_obj / (pb.n + 1)})

    train_loss = train_loss_obj / len(train_dataloader)
    train_accuracy = train_accuracy_obj / len(train_dataloader)

    # test
    test_loss_obj = 0.0
    test_accuracy_obj = 0.0
    model.eval()
    pb = tqdm(test_dataloader, desc = "Evaluating")

    for image_features, sentences, teacher_signals in pb:
        image_features = image_features.to(device)
        sentences = sentences.to(device)
        teacher_signals = teacher_signals.to(device)

        loss, accuracy = evaluate(model, (image_features, sentences), teacher_signals)
        test_loss_obj += loss
        test_accuracy_obj += accuracy
        pb.set_postfix({"test_loss": test_loss_obj / (pb.n + 1), "test_accuracy": test_accuracy_obj / (pb.n + 1)})

    test_loss = test_loss_obj / len(test_dataloader)
    test_accuracy = test_accuracy_obj / len(test_dataloader)

    print(f"Epoch: {epoch+1}/{EPOCH}, "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("-" * 25)

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    torch.save(model.state_dict(), f"{RESULT_DIR}model_{len(train_loss_history):03}.pth")
    if os.path.exists(f"{RESULT_DIR}model_{len(train_loss_history) - 1:03}.pth"):
        os.remove(f"{RESULT_DIR}model_{len(train_loss_history) - 1:03}.pth")
        
    # 学習精度を更新した場合、重みを保存
    if max(train_accuracy_history) == train_accuracy:
        torch.save(model.state_dict(), f"{RESULT_DIR}best_model.pth")

    # 学習結果を保存
    with open(f"{RESULT_DIR}history.json", "w") as f:
        json.dump({
            "train_loss": train_loss_history,
            "train_accuracy": train_accuracy_history,
            "test_loss": test_loss_history,
            "test_accuracy": test_accuracy_history
        }, f)

# 学習結果を描画
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(train_loss_history, label = "train")
ax.plot(test_loss_history, label = "test")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
ax.grid()

ax = fig.add_subplot(1, 2, 2)
ax.plot(train_accuracy_history, label = "train")
ax.plot(test_accuracy_history, label = "test")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.legend()
ax.grid()

fig.savefig(f"{RESULT_DIR}history.png")