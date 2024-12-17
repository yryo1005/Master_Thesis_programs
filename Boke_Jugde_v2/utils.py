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

import japanese_clip as ja_clip
from transformers import MLukeTokenizer, LukeModel

# ハイパーパラメータ（実験ごとに変更しない）
EPOCH = 25
BATCH_SIZE = 64
LEARNING_RATO = 0.0001

DATA_DIR = "../../datas/boke_data_assemble/"
CLIP_IMAGE_FEATURE_DIR = "../../datas/encoded/clip_image_feature/"
CLIP_SENTENCE_FEATURE_DIR = "../../datas/encoded/clip_sentence_feature/"
LUKE_SENTENCE_FEATURE_DIR = "../../datas/encoded/luke_sentence_feature/"

# データローダを作る関数
def make_dataloader(boke_datas, caption_datas, 
                    use_caption = False, use_miss_boke = False, num_ratio_miss_boke = 1,
                    num_workers = 4):
    """
        boke_datas: {"image_number":画像のお題番号 ,"boke_number":image_numberの画像に投稿されたboke_number番目の大喜利}からなるリスト
        caption_datas: {"image_number":画像のお題番号}からなるリスト
        use_caption: キャプションを負例として用いるか
        use_miss_boke: 他の画像の大喜利を負例として用いるか
        num_ratio_miss_boke: 正例の何倍の別の画像の大喜利を使用するか
        num_workers: データローダが使用するCPUのスレッド数
    """
    class LoadNpyDataset(Dataset):
        def __init__(self, image_file_paths, sentence_file_paths, teacher_signals):
            """
                image_file_paths: 画像の特徴量のパス(ディレクトリ，.npyを含めない)からなるリスト
                sentence_file_paths: 文章の特徴量のパス(boke/image_number/boke_number，またはcaption/image_number，.npyを含めない)からなるリスト
                teacher_signals: 教師信号(0または1)からなるリスト
            """
            if len(image_file_paths) != len(sentence_file_paths) and len(sentence_file_paths) != len(teacher_signals):
                raise ValueError("データリストの長さが一致しません")

            self.image_file_paths = image_file_paths
            self.sentence_file_paths = sentence_file_paths
            self.teacher_signals = teacher_signals

        def __len__(self):
            return len(self.teacher_signals)

        def __getitem__(self, idx):
            clip_image_feature = np.load(f"{CLIP_IMAGE_FEATURE_DIR}{self.image_file_paths[idx]}.npy")
            clip_sentence_feature = np.load(f"{CLIP_SENTENCE_FEATURE_DIR}{self.sentence_file_paths[idx]}.npy")
            luke_sentence_feature = np.load(f"{LUKE_SENTENCE_FEATURE_DIR}{self.sentence_file_paths[idx]}.npy")
            teacher_signal = self.teacher_signals[idx]

            return clip_image_feature, clip_sentence_feature, luke_sentence_feature, teacher_signal

    def collate_fn_tf(batch):
        clip_image_features = torch.Tensor(np.array([B[0] for B in batch]))
        clip_sentence_features = torch.Tensor(np.array([B[1] for B in batch]))
        luke_sentence_features = torch.Tensor(np.array([B[2] for B in batch]))
        teacher_signals = torch.Tensor(np.array([float(B[3]) for B in batch])[..., np.newaxis])
        
        return clip_image_features, clip_sentence_features, luke_sentence_features, teacher_signals

    image_file_numbers = list()
    sentence_file_numbers = list()
    teacher_signals = list()

    for D in boke_datas:
        image_file_numbers.append(D["image_number"])
        sentence_file_numbers.append(f'boke/{D["image_number"]}/{D["boke_number"]}')
        teacher_signals.append(1)

    if use_caption:
        for D in caption_datas:
            image_file_numbers.append(D["image_number"])
            sentence_file_numbers.append(f'caption/{D["image_number"]}')
            teacher_signals.append(0)
    
    if use_miss_boke:
        miss_boke_datas = list()

        # num_ratio_miss_boke回だけ負例を作る
        for _ in range(num_ratio_miss_boke):
            tmp_idx = np.random.randint(0, len(boke_datas), size = (len(boke_datas), ))
            for i, idx in tqdm(enumerate(tmp_idx)):
                # ランダムに選んだ大喜利の画像が正例の画像と同じ限り繰り返す
                while boke_datas[idx]["image_number"] == boke_datas[i]["image_number"]:
                    idx = np.random.randint(0, len(boke_datas))

                miss_boke_datas.append({
                    "boke_path": f'boke/{boke_datas[idx]["image_number"]}/{boke_datas[idx]["boke_number"]}',
                    "image_number": boke_datas[i]["image_number"]
                })
            
        for D in miss_boke_datas:
            image_file_numbers.append(D["image_number"])
            sentence_file_numbers.append(D["boke_path"])
            teacher_signals.append(0)
    
    print(f"num data: {len(teacher_signals)}")

    tmp = list(zip(image_file_numbers, sentence_file_numbers, teacher_signals))
    np.random.shuffle(tmp)
    image_file_numbers, sentence_file_numbers, teacher_signals = zip(*tmp)

    dataset = LoadNpyDataset(image_file_numbers, sentence_file_numbers, teacher_signals)
    dataloader = DataLoader(
        dataset, 
        batch_size = BATCH_SIZE, 
        num_workers = num_workers, 
        collate_fn = collate_fn_tf
    )

    return dataloader

# 大喜利適合判定モデルのクラス
class BokeJudgeModel(nn.Module):
    def __init__(self, cif_dim = 512, csf_dim = 512, lsf_dim = 768):
        """
            cif_dim: CLIPの画像の特徴量の次元数
            csf_dim: CLIPの文章の特徴量の次元数
            lsf_dim: Sentene-LUKEの文章の特徴量の次元数
        """
        super(BokeJudgeModel, self).__init__()
        self.cif_dim = cif_dim
        self.csf_dim = csf_dim
        self.lsf_dim = lsf_dim
        
        self.fc1 = nn.Linear(cif_dim + csf_dim + lsf_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.output_layer = nn.Linear(1024, 1)
        
    def forward(self, cif, csf, lsf):
        """
            cif: CLIPの画像の特徴量
            csf: CLIPの文章の特徴量
            lsf: Sentence-LUKEの文章の特徴量
        """
        x = torch.cat([cif, csf, lsf], dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        output = torch.sigmoid(self.output_layer(x))
        return output

# 二値分類の精度を計算する関数
def calculate_accuracy(teacher_signals, outputs, threshould = 0.5):
    """
        teacher_signals: 教師信号
        outputs: モデルの出力
        threshould: しきい値
    """
    return ((outputs > 0.5).float() == teacher_signals).float().mean()

# 1イテレーション学習する関数
def train_step(model, optimizer, batch_data, batch_labels):
    optimizer.zero_grad()
    outputs = model(*batch_data)
    loss = nn.BCELoss()(outputs, batch_labels)
    accuracy = calculate_accuracy(batch_labels, outputs)
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy.item()

# 1イテレーション検証する関数
def evaluate(model, batch_data, batch_labels):
    with torch.no_grad():
        outputs = model(*batch_data)
        loss = nn.BCELoss()(outputs, batch_labels)
        accuracy = calculate_accuracy(batch_labels, outputs)
    return loss.item(), accuracy.item()

# Sentence-LUKEのクラス
class SentenceLukeJapanese:
    def __init__(self, device = None):
        self.tokenizer = MLukeTokenizer.from_pretrained("sonoisa/sentence-luke-japanese-base-lite")
        self.model = LukeModel.from_pretrained("sonoisa/sentence-luke-japanese-base-lite",
                                               torch_dtype = torch.float16)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size = 256):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest",
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)

# 大喜利適合判定AI
class BokeJugeAI:
    def __init__(self, weight_path):
        """
            weight_path: 大喜利適合判定モデルの学習済みの重みのパス
        """
        # 大喜利適合判定AIの読み込み
        self.boke_judge_model = BokeJudgeModel()
        self.boke_judge_model.load_state_dict(torch.load(weight_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.boke_judge_model.to(self.device)
        self.boke_judge_model.eval()

        # CLIP
        self.clip_model, self.clip_preprocesser = ja_clip.load("rinna/japanese-clip-vit-b-16",
                                             cache_dir="/tmp/japanese_clip",
                                             torch_dtype = torch.float16,
                                             device = self.device)
        self.clip_tokenizer = ja_clip.load_tokenizer()

        # Sentence-LUKE
        self.luke_model = SentenceLukeJapanese()

    def __call__(self, image_path, sentence):
        """
            image_path: 判定したい大喜利のお題画像
            sentence: 判定したい大喜利
        """
        # CLIPによる特徴量への変換
        tokenized_sentences = ja_clip.tokenize(
            texts = [sentence],
            max_seq_len = 77,
            device = self.device,
            tokenizer = self.clip_tokenizer,
            )
        image = Image.open(image_path)
        preprcessed_image = self.clip_preprocesser(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            clip_image_features = self.clip_model.get_image_features(preprcessed_image)
            clip_sentence_features = self.clip_model.get_text_features(**tokenized_sentences)

        # Sentence-LUKEによる特徴量への変換
        luke_sentence_feature = self.luke_model.encode([sentence])

        # 大喜利適合判定AIの推論
        with torch.no_grad():
            outputs = self.boke_judge_model(clip_image_features,
                                        clip_sentence_features,
                                        luke_sentence_feature.to(self.device))

        return outputs.cpu().numpy()