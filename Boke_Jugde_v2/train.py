import argparse
from utils import *

# コマンドライン引数の処理
parser = argparse.ArgumentParser(description="設定用プログラム")

# 実験番号
parser.add_argument("--experience_number", type = str, help = "実験番号")
# PCによって変更する
parser.add_argument("--num_workers", type = int, default = 16, help = "データローダが使用するCPUのスレッド数(GPUの総スレッド数の8割が推奨)")

# データセットが既に存在する場合に，再度作り直すか
parser.add_argument("--reset_data", action = "store_true", help = "データセットを再作成するか")
# 現実写真以外を使用するか
parser.add_argument("--use_unreal_image", action = "store_true", help = "現実写真以外を使用する")
# 文字を含む画像を使用するか
parser.add_argument("--use_word_image", action = "store_true", help = "文字を含む画像を使用する")
# 固有名詞を含む大喜利を使用するか
parser.add_argument("--use_unique_noun_boke", action = "store_true", help = "固有名詞を含む大喜利を使用する")
# 負例としてキャプションを使用するか
parser.add_argument("--use_caption", action = "store_true", help = "負例としてキャプションを使用する")
# 負例として別の画像の大喜利を使用するか
parser.add_argument("--use_miss_boke", action = "store_true", help = "負例として別の画像の大喜利を使用する")
# 正例の何倍の別の画像の大喜利を使用するか
parser.add_argument("--num_ratio_miss_boke", type = int, default = 1, help = "正例の何倍の別の画像の大喜利を使用するか")

args = parser.parse_args()


# PCによって変更する
NUM_WORKERS = args.num_workers
# データセットが既に存在する場合に，再度作り直すか
RESET_DATA = args.reset_data

EXPERIENCE_NUMBER = args.experience_number

# 現実写真以外を使用するか
USE_UNREAL_IMAGE = args.use_unreal_image
# 文字を含む画像を使用するか
USE_WORD_IMAGE = args.use_word_image
# 固有名詞を含む大喜利を使用するか
USE_UNIQUE_NOUN_BOKE = args.use_unique_noun_boke
# 負例としてキャプションを使用するか
USE_CAPTION = args.use_caption
# 負例として別の画像の大喜利を使用するか
USE_MISS_BOKE = args.use_miss_boke
# 正例の何倍の別の画像の大喜利を使用するか
NUM_RATIO_MISS_BOKE = args.num_ratio_miss_boke

RESULT_DIR = f"../../results/Boke_Judge/{EXPERIENCE_NUMBER}/"
if not os.path.exists("../../results/Boke_Judge/"):
    os.mkdir("../../results/Boke_Judge/")
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)
print(f"result directory: {RESULT_DIR}")

# データセットの作成
if not os.path.exists(f"{RESULT_DIR}test_caption_datas.json") or RESET_DATA:
    
    boke_datas = list()
    caption_datas = list()

    max_num_boke = 0
    for JP in tqdm(os.listdir(DATA_DIR)):
        N = int(JP.split(".")[0])

        with open(f"{DATA_DIR}{JP}", "r") as f:
            a = json.load(f)

        image_information = a["image_information"]
        is_photographic_probability = image_information["is_photographic_probability"]
        ja_caption = image_information["ja_caption"]
        ocr = image_information["ocr"]

        # 現実写真以外を除去
        if not USE_UNREAL_IMAGE:
            if is_photographic_probability < 0.8: continue
            
        # 文字のある画像を除去
        if not USE_WORD_IMAGE:
            if len(ocr) != 0: continue

        bokes = a["bokes"]

        max_num_boke = max(max_num_boke, len(a["bokes"]))
        for i, B in enumerate(bokes):

            # 固有名詞を含む大喜利を除去
            if not USE_UNIQUE_NOUN_BOKE:
                if len(B["unique_nouns"]) != 0: continue

            boke_datas.append({
                "boke_number": i,
                "image_number": N
            })

        caption_datas.append({
            "caption_number": N,
            "image_number": N
        })

    # データセットの保存
    train_boke_datas, test_boke_datas = train_test_split(boke_datas, test_size = 0.01)
    train_caption_datas, test_caption_datas = train_test_split(caption_datas, test_size = 0.01)

    with open(f"{RESULT_DIR}train_boke_datas.json", "w") as f:
        json.dump(train_boke_datas, f)
    with open(f"{RESULT_DIR}train_caption_datas.json", "w") as f:
        json.dump(train_caption_datas, f)

    with open(f"{RESULT_DIR}test_boke_datas.json", "w") as f:
        json.dump(test_boke_datas, f)
    with open(f"{RESULT_DIR}test_caption_datas.json", "w") as f:
        json.dump(test_caption_datas, f)

# データセットの読み込み
else:
    with open(f"{RESULT_DIR}train_boke_datas.json", "r") as f:
        train_boke_datas = json.load(f)
    with open(f"{RESULT_DIR}train_caption_datas.json", "r") as f:
        train_caption_datas = json.load(f)

    with open(f"{RESULT_DIR}test_boke_datas.json", "r") as f:
        test_boke_datas = json.load(f)
    with open(f"{RESULT_DIR}test_caption_datas.json", "r") as f:
        test_caption_datas = json.load(f)

print(f"学習に用いる大喜利の数: {len(train_boke_datas)}\n", 
      f"学習に用いるキャプションの数: {len(train_caption_datas)}\n", 
      f"検証に用いる大喜利の数: {len(test_boke_datas)}\n", 
      f"検証に用いるキャプションの数: {len(test_caption_datas)}")

# モデルの学習
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BokeJudgeModel()
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATO)

for epoch in range(EPOCH):
    # train
    train_loss_obj = 0.0
    train_accuracy_obj = 0.0
    model.train()
    train_dataloader = make_dataloader(train_boke_datas, train_caption_datas,
                                       use_caption = USE_CAPTION, use_miss_boke = USE_MISS_BOKE, num_ratio_miss_boke = NUM_RATIO_MISS_BOKE,
                                       num_workers = NUM_WORKERS)
    pb = tqdm(train_dataloader, desc = f"Epoch {epoch+1}/{EPOCH}")
    
    for CIF, CSF, LSF, TS in pb:
        CIF = CIF.to(device)
        CSF = CSF.to(device)
        LSF = LSF.to(device)
        TS = TS.to(device)

        loss, accuracy = train_step(model, optimizer, (CIF, CSF, LSF), TS)
        train_loss_obj += loss
        train_accuracy_obj += accuracy
        pb.set_postfix({"train_loss": train_loss_obj / (pb.n + 1), "train_accuracy": train_accuracy_obj / (pb.n + 1)})

    train_loss = train_loss_obj / len(train_dataloader)
    train_accuracy = train_accuracy_obj / len(train_dataloader)

    # test
    test_loss_obj = 0.0
    test_accuracy_obj = 0.0
    model.eval()
    test_dataloader = make_dataloader(test_boke_datas, test_caption_datas,
                                      use_caption = USE_CAPTION, use_miss_boke = USE_MISS_BOKE, num_ratio_miss_boke = NUM_RATIO_MISS_BOKE,
                                       num_workers = NUM_WORKERS)
    pb = tqdm(test_dataloader, desc = "Evaluating")

    for CIF, CSF, LSF, TS in pb:
        CIF = CIF.to(device)
        CSF = CSF.to(device)
        LSF = LSF.to(device)
        TS = TS.to(device)
        
        loss, accuracy = evaluate(model, (CIF, CSF, LSF), TS)
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

    # 検証精度を更新した場合、重みを保存
    if max(test_accuracy_history) == test_accuracy:
        torch.save(model.state_dict(), f"{RESULT_DIR}best_model_weights.pth")

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

# テストデータでモデルを評価
evaluate_result_dict = evaluate_model(f"{RESULT_DIR}best_model_weights.pth",
                                      boke_data_path = f"{RESULT_DIR}test_boke_datas.json", 
                                      caption_data_path = f"{RESULT_DIR}test_caption_datas.json")
with open(f"{RESULT_DIR}evaluation_result.json", "w") as f:
    json.dump(evaluate_result_dict, f)

x = list()
caption_evaluations = list()
boke_evaluations = list()
miss_boke_evaluations = list()
for K, V in evaluate_result_dict.items():
    x.append(K)
    caption_evaluations.append(V["caption"])
    boke_evaluations.append(V["boke"])
    miss_boke_evaluations.append(V["miss_boke"])

fig = plt.figure(figsize = (5, 5))
ax = fig.add_subplot()
ax.plot(x, caption_evaluations, label = "caption", color = "red")
ax.plot(x, boke_evaluations, label = "boke", color = "blue")
ax.plot(x, miss_boke_evaluations, label = "miss boke", color = "green")
ax.scatter(x, caption_evaluations, color = "red")
ax.scatter(x, boke_evaluations, color = "blue")
ax.scatter(x, miss_boke_evaluations, color = "green")
ax.legend()
ax.grid()
ax.set_ylabel("accuracy")
ax.set_xlabel("threshold")

fig.savefig(f"{RESULT_DIR}evaluation_result.png")