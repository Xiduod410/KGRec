import os
import random
from collections import defaultdict

random.seed(42)

# 路径设置
data_dir = "."
output_dir = os.path.join(data_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# 映射表
user2id = {}
item2id = {}
entity2id = {}
genre2id = {}
relation2id = {"interact": 0, "has-genre": 1}
user_hist = defaultdict(list)

# Step 1: 处理 ratings.dat
with open(os.path.join(data_dir, "ratings.dat"), "r", encoding="utf-8") as f:
    for line in f:
        user, movie, rating, _ = line.strip().split("::")
        if float(rating) < 4:  # 正反馈
            continue
        if user not in user2id:
            user2id[user] = len(user2id)
        if movie not in item2id:
            item2id[movie] = len(item2id)
        if movie not in entity2id:
            entity2id[movie] = len(entity2id)  # 电影也是实体
        user_hist[user2id[user]].append(item2id[movie])

# Step 2: 处理 movies.dat，生成电影-genre 的三元组
kg_triples = []
with open(os.path.join(data_dir, "movies.dat"), "r", encoding="ISO-8859-1") as f:
    for line in f:
        try:
            movie_id, title, genres = line.strip().split("::")
        except:
            continue  # 忽略异常行
        if movie_id not in item2id:
            continue  # 忽略未在评分中出现的电影
        movie_ent_id = entity2id[movie_id]
        for genre in genres.split("|"):
            if genre not in genre2id:
                genre2id[genre] = len(entity2id)
                entity2id[genre] = genre2id[genre]
            genre_ent_id = genre2id[genre]
            kg_triples.append((movie_ent_id, relation2id["has-genre"], genre_ent_id))

# Step 3: 写 relation_list.txt
with open(os.path.join(output_dir, "relation_list.txt"), "w") as f:
    for rel, idx in relation2id.items():
        f.write(f"{rel} {idx}\n")

# Step 4: 写 entity_list.txt
with open(os.path.join(output_dir, "entity_list.txt"), "w") as f:
    for ent, idx in entity2id.items():
        f.write(f"{ent} {idx}\n")

# Step 5: 写 item_list.txt
with open(os.path.join(output_dir, "item_list.txt"), "w") as f:
    for item, idx in item2id.items():
        f.write(f"{item} {idx} {item}\n")  # freebase_id 使用自身

# Step 6: 写 user_list.txt
with open(os.path.join(output_dir, "user_list.txt"), "w") as f:
    for user, idx in user2id.items():
        f.write(f"{user} {idx}\n")

# Step 7: 写 kg_final.txt（包括交互和has-genre）
with open(os.path.join(output_dir, "kg_final.txt"), "w") as f:
    # 用户-物品交互关系
    for user_id, items in user_hist.items():
        for item_id in items:
            f.write(f"{user_id} 0 {item_id}\n")
    # 电影-类型 三元组
    for h, r, t in kg_triples:
        f.write(f"{h} {r} {t}\n")

# Step 8: 写 train.txt / test.txt
with open(os.path.join(output_dir, "train.txt"), "w") as f_train, \
     open(os.path.join(output_dir, "test.txt"), "w") as f_test:
    for user_id, items in user_hist.items():
        if len(items) < 2:
            continue
        random.shuffle(items)
        split = int(0.8 * len(items))
        f_train.write(f"{user_id} " + " ".join(map(str, items[:split])) + "\n")
        f_test.write(f"{user_id} " + " ".join(map(str, items[split:])) + "\n")

print("✅ 处理完成，输出路径:", output_dir)
