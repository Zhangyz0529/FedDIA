import os
import pickle
import random
from collections import defaultdict
import numpy as np

from fcl_data_utils import count_nested_samples, finalize_fcl_splits

# ==========================================
# ⚙️ USTC 专用核心配置
# ==========================================
INPUT_PKL = "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/USTC/multimodal_dataset/ustc_multimodal_512.pkl"
OUTPUT_DIR = "./fcl_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# USTC 的标签映射（无需合并，保持原样）
ATTACK_MAPPING = {}  # USTC 不需要像 CIC-IDS 那样合并 DoS
NORMAL_LABEL = "Benign"  # USTC 的正常流量标签是 "Benign"（不是 "BENIGN"）

# 每个 Task 中攻击样本数量
ATTACK_PER_TASK = 500
# 每个 Task 中 Benign 样本数量
BENIGN_PER_TASK = 2000

# ==========================================
# 🎬 USTC 联邦时空剧本
# ==========================================
FCL_SCENARIO_USTC = {
    'Client_A': {'Task_1': 'Neris',   'Task_2': 'Virut',   'Task_3': 'Htbot'},
    'Client_B': {'Task_1': 'Virut',   'Task_2': 'Htbot',   'Task_3': 'Miuref'},
    'Client_C': {'Task_1': 'Htbot',   'Task_2': 'Miuref',  'Task_3': 'Neris'}
}

# 确保所有攻击标签都在数据集中存在（使用你提供的 attack_classes）
AVAILABLE_ATTACKS = ['Htbot', 'Neris', 'Miuref', 'Virut']

# ==========================================
# 核心切分函数
# ==========================================
def split_fcl_data():
    print(f"\n{'='*60}")
    print("🔧 正在生成 USTC 联邦数据集（跨数据集泛化）")
    print('='*60)
    
    print(f"📦 加载原始全量数据集: {INPUT_PKL} ...")
    with open(INPUT_PKL, 'rb') as f:
        raw_data = pickle.load(f)
        
    # 1. 按标签归类
    data_by_label = defaultdict(list)
    for sample in raw_data:
        label = sample['label']
        if label in ATTACK_MAPPING:
            label = ATTACK_MAPPING[label]
            sample['label'] = label
        data_by_label[label].append(sample)
        
    print("\n📊 USTC 数据集标签分布:")
    # 只显示与攻击相关的标签
    all_labels = sorted(data_by_label.keys())
    for label in all_labels:
        if label == NORMAL_LABEL or label in AVAILABLE_ATTACKS:
            print(f"   - [{label:<15}]: {len(data_by_label[label])} 条")

    # 2. 解析客户端和任务
    clients = list(FCL_SCENARIO_USTC.keys())
    tasks = list(FCL_SCENARIO_USTC[clients[0]].keys())
    print(f"\n🔍 客户端: {clients}, 任务列表: {tasks}")

    # 3. 初始化容器
    fcl_data = {client: {task: [] for task in tasks} for client in clients}
    
    # 4. 分配 Benign（背景流量）
    benign_samples = data_by_label[NORMAL_LABEL]
    random.shuffle(benign_samples)
    
    total_needed_benign = len(clients) * len(tasks) * BENIGN_PER_TASK
    if len(benign_samples) > total_needed_benign:
        benign_samples = benign_samples[:total_needed_benign]
    else:
        print(f"⚠️ 警告: Benign 样本不足！需要 {total_needed_benign}，实际只有 {len(benign_samples)}")
        
    idx = 0
    for client in clients:
        for task in tasks:
            fcl_data[client][task].extend(benign_samples[idx : idx + BENIGN_PER_TASK])
            idx += BENIGN_PER_TASK

    # 5. 分配攻击流量（按照剧本）
    attack_pointers = defaultdict(int)
    for client in clients:
        for task, attack_label in FCL_SCENARIO_USTC[client].items():
            if attack_label not in data_by_label:
                print(f"⚠️ 警告: 攻击标签 {attack_label} 在数据集中不存在，跳过")
                continue
            pool = data_by_label[attack_label]
            random.shuffle(pool)
            start_idx = attack_pointers[attack_label]
            end_idx = min(start_idx + ATTACK_PER_TASK, len(pool))
            selected_attacks = pool[start_idx:end_idx]
            fcl_data[client][task].extend(selected_attacks)
            attack_pointers[attack_label] = end_idx
            print(f"   [{client}][{task}] 分配 {attack_label}: {len(selected_attacks)} 条")

    # 6. 打乱每个 Task 内部顺序
    for client in clients:
        for task in tasks:
            random.shuffle(fcl_data[client][task])

    # 7. 统一切分 train/val/test 并保存
    fcl_data = finalize_fcl_splits(fcl_data)
    output_pkl = os.path.join(OUTPUT_DIR, "ustc_fcl_3clients_3tasks.pkl")
    with open(output_pkl, 'wb') as f:
        pickle.dump(fcl_data, f)
    
    print(f"\n✅ USTC 联邦数据集已保存至: {output_pkl}")
    print(f"   总样本数: {count_nested_samples(fcl_data)}")
    
    # 打印每个客户端的任务分布
    print("\n📊 USTC 数据集分布:")
    for client in clients:
        print(f"   {client}:")
        for task in tasks:
            total_len = sum(len(fcl_data[client][task][split]) for split in ("train", "val", "test"))
            attack_label = FCL_SCENARIO_USTC[client].get(task, "Benign_only")
            print(f"      [{task}] {attack_label}: {total_len} 条")
    print('='*60)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("🚀 开始生成 USTC 跨数据集泛化实验数据集...")
    split_fcl_data()
    print("\n✅ USTC 联邦数据集生成完毕！")
    print("📁 文件: ./fcl_datasets/ustc_fcl_3clients_3tasks.pkl")
    print("💡 下一步: 修改 config.py 中的 fcl_data_path 指向此文件，然后运行跨数据集实验")
