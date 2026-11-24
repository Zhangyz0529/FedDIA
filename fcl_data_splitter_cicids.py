"""
生成极度不平衡的联邦数据集（10:1、20:1、50:1、100:1）
用法： python fcl_data_splitter_imbalance.py
"""

import os
import pickle
import random
from collections import defaultdict
import numpy as np

from fcl_data_utils import count_nested_samples, finalize_fcl_splits

# ==========================================
# ⚙️ 核心配置
# ==========================================
INPUT_PKL = "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/CICIDS/multimodal_dataset/cicids_multimodal_512.pkl"
OUTPUT_DIR = "./fcl_datasets/imbalance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 标签映射
ATTACK_MAPPING = {
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS"
}
NORMAL_LABEL = "BENIGN"

# 每个 Task 中攻击样本数量（固定）
ATTACK_PER_TASK = 500

# 定义不同的 Normal:Attack 比例（Normal 样本数）
BENIGN_CONFIGS = {
    "4to1": 2000,     # 4:1
    "10to1": 5000,    # 10:1
    "20to1": 10000,   # 20:1
    "50to1": 25000,   # 50:1
    "100to1": 50000   # 100:1
}

# 使用原始顺序（你可以根据需要替换为其他场景）
SCENARIO = {
    'Client_A': {'Task_1': 'DoS',      'Task_2': 'DDoS',     'Task_3': 'PortScan'},
    'Client_B': {'Task_1': 'DDoS',     'Task_2': 'DoS',      'Task_3': 'FTP-Patator'},
    'Client_C': {'Task_1': 'PortScan', 'Task_2': 'FTP-Patator', 'Task_3': 'SSH-Patator'}
}

# ==========================================
# 核心切分函数（支持指定 BENIGN 数量）
# ==========================================
def split_imbalance_data(scenario, benign_per_task, output_suffix):
    """
    生成指定比例的数据集。
    """
    print(f"\n{'='*60}")
    print(f"🔧 正在生成数据集: {output_suffix} (Normal:Attack = {benign_per_task}:{ATTACK_PER_TASK})")
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
        
    print("\n📊 原始标签分布 (前5个):")
    for i, (k, v) in enumerate(data_by_label.items()):
        if i >= 5:
            break
        print(f"   - [{k:<15}]: {len(v)} 条")

    # 2. 解析客户端和任务
    clients = list(scenario.keys())
    tasks = list(scenario[clients[0]].keys())
    print(f"\n🔍 客户端: {clients}, 任务列表: {tasks}")

    # 3. 初始化容器
    fcl_data = {client: {task: [] for task in tasks} for client in clients}
    
    # 4. 分配 BENIGN（使用指定的比例）
    benign_samples = data_by_label[NORMAL_LABEL]
    random.shuffle(benign_samples)
    
    total_needed_benign = len(clients) * len(tasks) * benign_per_task
    if len(benign_samples) > total_needed_benign:
        benign_samples = benign_samples[:total_needed_benign]
    else:
        print(f"⚠️ 警告: BENIGN 样本不足！需要 {total_needed_benign}，实际只有 {len(benign_samples)}")
        # 如果不足，全部使用
        benign_samples = benign_samples[:]
    
    idx = 0
    for client in clients:
        for task in tasks:
            fcl_data[client][task].extend(benign_samples[idx : idx + benign_per_task])
            idx += benign_per_task

    # 5. 分配攻击流量（固定数量）
    attack_pointers = defaultdict(int)
    for client in clients:
        for task, attack_label in scenario[client].items():
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
    output_pkl = os.path.join(OUTPUT_DIR, f"cicids_fcl_3clients_3tasks_{output_suffix}.pkl")
    with open(output_pkl, 'wb') as f:
        pickle.dump(fcl_data, f)
    
    print(f"✅ 数据集已保存至: {output_pkl}")
    print(f"   总样本数: {count_nested_samples(fcl_data)}")
    print('='*60)

# ==========================================
# 主程序：批量生成所有不平衡数据集
# ==========================================
if __name__ == "__main__":
    print("🚀 开始生成极度不平衡联邦数据集...")
    for suffix, benign_count in BENIGN_CONFIGS.items():
        split_imbalance_data(SCENARIO, benign_count, suffix)
    print("\n✅ 所有不平衡数据集生成完毕！")
    print(f"📁 文件列表:")
    for suffix in BENIGN_CONFIGS.keys():
        print(f"   - {OUTPUT_DIR}/cicids_fcl_3clients_3tasks_{suffix}.pkl")
    print("\n💡 提示: 这些数据集用于极端类别不平衡鲁棒性实验。")
