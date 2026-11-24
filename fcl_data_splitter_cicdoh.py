import os
import pickle
import random
import argparse
from collections import defaultdict
import numpy as np

from fcl_data_utils import count_nested_samples, finalize_fcl_splits

# ==========================================
# ⚙️ 核心配置: CIC_DOH (DNS-over-HTTPS 加密流量)
# ==========================================
INPUT_PKL = "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/CIC_DOH/multimodal_dataset/doh2020_multimodal_512.pkl"
OUTPUT_DIR = "./fcl_datasets_cicdoh"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 标签设定
NORMAL_LABEL = "Benign"
ATTACK_LABELS = ['dns2tcp', 'dnscat2', 'iodine']

# 每个 Task 中分配的样本数量 (保持 4:1 的极度类别不平衡)
ATTACK_PER_TASK = 500
BENIGN_PER_TASK = 2000

# ==========================================
# 🎬 场景生成函数 (支持 3/5/10 客户端)
# ==========================================
def generate_scenario(num_clients, attack_labels, seed=42):
    """
    为 N 个客户端生成联邦时空剧本。
    由于 CIC_DOH 只有 3 种攻击，每个客户端的 3 个 Task 将是这 3 种攻击的全排列。
    确保 Spatial Non-IID 和 Temporal Concept Drift。
    """
    random.seed(seed)
    np.random.seed(seed)
    
    clients = [f"Client_{chr(65+i)}" for i in range(num_clients)]
    scenario = {}
    
    for client in clients:
        # 对 3 种攻击进行随机打乱排列，分配给 3 个 Task
        selected = random.sample(attack_labels, 3)
        scenario[client] = {
            'Task_1': selected[0],
            'Task_2': selected[1],
            'Task_3': selected[2]
        }
    
    return scenario

# ==========================================
# 📌 固定的 3 客户端基准场景 (用于对比实验的绝对公平性)
# ==========================================
SCENARIO_ORIGINAL = {
    'Client_A': {'Task_1': 'dns2tcp', 'Task_2': 'dnscat2', 'Task_3': 'iodine'},
    'Client_B': {'Task_1': 'dnscat2', 'Task_2': 'iodine',  'Task_3': 'dns2tcp'},
    'Client_C': {'Task_1': 'iodine',  'Task_2': 'dns2tcp', 'Task_3': 'dnscat2'}
}

SCENARIO_REVERSE = {
    'Client_A': {'Task_1': 'iodine',  'Task_2': 'dnscat2', 'Task_3': 'dns2tcp'},
    'Client_B': {'Task_1': 'dns2tcp', 'Task_2': 'iodine',  'Task_3': 'dnscat2'},
    'Client_C': {'Task_1': 'dnscat2', 'Task_2': 'dns2tcp', 'Task_3': 'iodine'}
}

ALL_SCENARIOS_3CLIENTS = {
    "original": SCENARIO_ORIGINAL,
    "reverse": SCENARIO_REVERSE,
}

# ==========================================
# 🚀 核心切分与保存函数
# ==========================================
def split_fcl_data(scenario, output_suffix=""):
    print(f"\n{'='*60}")
    print(f"🔧 正在生成 CIC_DOH 数据集: {output_suffix}")
    print('='*60)
    
    if not os.path.exists(INPUT_PKL):
        raise FileNotFoundError(f"🚨 找不到输入文件: {INPUT_PKL}")

    print(f"📦 加载原始全量数据集: {INPUT_PKL} ...")
    with open(INPUT_PKL, 'rb') as f:
        raw_data = pickle.load(f)
        
    # 1. 归类样本
    data_by_label = defaultdict(list)
    for sample in raw_data:
        label = sample['label']
        data_by_label[label].append(sample)
        
    print("\n📊 CIC_DOH 原始标签分布:")
    for k, v in data_by_label.items():
        print(f"   - [{k:<15}]: {len(v)} 条")

    clients = list(scenario.keys())
    tasks = list(scenario[clients[0]].keys())
    num_clients = len(clients)
    print(f"\n🔍 客户端数: {num_clients}, 任务列表: {tasks}")

    # 2. 初始化容器
    fcl_data = {client: {task: [] for task in tasks} for client in clients}
    
    # 3. 分配背景加密流量 (Benign)
    benign_samples = data_by_label[NORMAL_LABEL]
    random.shuffle(benign_samples)
    total_needed_benign = num_clients * len(tasks) * BENIGN_PER_TASK
    
    if len(benign_samples) > total_needed_benign:
        benign_samples = benign_samples[:total_needed_benign]
    else:
        print(f"⚠️ 警告: Benign 样本不足！需要 {total_needed_benign}，实际仅有 {len(benign_samples)}")
        # 允许复用样本以应对背景流量不足的情况
        benign_samples = random.choices(benign_samples, k=total_needed_benign)
        
    idx = 0
    for client in clients:
        for task in tasks:
            fcl_data[client][task].extend(benign_samples[idx:idx + BENIGN_PER_TASK])
            idx += BENIGN_PER_TASK

    # 4. 分配加密隧道攻击流量 (Attack)
    attack_pointers = defaultdict(int)
    for client in clients:
        for task, attack_label in scenario[client].items():
            pool = data_by_label[attack_label]
            random.shuffle(pool)
            
            start_idx = attack_pointers[attack_label]
            end_idx = start_idx + ATTACK_PER_TASK
            
            # 如果样本不够，循环复用池中的样本
            if end_idx > len(pool):
                selected_attacks = pool[start_idx:] + random.choices(pool, k=end_idx - len(pool))
            else:
                selected_attacks = pool[start_idx:end_idx]
                
            fcl_data[client][task].extend(selected_attacks)
            attack_pointers[attack_label] = end_idx % len(pool)  # 更新指针
            print(f"   [{client}][{task}] 注入加密攻击 {attack_label}: {len(selected_attacks)} 条")

    # 5. 打乱 Task 内部顺序
    for client in clients:
        for task in tasks:
            random.shuffle(fcl_data[client][task])

    # 6. 统一切分 train/val/test 并保存到磁盘
    fcl_data = finalize_fcl_splits(fcl_data)
    output_pkl = os.path.join(OUTPUT_DIR, f"cicdoh_fcl_{num_clients}clients_{output_suffix}.pkl")
    with open(output_pkl, 'wb') as f:
        pickle.dump(fcl_data, f)
    
    print(f"✅ CIC_DOH 数据集已保存至: {output_pkl}")
    print(f"   总样本数: {count_nested_samples(fcl_data)}")
    print('='*60)

# ==========================================
# 🎮 主程序执行入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 CIC_DOH 联邦持续学习加密数据集")
    parser.add_argument("--num_clients", type=int, default=3, help="客户端数量 (3, 5, 10)")
    parser.add_argument("--scenario", type=str, default="all", 
                        choices=["original", "reverse", "random", "all"],
                        help="场景类型。对于 all，自动生成 3 客户端的所有固定对比场景")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    scenarios_to_run = {}
    
    if args.num_clients == 3:
        if args.scenario == "all":
            scenarios_to_run = ALL_SCENARIOS_3CLIENTS
        elif args.scenario in ALL_SCENARIOS_3CLIENTS:
            scenarios_to_run = {args.scenario: ALL_SCENARIOS_3CLIENTS[args.scenario]}
        else:
            scenarios_to_run = {"random": generate_scenario(3, ATTACK_LABELS, seed=args.seed)}
    else:
        # 对于 5 或 10 客户端，使用完全随机全排列场景
        scenarios_to_run = {f"random_seed{args.seed}": generate_scenario(args.num_clients, ATTACK_LABELS, seed=args.seed)}
    
    print(f"🚀 开始生成 CIC_DOH 联邦数据集 (目标客户端数: {args.num_clients})")
    for suffix, scenario in scenarios_to_run.items():
        split_fcl_data(scenario, suffix)
    
    print("\n🎉 所有 CIC_DOH 加密流量数据集生成完毕！请在您的 run_federated_cl_v2.py 中加载使用。")
