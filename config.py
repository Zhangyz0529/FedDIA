import os

# ==========================================
# 📁 0. 全局路径配置
# ==========================================
BASE_DIR = "."
OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "result/")
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# ==========================================
# 🎯 1. 全局实验控制中心
# ==========================================
# 🔥 可选数据集: "CIC_IDS", "USTC", "CIC_DOH", 
#                 "CIC_IDS_10to1", "CIC_IDS_20to1", "CIC_IDS_50to1", "CIC_IDS_100to1"
# CURRENT_DATASET = os.environ.get("TARGET_DATASET", "CIC_IDS_WeakTask1")  # 默认使用 50:1 不平衡数据集
CURRENT_DATASET = os.environ.get("TARGET_DATASET", "CIC_IDS")  # 默认使用 Dirichlet 极度 Non-IID 数据集
COMPARE_DIR = f"${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/{CURRENT_DATASET}/compare/"
TOWER_MODE = "dual"

# ==========================================
# 🔥 基线与模型运行模式开关 (RUN_MODE)
# ==========================================
RUN_MODE = os.environ.get("RUN_MODE", "Ours")  # 默认使用 Ours

if RUN_MODE == "Ours":
    VERSION_TAG = "V30_Tabular_SE_Cosine_FCL"
else:
    VERSION_TAG = f"V30_Baseline_{RUN_MODE}"

SEQ_LENGTHS = [512]          
METRICS_LIST = ['F1', 'AUC', 'ACC', 'REC', 'PRE']

# ==========================================
# 🧠 1.1 持续学习 (Continual Learning) 专属配置
# ==========================================
CL_PARAMS = {
    'task_sequence': ['Task_1', 'Task_2', 'Task_3'] 
}

# ==========================================
# 📊 2. 各大数据集专属注册表 (Dataset Registry)
# ==========================================
DATASET_REGISTRY = {
    # New source-audited schema: deliberately distinct from legacy USTC25/DoH29.
    "CIC_DOH_SOURCE25": {
        "stat_dim": 25,
        "feature_schema": "ip_packet_stats25_v1",
        "normal_label": "Benign",
        "attack_classes": ['dns2tcp', 'dnscat2', 'iodine'],
        "fcl_data_path": os.environ.get("FCL_DATA_PATH", os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fcl_datasets", "doh_federated_v1_20261109", "dataset.pkl")),
        "compare_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_logs", "compare_doh_source25"),
        "baseline_suffix": "_doh_source25",
    },
    # ---------- 原始 CIC-IDS ----------
    "CIC_IDS": {
        "stat_dim": 77,  
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'], 
        "fcl_data_path": os.environ.get("FCL_DATA_PATH", "./fcl_datasets/cicids_fcl_3clients_3tasks_original_split.pkl"),
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/IDS-generation/new_compare/",
        "baseline_suffix": "",
    },
    # ---------- USTC ----------
    "USTC": {
        "stat_dim": 25,  
        "normal_label": "Benign",
        "attack_classes": ['Htbot', 'Neris', 'Miuref', 'Virut'],
        "fcl_data_path": "./fcl_datasets/ustc_fcl_3clients_3tasks.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/USTC/compare/",
        "baseline_suffix": "_ustc",
    },
    # ---------- CIC-DOH ----------
    "CIC_DOH": {
        "stat_dim": 29,
        "normal_label": "Benign",
        "attack_classes": ['dns2tcp', 'dnscat2', 'iodine'],
        "fcl_data_path": os.environ.get(
            "FCL_DATA_PATH", 
            "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/fcl_datasets_cicdoh/cicdoh_fcl_3clients_original.pkl"
        ),
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/CIC_DOH/compare/",
        "baseline_suffix": "_cicdoh"
    },
    # ============================================================
    # 🔥 新增：极端类别不平衡数据集 (用于大修实验)
    # ============================================================
    "CIC_IDS_10to1": {
        "stat_dim": 77,
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'],
        "fcl_data_path": "./fcl_datasets/imbalance/cicids_fcl_3clients_3tasks_10to1.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/experiment_logs/compare_imbalance/",
        "baseline_suffix": "_10to1",
    },
    "CIC_IDS_20to1": {
        "stat_dim": 77,
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'],
        "fcl_data_path": "./fcl_datasets/imbalance/cicids_fcl_3clients_3tasks_20to1.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/experiment_logs/compare_imbalance/",
        "baseline_suffix": "_20to1",
    },
    "CIC_IDS_50to1": {
        "stat_dim": 77,
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'],
        "fcl_data_path": "./fcl_datasets/imbalance/cicids_fcl_3clients_3tasks_50to1.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/experiment_logs/compare_imbalance/",
        "baseline_suffix": "_50to1",
    },
    "CIC_IDS_100to1": {
        "stat_dim": 77,
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'],
        "fcl_data_path": "./fcl_datasets/imbalance/cicids_fcl_3clients_3tasks_100to1.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/experiment_logs/compare_imbalance/",
        "baseline_suffix": "_100to1",
    },
    "CIC_IDS_WeakTask1": {
        "stat_dim": 77,
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'],
        "fcl_data_path": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/fcl_datasets/ablation/cicids_fcl_3clients_3tasks_weak_task1.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/experiment_logs/compare_weaktask1/",
        "baseline_suffix": "_weaktask1",
    },
    "CIC_IDS_Seq784": {
        "stat_dim": 77,
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'],
        "fcl_data_path": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/fcl_datasets/ablation/cicids_fcl_3clients_3tasks_784.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/experiment_logs/compare_seqlen/",
        "baseline_suffix": "_seq784",
    },
    "CIC_IDS_Seq1024": {
        "stat_dim": 77,
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'],
        "fcl_data_path": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/fcl_datasets/ablation/cicids_fcl_3clients_3tasks_1024.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/experiment_logs/compare_seqlen/",
        "baseline_suffix": "_seq1024",
    },
    # ============================================================
    # 🌪️ 新增：狄利克雷极度 Non-IID 数据集 (用于回应大修审稿人质疑 3)
    # ============================================================
    "CIC_IDS_Dirichlet_a01": {
        "stat_dim": 77,
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'],
        "fcl_data_path": "./fcl_datasets/dirichlet/cicids_fcl_3clients_3tasks_dirichlet_a01.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/experiment_logs/compare_dirichlet/",
        "baseline_suffix": "_dirichlet_a01",
    },
    "CIC_IDS_Dirichlet_a05": {
        "stat_dim": 77,
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'],
        "fcl_data_path": "./fcl_datasets/dirichlet/cicids_fcl_3clients_3tasks_dirichlet_a05.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/experiment_logs/compare_dirichlet/",
        "baseline_suffix": "_dirichlet_a05",
    },
    "CIC_IDS_Dirichlet_a50": {
        "stat_dim": 77,
        "normal_label": "BENIGN",
        "attack_classes": ['PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator', 'DoS'],
        "fcl_data_path": "./fcl_datasets/dirichlet/cicids_fcl_3clients_3tasks_dirichlet_a50.pkl",
        "compare_dir": "${WORKSPACE_ROOT}/TNSM/datasets/V_second_DSTA/Paper4_FCL/experiment_logs/compare_dirichlet/",
        "baseline_suffix": "_dirichlet_a50",
    },
}

# ==========================================
# ⚙️ 3. 硬件绑定与动态解析
# ==========================================
GPU_CPU_BINDINGS = {
    0: list(range(46, 58)), 
    1: [58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71]
}

CFG = DATASET_REGISTRY[CURRENT_DATASET].copy()
if os.environ.get("FCL_DATA_PATH"):
    CFG["fcl_data_path"] = os.environ["FCL_DATA_PATH"]
STUDY_PREFIX = f"{CURRENT_DATASET.lower()}_{TOWER_MODE}_{VERSION_TAG.lower()}"

# ==========================================
# 🔬 消融实验控制（通过环境变量覆盖）
# ==========================================
ABLATION = os.environ.get("ABLATION", "Full")

if __name__ == "__main__":
    print("\n" + "✅" * 20)
    print("  全局配置加载测试成功！")
    print("✅" * 20)
    print(f"\n🎯 当前激活的数据集 : {CURRENT_DATASET}")
    print(f"🚀 当前运行模式     : 【 {RUN_MODE} 】")  
    print(f"🏷️ 版本标签         : {VERSION_TAG}")
    print(f"🔬 消融配置         : {ABLATION}")
    print("-" * 50)
    print(f"📂 绑定的数据路径   : \n   {CFG['fcl_data_path']}")
    print(f"🦠 任务流顺序(CL)   : {CL_PARAMS['task_sequence']}")
    print(f"📏 统计特征维度     : {CFG['stat_dim']} 维")
    print("-" * 50 + "\n")
