import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# ====================================================================
# 设置绘图样式
# ====================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm' 
plt.rcParams['axes.unicode_minus'] = False 
plt.style.use('seaborn-v0_8-whitegrid') 
plt.rcParams.update({'font.size': 12})
plt.rcParams['legend.framealpha'] = 0.8 

# ====================================================================
# H1 核心模型函数
# ====================================================================

def calculate_steady_state(x, L, D, k_uptake, c0):
    """计算H1假设的理论稳态解析解"""
    if k_uptake <= 0 or D <= 0:
        return c0 * (L - x) / L
    mu = np.sqrt(k_uptake / D)
    if np.sinh(mu * L) < 1e-10:
        return np.zeros_like(x)
    numerator = np.sinh(mu * (L - x))
    denominator = np.sinh(mu * L)
    c_steady = c0 * (numerator / denominator)
    return c_steady

# ====================================================================
# H2 核心模型函数
# ====================================================================

def model_hill_response(c, K_d, n, inhibitory=False, G_max=1.0):
    """计算H2假设的Hill方程响应"""
    c_n = np.power(c, n)
    K_d_n = np.power(K_d, n)
    epsilon = 1e-12
    if inhibitory:
        return G_max * (K_d_n / (K_d_n + c_n + epsilon))
    else:
        return G_max * (c_n / (K_d_n + c_n + epsilon))

# ====================================================================
# 新增的可视化函数
# ====================================================================

def plot_uniform_control(x, L, c0, save_path):
    """
    可视化 6: H2 反向实验 - 均匀浓度分布
    """
    K_d_A, K_d_B, K_d_C, n_hill = 5.0, 2.0, 1.0, 4.0
    c_uniform_val = K_d_B
    c_uniform = np.full_like(x, c_uniform_val)
    
    gene_A = model_hill_response(c_uniform, K_d_A, n_hill)
    gene_B = model_hill_response(c_uniform, K_d_B, n_hill)
    gene_C = model_hill_response(c_uniform, K_d_C, n_hill, inhibitory=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    
    axes[0].plot(x, c_uniform, 'k-', linewidth=3, label=f'Uniform Conc. $c(x) = {c_uniform_val}$')
    axes[0].axhline(K_d_A, color='r', linestyle='--', label=f'Gene A Threshold ($K_d={K_d_A}$)')
    axes[0].axhline(K_d_B, color='g', linestyle='--', label=f'Gene B Threshold ($K_d={K_d_B}$)')
    axes[0].axhline(K_d_C, color='b', linestyle='--', label=f'Gene C Threshold ($K_d={K_d_C}$)')
    axes[0].set_title('Fig 7A. Control Test: Uniform Concentration Profile', fontsize=14)
    axes[0].set_ylabel('Concentration $c(x)$ (a.u.)')
    axes[0].legend()
    axes[0].set_ylim(bottom=0, top=c0 * 0.6)

    axes[1].plot(x, gene_A, 'r-', linewidth=3, label='Gene A (Off)')
    axes[1].plot(x, gene_B, 'g-', linewidth=3, label='Gene B (50% On)')
    axes[1].plot(x, gene_C, 'b-', linewidth=3, label='Gene C (Partially Repressed)')
    axes[1].set_title('Fig 7B. Response: Loss of Spatial Patterning', fontsize=14)
    axes[1].set_xlabel('Spatial Position $x$ ($\mu$m)')
    axes[1].set_ylabel('Gene Expression $G(x)$ (a.u.)')
    axes[1].legend()
    axes[1].set_ylim(0, 1.2)
    axes[1].set_xlim(0, L)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Vis6_H2_Uniform_Control.png'), dpi=300)
    plt.show()

def plot_genetic_cascade(x, c_steady, L, save_path):
    """
    可视化 7: H2 下游基因级联
    """
    K_d_A, n_A = 4.0, 4.0
    K_d_D, n_D = 0.5, 8.0 # 基因D对基因A的响应更敏感 (n_D更大)
    
    gene_A = model_hill_response(c_steady, K_d_A, n_A)
    gene_D = model_hill_response(gene_A, K_d_D, n_D, inhibitory=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    
    axes[0].plot(x, c_steady, 'k-', linewidth=3, label='Morphogen X Gradient $c(x)$')
    axes[0].plot(x, gene_A, 'r-', linewidth=3, alpha=0.8, label='Gene A Expression $G_A(x)$')
    axes[0].axhline(K_d_A, color='k', linestyle=':', alpha=0.7, label=f'Gene A Activation Threshold ($K_d={K_d_A}$)')
    axes[0].set_title('Fig 8A. Primary Response: Morphogen Activates Gene A', fontsize=14)
    axes[0].set_ylabel('Concentration / Expression')
    axes[0].legend()

    axes[1].plot(x, gene_A, 'r-', linewidth=3, alpha=0.5, label='Inhibitor: Gene A Expression $G_A(x)$')
    axes[1].plot(x, gene_D, '-', color='m', linewidth=3, label='Output: Gene D Expression $G_D(x)$')
    axes[1].fill_between(x, 0, gene_D, color='m', alpha=0.2)
    axes[1].axhline(K_d_D, color='r', linestyle=':', label=f'Gene D Repression Threshold ($K_d={K_d_D}$)')
    axes[1].set_title('Fig 8B. Cascade: Gene A Represses Gene D, Forming a Stripe', fontsize=14)
    axes[1].set_xlabel('Spatial Position $x$ ($\mu$m)')
    axes[1].set_ylabel('Gene Expression (a.u.)')
    axes[1].legend(loc='upper right')
    axes[1].set_xlim(0, L)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Vis7_H2_Genetic_Cascade.png'), dpi=300)
    plt.show()

def plot_shape_perturbation(x, L, D, k_uptake, c0, save_path):
    """
    可视化 8: H2 通过改变梯度形状进行因果验证
    """
    k_uptake_wt = k_uptake
    k_uptake_steep = k_uptake * 2.5 # 陡峭
    k_uptake_shallow = k_uptake / 2.5 # 平缓
    
    c_wt = calculate_steady_state(x, L, D, k_uptake_wt, c0)
    c_steep = calculate_steady_state(x, L, D, k_uptake_steep, c0)
    c_shallow = calculate_steady_state(x, L, D, k_uptake_shallow, c0)
    
    K_d_A, n_A = 5.0, 4.0
    gene_A_wt = model_hill_response(c_wt, K_d_A, n_A)
    gene_A_steep = model_hill_response(c_steep, K_d_A, n_A)
    gene_A_shallow = model_hill_response(c_shallow, K_d_A, n_A)
    
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    
    axes[0].plot(x, c_wt, 'k-', linewidth=3, label='WT Gradient')
    axes[0].plot(x, c_shallow, 'b--', linewidth=2, label='Shallow Gradient (low $k_{uptake}$)')
    axes[0].plot(x, c_steep, 'r:', linewidth=2, label='Steep Gradient (high $k_{uptake}$)')
    axes[0].axhline(K_d_A, color='grey', linestyle='-.', label=f'Gene A Threshold ($K_d={K_d_A}$)')
    axes[0].set_title('Fig 9A. Causal Test: Perturbing Gradient Shape ($\lambda$)', fontsize=14)
    axes[0].set_ylabel('Concentration $c(x)$ (a.u.)')
    axes[0].legend()

    axes[1].plot(x, gene_A_wt, 'k-', linewidth=3, label='WT Domain')
    axes[1].plot(x, gene_A_shallow, 'b--', linewidth=2, label='Domain Expands')
    axes[1].plot(x, gene_A_steep, 'r:', linewidth=2, label='Domain Shrinks')
    axes[1].axhline(0.5, color='grey', linestyle=':', label='50% Expression Threshold')
    axes[1].set_title('Fig 9B. Response: Gene Expression Boundary Shifts', fontsize=14)
    axes[1].set_xlabel('Spatial Position $x$ ($\mu$m)')
    axes[1].set_ylabel('Gene A Expression (a.u.)')
    axes[1].legend()
    axes[1].set_xlim(0, L)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Vis8_H2_Shape_Perturbation.png'), dpi=300)
    plt.show()

# ====================================================================
# 主执行模块 (Main)
# ====================================================================
if __name__ == "__main__":
    
    SAVE_DIR = "Morphogen_Advanced_Analysis"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"创建/检查目录: {SAVE_DIR}")
    
    # 定义通用参数
    L = 100.0
    D = 1.0
    k_uptake = 5e-4
    c0 = 10.0
    x_coords = np.linspace(0, L, 501)
    
    # 计算稳态浓度
    c_steady_wt = calculate_steady_state(x_coords, L, D, k_uptake, c0)
    
    # --- 运行新增的可视化 ---
    print("\n--- 生成 Fig 6: H2 反向实验 (Vis6_H2_Uniform_Control.png) ---")
    plot_uniform_control(x_coords, L, c0, SAVE_DIR)
    
    print("\n--- 生成 Fig 7: H2 基因级联 (Vis7_H2_Genetic_Cascade.png) ---")
    plot_genetic_cascade(x_coords, c_steady_wt, L, SAVE_DIR)
    
    print("\n--- 生成 Fig 8: H2 梯度形状扰动 (Vis8_H2_Shape_Perturbation.png) ---")
    plot_shape_perturbation(x_coords, L, D, k_uptake, c0, SAVE_DIR)
    
    print(f"\n所有补充模拟分析图已生成并保存到 '{SAVE_DIR}' 文件夹。")
