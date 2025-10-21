import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
from scipy.optimize import curve_fit

# ====================================================================
# 设置绘图样式 (与原文件一致)
# ====================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm' 
plt.rcParams['axes.unicode_minus'] = False 
plt.style.use('seaborn-v0_8-whitegrid') 
plt.rcParams.update({'font.size': 12})
plt.rcParams['legend.framealpha'] = 0.8 


# ====================================================================
# H1 核心模型函数 (与原文件一致)
# ====================================================================

def calculate_steady_state(x, L, Deff, k_uptake, c0):
    """
    计算H1假设的理论稳态解析解 (Eq. 2.10) 
    """
    if k_uptake <= 0 or Deff <= 0:
        # 纯扩散情况
        return c0 * (L - x) / L
        
    lambda_val = np.sqrt(k_uptake / Deff) # 
    
    # 避免分母为0
    if np.sinh(lambda_val * L) < 1e-10:
        return np.zeros_like(x)
        
    numerator = np.sinh(lambda_val * (L - x))
    denominator = np.sinh(lambda_val * L)
    
    c_steady = c0 * (numerator / denominator)
    return c_steady

def simulate_pde_dynamic(
    L=100.0, T_total=8000.0, nx=101, nt=50000,
    Deff=1.0, K=3.0, k_uptake=5e-4, c0=10.0,
    plot_snapshot_count=6):
    """
    使用FDM模拟1D 扩散-吸附-摄取 PDE (Eq. 2.6) 
    """
    
    # 1. 初始化
    dx = L / (nx - 1)
    dt = T_total / nt
    x = np.linspace(0, L, nx)
    
    # 3. 浓度数组
    c = np.zeros(nx)      
    c_new = np.zeros(nx)  
    results = [] 
    
    # 确定快照索引
    plot_times = np.linspace(0, T_total, plot_snapshot_count)
    plot_indices = np.unique(np.append((plot_times / dt).astype(int), nt)) 

    # 4. FDM 迭代系数 (基于 Eq. 3.5) 
    # (1+K) dC/dt = Deff * d2C/dx2 - k_uptake * C
    # dC/dt = (Deff / (1+K)) * d2C/dx2 - (k_uptake / (1+K)) * C
    A = (Deff * dt) / (dx**2 * (1 + K))
    B = (k_uptake * dt) / (1 + K)
    
    for t_step in range(nt + 1):
        # 5. 边界条件 (Dirichlet BCs) 
        c[0] = c0 
        c[-1] = 0.0
        
        # 6. 内部空间点迭代
        for i in range(1, nx - 1):
            c_new[i] = c[i] + A * (c[i+1] - 2*c[i] + c[i-1]) - B * c[i]
            
        # 7. 更新数组
        c = c_new.copy()
        
        # 8. 存储结果
        if t_step in plot_indices:
            results.append((t_step * dt, c.copy()))
            
    return x, results, L, Deff, k_uptake, c0, K

# ====================================================================
# H2 核心模型函数 (*** 新增 ***)
# ====================================================================

def model_hill_response(c, K_d, n, inhibitory=False, G_max=1.0):
    """
    计算H2假设的Hill方程响应 
    K_d: 阈值 (半饱和浓度) 
    n: Hill系数 (陡峭度) 
    inhibitory: 是否为负调控 (Eq. 4.2)
    """
    c_n = np.power(c, n)
    K_d_n = np.power(K_d, n)
    
    # 避免除以0
    epsilon = 1e-12
    
    if inhibitory:
        # 抑制型 (负调控) Hill 方程 (Eq. 4.2)
        return G_max * (K_d_n / (K_d_n + c_n + epsilon))
    else:
        # 激活型 (正调控) Hill 方程 (Eq. 4.1)
        return G_max * (c_n / (K_d_n + c_n + epsilon))

# ====================================================================
# 绘图函数 (H1部分与原文件一致, H2部分为新增)
# ====================================================================

def plot_dynamics_H1(x, results, L, Deff, k_uptake, c0, K, save_path):
    """
    可视化 1: H1 梯度动态形成 (论文图1)
    """
    plt.figure(figsize=(8, 6)) 
    
    # 1. 绘制动态过程
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(results)))
    for i, (t, c) in enumerate(results):
        plt.plot(x, c, color=colors[i], alpha=0.8, 
                 label=f'$t$ = {t/1000.0:.1f} ks')
        
    # 2. 绘制稳态解
    c_steady = calculate_steady_state(x, L, Deff, k_uptake, c0) 
    lambda_val = np.sqrt(k_uptake / Deff) 
    
    plt.plot(x, c_steady, 'k--', linewidth=3, alpha=0.9,
             label=f'Steady-State Solution (Eq. 2.10)')

    # 3. Aesthetics
    plt.title(f'Fig 2. H1: Dynamic Gradient Formation ($K$ = {K:.1f})', fontsize=14)
    plt.xlabel('Spatial Position $x$ ($\mu$m)', fontsize=12)
    plt.ylabel('Free Concentration $c(x, t)$ (a.u.)', fontsize=12)
    plt.legend(fontsize=10, loc='lower right') 
    
    plt.text(L * 0.5, c0 * 0.9, 
             f'Decay Length: $\\lambda^{{-1}} \\approx {1/lambda_val:.1f}$ $\mu m$',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
             
    plt.ylim(bottom=-0.1, top=c0 * 1.05)
    plt.xlim(0, L)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'Vis1_H1_Dynamics.png'), dpi=900)
    plt.show()


def plot_decay_length_analysis_H1(L, c0, save_path):
    """
    可视化 2: H1 稳态参数敏感性 (D_eff, k_uptake) (论文图2) 
    """
    x = np.linspace(0, L, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # A. D_eff 的影响 
    k_uptake_fixed = 5e-4
    deff_variants = [0.1, 0.5, 1.0, 2.0]
    for Deff in deff_variants:
        c_steady = calculate_steady_state(x, L, Deff, k_uptake_fixed, c0)
        axes[0].plot(x, c_steady, label=f'$D_{{eff}}$ = {Deff:.1f} $\\mu m^2/s$', linewidth=2)
    axes[0].set_title('A. $D_{eff}$ Effect on Gradient Shape', fontsize=13)
    axes[0].set_xlabel('Position $x$ ($\mu$m)')
    axes[0].set_ylabel('Steady-State $c(x)$ (a.u.)')
    axes[0].legend(loc='upper right')

    # B. k_uptake 的影响 
    Deff_fixed = 1.0
    k_uptake_variants = [1e-4, 5e-4, 10e-4, 20e-4] 
    for k_uptake in k_uptake_variants:
        c_steady = calculate_steady_state(x, L, Deff_fixed, k_uptake, c0)
        axes[1].plot(x, c_steady, label=f'$k_{{uptake}}$ = {k_uptake*1e4:.0f}e-4 $s^{{-1}}$', linewidth=2)
    axes[1].set_title('B. $k_{uptake}$ Effect on Gradient Shape', fontsize=13)
    axes[1].set_xlabel('Position $x$ ($\mu$m)')
    axes[1].legend(loc='upper right')

    fig.suptitle('Fig 3. H1: $D_{eff}$ and $k_{uptake}$ Define Steady-State Shape', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'Vis2_H1_Decay_Length_Analysis.png'), dpi=900)
    plt.show()

def plot_k_formation_time_analysis_H1(L, T_total, nx, nt, Deff, k_uptake, c0, save_path):
    """
    可视化 3: H1 Adsorption (K) 效应 (T_1/2) (论文图3) 
    """
    x_mid_index = nx // 2
    x_mid = L / 2.0
    k_variants = [0.1, 3.0, 10.0, 20.0]
    
    plt.figure(figsize=(8, 6))
    
    # 理论稳态中点浓度 (与K无关) 
    c_steady_mid = calculate_steady_state(np.array([x_mid]), L, Deff, k_uptake, c0)[0]
    c_half = c_steady_mid / 2.0
    
    for K_val in k_variants:
        # 运行动态模拟
        _, sim_results_full, _, _, _, _, _ = simulate_pde_dynamic(
            L, T_total, nx, nt, Deff, K_val, k_uptake, c0, plot_snapshot_count=nt // 500
        )
        
        times = np.array([t for t, _ in sim_results_full])
        c_at_mid = np.array([c[x_mid_index] for _, c in sim_results_full])
        
        # 估计 T_half
        t_half_idx = np.searchsorted(c_at_mid, c_half)
        if t_half_idx >= len(times):
             t_half_label = f'$T_{{1/2}}$ > {T_total/1000.0:.1f} ks'
        else:
            t_half = times[t_half_idx]
            t_half_label = f'$T_{{1/2}}$ $\\approx {t_half/1000.0:.1f}$ ks'

        plt.plot(times / 1000.0, c_at_mid, 
                 label=f'$K$ = {K_val:.1f} ({t_half_label})', 
                 linewidth=2)

    plt.axhline(c_steady_mid, color='k', linestyle='--', alpha=0.7, 
                label=f'Steady-State $c(x_{{mid}})$ (K-independent)')
    
    plt.title(f'Fig 4. H1: Adsorption Coeff $K$ Controls Time-to-Steady-State ($T_{{1/2}}$)', fontsize=14)
    plt.xlabel('Time $t$ (ks)', fontsize=12)
    plt.ylabel('Concentration $c(x_{mid}, t)$ (a.u.)', fontsize=12)
    plt.legend(fontsize=10, loc='lower right')
    plt.ylim(bottom=-0.1)
    plt.xlim(0, T_total / 1000.0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'Vis3_H1_K_Effect_Time.png'), dpi=900)
    plt.show()

# ====================================================================
# *** H2 绘图函数 (新增) ***
# ====================================================================

def plot_h2_gene_expression(x, c_steady, c0, save_path):
    """
    可视化 4: H2 稳态梯度 -> 下游基因表达谱 (论文图4)
    """
    
    # 1. H2 Hill 方程参数
    K_d_A = 5.0  # 高阈值 (Gene A) 
    K_d_B = 2.0  # 中阈值 (Gene B) 
    K_d_C = 1.0  # 低阈值 (Gene C, 负调控)
    n_hill = 4   # Hill 系数, 产生陡峭响应 
    
    # 2. 计算基因表达谱
    gene_A = model_hill_response(c_steady, K_d_A, n_hill, inhibitory=False)
    gene_B = model_hill_response(c_steady, K_d_B, n_hill, inhibitory=False)
    gene_C = model_hill_response(c_steady, K_d_C, n_hill, inhibitory=True) #
    
    # 3. 绘图
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, 
                             gridspec_kw={'height_ratios': [1, 1]})
    
    # --- 图A: H1 梯度与阈值 ---
    axes[0].plot(x, c_steady, 'k-', linewidth=3, label='Protein X Gradient $c(x)$')
    
    # 绘制阈值线
    axes[0].axhline(K_d_A, color='r', linestyle='--', label=f'Gene A Threshold ($K_d={K_d_A}$)')
    axes[0].axhline(K_d_B, color='g', linestyle='--', label=f'Gene B Threshold ($K_d={K_d_B}$)')
    axes[0].axhline(K_d_C, color='b', linestyle='--', label=f'Gene C Threshold ($K_d={K_d_C}$)')
    
    axes[0].set_title('Fig 5A. H1: Steady-State Gradient and H2 Thresholds', fontsize=14)
    axes[0].set_ylabel('Concentration $c(x)$ (a.u.)', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_ylim(bottom=0, top=c0 * 1.05)

    # --- 图B: H2 基因表达域 ---
    axes[1].plot(x, gene_A, 'r-', linewidth=3, label='Gene A (High Threshold)')
    axes[1].plot(x, gene_B, 'g-', linewidth=3, label='Gene B (Mid Threshold)')
    axes[1].plot(x, gene_C, 'b-', linewidth=3, label='Gene C (Negative Reg.)')
    
    axes[1].fill_between(x, 0, gene_A, color='r', alpha=0.2)
    axes[1].fill_between(x, 0, gene_B, color='g', alpha=0.2)
    axes[1].fill_between(x, 0, gene_C, color='b', alpha=0.2)
    
    axes[1].set_title('Fig 5B. H2: Downstream Gene Expression Domains ($n=4$)', fontsize=14)
    axes[1].set_xlabel('Spatial Position $x$ ($\mu$m)', fontsize=12)
    axes[1].set_ylabel('Gene Expression $G(x)$ (a.u.)', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_xlim(0, L)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Vis4_H2_WildType_Expression.png'), dpi=900)
    plt.show()

def plot_h2_perturbations(x, L, Deff, k_uptake, save_path):
    """
    可视化 5: H2 因果验证 (LoF/GoF) (论文图5) 
    """
    
    # 1. 定义 H2 参数
    K_d_A = 5.0  # 高阈值
    K_d_C = 1.0  # 负调控阈值
    n_hill = 4.0
    
    # 2. 定义扰动
    c0_lof = 5.0  # LoF: 50% 浓度 
    c0_wt = 10.0 # Wild-Type
    c0_gof = 15.0 # GoF: 150% 浓度 
    
    # 3. 计算扰动后的梯度
    c_lof = calculate_steady_state(x, L, Deff, k_uptake, c0_lof)
    c_wt  = calculate_steady_state(x, L, Deff, k_uptake, c0_wt)
    c_gof = calculate_steady_state(x, L, Deff, k_uptake, c0_gof)
    
    # 4. 计算扰动后的基因表达
    gene_A_lof = model_hill_response(c_lof, K_d_A, n_hill)
    gene_A_wt  = model_hill_response(c_wt,  K_d_A, n_hill)
    gene_A_gof = model_hill_response(c_gof, K_d_A, n_hill)
    
    gene_C_lof = model_hill_response(c_lof, K_d_C, n_hill, inhibitory=True)
    gene_C_wt  = model_hill_response(c_wt,  K_d_C, n_hill, inhibitory=True)
    gene_C_gof = model_hill_response(c_gof, K_d_C, n_hill, inhibitory=True)
    
    # 5. 绘图
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    
    # --- 图A: 梯度扰动 (LoF/GoF) ---
    axes[0].plot(x, c_lof, 'b:', linewidth=2, label='LoF ($c_0 = 5.0$)')
    axes[0].plot(x, c_wt,  'k-', linewidth=3, label='Wild-Type ($c_0 = 10.0$)')
    axes[0].plot(x, c_gof, 'r--', linewidth=2, label='GoF ($c_0 = 15.0$)')
    axes[0].set_title('Fig 6A. H2 Causal Test: Gradient Perturbation', fontsize=14)
    axes[0].set_ylabel('Concentration $c(x)$ (a.u.)', fontsize=12)
    axes[0].legend(loc='upper right')

    # --- 图B: Gene A (高阈值) 响应 ---
    axes[1].plot(x, gene_A_lof, 'b:', linewidth=2, label='LoF (Domain Shrinks)')
    axes[1].plot(x, gene_A_wt,  'k-', linewidth=3, label='Wild-Type')
    axes[1].plot(x, gene_A_gof, 'r--', linewidth=2, label='GoF (Domain Expands)')
    axes[1].axhline(0.5, color='grey', linestyle=':', label='50% Expression Threshold')
    axes[1].set_title('Fig 6B. Response of High-Threshold Gene (Gene A)', fontsize=14)
    axes[1].set_ylabel('Gene A Expression (a.u.)', fontsize=12)
    axes[1].legend(loc='upper right')

    # --- 图C: Gene C (负调控) 响应 ---
    axes[2].plot(x, gene_C_lof, 'b:', linewidth=2, label='LoF (Domain Expands)')
    axes[2].plot(x, gene_C_wt,  'k-', linewidth=3, label='Wild-Type')
    axes[2].plot(x, gene_C_gof, 'r--', linewidth=2, label='GoF (Domain Shrinks)')
    axes[2].axhline(0.5, color='grey', linestyle=':', label='50% Expression Threshold')
    axes[2].set_title('Fig 6C. Response of Negatively-Regulated Gene (Gene C)', fontsize=14)
    axes[2].set_xlabel('Spatial Position $x$ ($\mu$m)', fontsize=12)
    axes[2].set_ylabel('Gene C Expression (a.u.)', fontsize=12)
    axes[2].legend(loc='center right')
    
    plt.xlim(0, L)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Vis5_H2_Perturbation_Analysis.png'), dpi=900)
    plt.show()


# ====================================================================
# 主执行模块 (Main)
# ====================================================================

if __name__ == "__main__":
    
    # 0. 文件保存设置
    SAVE_DIR = "Morphogen_Analysis_Plots_H1_H2"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"创建/检查目录: {SAVE_DIR}")
    
    # 1. H1 生物物理参数 (参考集)
    simulation_params = {
        'L': 100.0,         
        'T_total': 8000.0,  # 用于 Vis 1
        'nx': 101,          
        'nt': 50000,        # 用于 Vis 1
        'Deff': 1.0,        
        'K': 3.0,           
        'k_uptake': 5e-4,   
        'c0': 10.0,         
        'plot_snapshot_count': 6 
    }
    
    L, Deff, k_uptake, c0, K, nx = (
        simulation_params['L'], 
        simulation_params['Deff'], 
        simulation_params['k_uptake'], 
        simulation_params['c0'], 
        simulation_params['K'],
        simulation_params['nx']
    )
    
    # 2. 运行 H1 可视化
    print("\n--- 生成 Fig 1: H1 动态过程 (Vis1_H1_Dynamics.png) ---")
    x_coords, sim_results, _, _, _, _, _ = simulate_pde_dynamic(**simulation_params)
    plot_dynamics_H1(x_coords, sim_results, L, Deff, k_uptake, c0, K, SAVE_DIR)

    print("\n--- 生成 Fig 2: H1 稳态参数 (Vis2_H1_Decay_Length_Analysis.png) ---")
    plot_decay_length_analysis_H1(L=L, c0=c0, save_path=SAVE_DIR)
    
    T_total_K = 150000.0  # 增加总时间以观察K的效应
    nt_K = 1000000        #
    print("\n--- 生成 Fig 3: H1 $K$ 对 T_1/2 的影响 (Vis3_H1_K_Effect_Time.png) ---")
    plot_k_formation_time_analysis_H1(L=L, T_total=T_total_K, 
                           nx=nx, nt=nt_K, 
                           Deff=Deff, k_uptake=k_uptake, c0=c0, save_path=SAVE_DIR)
    
    # 3. 运行 H2 可视化 (*** 新增 ***)
    
    # 为H2分析生成高分辨率的x坐标
    x_coords_hires = np.linspace(0, L, 501)
    
    print("\n--- 生成 Fig 4: H2 稳态基因表达 (Vis4_H2_WildType_Expression.png) ---")
    c_steady_wt = calculate_steady_state(x_coords_hires, L, Deff, k_uptake, c0)
    plot_h2_gene_expression(x_coords_hires, c_steady_wt, c0, SAVE_DIR)
    
    print("\n--- 生成 Fig 5: H2 因果验证 LoF/GoF (Vis5_H2_Perturbation_Analysis.png) ---")
    plot_h2_perturbations(x_coords_hires, L, Deff, k_uptake, SAVE_DIR)
    
    print("\n所有 H1 和 H2 模拟分析图已生成并保存到 'Morphogen_Analysis_Plots_H1_H2' 文件夹。")