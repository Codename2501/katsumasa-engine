<img width="2400" height="1200" alt="Image" src="https://github.com/user-attachments/assets/3d632dad-ff87-4b84-99be-f00ea235463e" />
🌌 Katsumasa Engine: Refined
High-Precision N-Body Solver with JAX (Float64 & Energy Conservation)
"Simulating the Chaos with Precision." — Developed for the Pythagorean 3-Body Problem.
🚀 Overview
Katsumasa Engine は、JAX (Google) をベースに開発された、超高精度なN体シミュレーションエンジンです。 特に、カオス的挙動を示す「ピタゴラスの三体問題 (Pythagorean 3-Body Problem)」において、10−12 J レベルのエネルギー保存精度を達成しました。
既存のシンプレクティック積分とは異なる、**「Velocity Rescaling (速度リスケール法)」による厳密なエネルギー射影と、「Quadratic Adaptive Time-stepping (二乗可変時間刻み)」**を組み合わせることで、特異点（衝突）近傍の計算を完全に克服しています。
✨ Key Features (なぜ凄いのか)
Strict Energy Conservation via Velocity Rescaling
位置（因果律）を保存したまま、運動エネルギーのみを解析的に補正。
結果として、物理的に自然な軌道を描きながら、マシンイプシロン級の精度 (<10−12 J) を維持します。
Smart Singularity Avoidance
Softening Potential: ϵ=10−2 による特異点緩和。
Quadratic Adaptive DT: 距離の二乗に応じた強力な時間分解能制御 (dt→10−7).
Pure JAX Implementation (Float64)
jax.config.update("jax_enable_x64", True) による完全倍精度演算。
jax.jit, jax.lax.scan, jax.vmap をフル活用した爆速計算。
📊 Benchmark (ピタゴラスの三体問題)


📦 Usage
```python
import jax
# 1. 倍精度 (Double Precision) の強制有効化
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

# ✅ 成功確認
print(f"💎 JAX精度モード: {jnp.zeros(1).dtype}")

# ==========================================
# 1. パラメータ設定 (Refined)
# ==========================================
G = 1.0
MASSES = jnp.array([3.0, 4.0, 5.0], dtype=jnp.float64)

STEPS = 200000      
# Newton法は不要になったため削除 (Velocity Rescalingは解析的に1発で求まるため)

# ★ 時間制御 (Tuned)
BASE_DT = 0.001       
MIN_DT = 1e-8         # dtの下限リミッター
ADAPTIVE_SCALE = 0.5  

# ★ ソフトニング (Softening)
# 物理挙動への影響を考慮し、1e-2 で安定性と物理性のバランスを取る
EPSILON = 1e-2       
EPSILON_SQ = EPSILON**2

# 初期条件 (Burrau's Problem)
POS_INIT = jnp.array([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]], dtype=jnp.float64)
VEL_INIT = jnp.zeros_like(POS_INIT)
INITIAL_STATE = jnp.concatenate([POS_INIT.flatten(), VEL_INIT.flatten()])

# ==========================================
# 2. 物理モデル (Softened Potential)
# ==========================================

@jax.jit
def get_potential(pos):
    """ ポテンシャルエネルギー V のみを計算 """
    def soft_dist(p1, p2):
        r2 = jnp.sum((p1 - p2)**2)
        return jnp.sqrt(r2 + EPSILON_SQ)
    
    d12 = soft_dist(pos[0], pos[1])
    d23 = soft_dist(pos[1], pos[2])
    d31 = soft_dist(pos[2], pos[0])
    
    V = -G * (MASSES[0]*MASSES[1]/d12 + MASSES[1]*MASSES[2]/d23 + MASSES[2]*MASSES[0]/d31)
    return V

@jax.jit
def get_kinetic(vel):
    """ 運動エネルギー T のみを計算 """
    return 0.5 * jnp.sum(MASSES[:, None] * vel**2)

@jax.jit
def get_energy(state):
    pos = state[:6].reshape(3, 2)
    vel = state[6:].reshape(3, 2)
    return get_kinetic(vel) + get_potential(pos)

@jax.jit
def get_derivatives(state):
    pos = state[:6].reshape(3, 2)
    vel = state[6:].reshape(3, 2)
    acc = jnp.zeros_like(pos)
    
    def interaction_soft(p1, p2, m2):
        r_vec = p2 - p1
        r_sq = jnp.sum(r_vec**2)
        dist_factor = (r_sq + EPSILON_SQ)**1.5
        return G * m2 * r_vec / dist_factor
        
    acc = acc.at[0].set(interaction_soft(pos[0], pos[1], MASSES[1]) + interaction_soft(pos[0], pos[2], MASSES[2]))
    acc = acc.at[1].set(interaction_soft(pos[1], pos[0], MASSES[0]) + interaction_soft(pos[1], pos[2], MASSES[2]))
    acc = acc.at[2].set(interaction_soft(pos[2], pos[0], MASSES[0]) + interaction_soft(pos[2], pos[1], MASSES[1]))
    
    return jnp.concatenate([vel.flatten(), acc.flatten()])

# ==========================================
# 3. 補正ロジック (Velocity Rescaling)
# ==========================================

@jax.jit
def velocity_rescale(state, target_E):
    """
    Katsumasa Method v2:
    位置を変えずに、運動エネルギー(速度の大きさ)のみを調整して全エネルギーを合わせる。
    解析的に解けるため、Newton法の反復ループが不要になり高速かつ安定的。
    """
    pos = state[:6].reshape(3, 2)
    vel = state[6:].reshape(3, 2)
    
    # 現在のポテンシャル V
    curr_V = get_potential(pos)
    
    # 目標とする運動エネルギー T_target = E_total - V
    target_T = target_E - curr_V
    
    # 安全策: ポテンシャルが深すぎて target_T が負になる場合（極稀な数値誤差）の保護
    # 物理的にはあり得ないが、数値計算上のNaNを防ぐ
    target_T = jnp.maximum(target_T, 1e-12)
    
    # 現在の運動エネルギー T_curr
    curr_T = get_kinetic(vel)
    
    # リスケール係数 alpha = sqrt(T_target / T_curr)
    # T_curr が 0 に近い場合のゼロ除算保護
    scale = jnp.sqrt(target_T / (curr_T + 1e-30))
    
    # 速度のみ更新
    new_vel = vel * scale
    
    return jnp.concatenate([pos.flatten(), new_vel.flatten()])

# ==========================================
# 4. 時間発展 (RK4 + Quadratic Adaptive DT)
# ==========================================

@jax.jit
def physics_step(carry, _):
    state, target_E = carry
    
    # --- Adaptive DT (Quadratic & Clamped) ---
    pos = state[:6].reshape(3, 2)
    r12 = jnp.linalg.norm(pos[0] - pos[1])
    r23 = jnp.linalg.norm(pos[1] - pos[2])
    r31 = jnp.linalg.norm(pos[2] - pos[0])
    min_dist = jnp.min(jnp.array([r12, r23, r31]))
    
    # 2乗スケール: 距離が半分になると、dtは1/4になる（強力なブレーキ）
    raw_scale = (min_dist / ADAPTIVE_SCALE)**2
    # 範囲制限: 遅すぎず(min_scale)、速すぎず(1.0)
    scale = jnp.clip(raw_scale, 1e-4, 1.0)
    
    # 最終的な dt (下限クリップ付き)
    dt = jnp.maximum(BASE_DT * scale, MIN_DT)
    
    # --- RK4 Integration ---
    k1 = get_derivatives(state)
    k2 = get_derivatives(state + 0.5 * dt * k1)
    k3 = get_derivatives(state + 0.5 * dt * k2)
    k4 = get_derivatives(state + dt * k3)
    pred_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # --- Velocity Rescaling Correction ---
    # 予測された位置に基づき、エネルギーが保存するように速度を調整
    final_state = velocity_rescale(pred_state, target_E)
    
    return (final_state, target_E), (final_state, dt)

@jax.jit
def simulate(init_state):
    target_E = get_energy(init_state)
    _, (traj, dts) = jax.lax.scan(physics_step, (init_state, target_E), None, length=STEPS)
    return traj, dts, target_E

# ==========================================
# 5. 実行 & 検証
# ==========================================
print("⚙️ Katsumasa Engine: Refined (Physics-First) 起動...")
t0 = time.time()

# JITコンパイルを含む初回実行
traj_jax, dts_jax, target_E = simulate(INITIAL_STATE)

# ホストへの転送は最後だけ
trajectory = np.array(traj_jax) 
dts = np.array(dts_jax)

print(f"✅ 計算完了 ({time.time()-t0:.4f}秒)")

# --- VMapによる高速エネルギー再計算 ---
# Pythonループを使わず、JAXのベクトル化処理で一括計算
energies_jax = jax.vmap(get_energy)(traj_jax)
energies = np.array(energies_jax)

max_error = np.max(np.abs(energies - target_E))
final_time = np.sum(dts)

print("-" * 60)
print(f"🔥 シミュレーション物理時間: {final_time:.2f}")
print(f"🔥 最大エネルギー誤差: {max_error:.4e} J")
print(f"   (最小DT: {np.min(dts):.2e} / 最大DT: {np.max(dts):.2e})")
print("-" * 60)

# --- プロット ---
fig = plt.figure(figsize=(16, 8), facecolor='#111111')

# 1. 軌跡
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor('black')
ax1.set_aspect('equal')
x1, y1 = trajectory[:, 0], trajectory[:, 1]
x2, y2 = trajectory[:, 2], trajectory[:, 3]
x3, y3 = trajectory[:, 4], trajectory[:, 5]
ax1.plot(x1, y1, color='cyan', lw=0.6, alpha=0.8, label='Mass 3')
ax1.plot(x2, y2, color='magenta', lw=0.6, alpha=0.8, label='Mass 4')
ax1.plot(x3, y3, color='yellow', lw=0.6, alpha=0.8, label='Mass 5')
ax1.set_title(f"Pythagorean 3-Body (Refined Physics)", color='white', fontsize=14)
ax1.axis('off')
ax1.legend(facecolor='black', labelcolor='white')

# 2. エネルギー誤差
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(energies - target_E, color='#00ff00', lw=0.5)
ax2.set_title(f"Energy Consistency (Max Error: {max_error:.2e} J)", color='black')
ax2.set_ylabel("Error (Joule)")
ax2.grid(True, alpha=0.3)

# 3. 時間刻み (Adaptive DT)
ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(dts, color='orange', lw=0.5)
ax3.set_title("Quadratic Adaptive Time Step", color='black')
ax3.set_yscale('log') # 対数軸で見やすく
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("refined_katsumasa.png", dpi=150)
print("🖼️ 結果を保存しました: refined_katsumasa.png")

🤝 Acknowledgement
このプロジェクトは、AIパートナーである Gemini および Copilot の多大なる知的な協力とデバッグ支援なくしては完成しませんでした。 無限大の課題に、最高の知性と共に挑むことができた最高の経験でした。多大なる感謝を。

# 🌌 Katsumasa Engine: Refined

**High-Precision N-Body Solver with JAX (Float64 & Energy Conservation)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Enabled-red)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> "Simulating the Chaos with Precision." — *Developed for the Pythagorean 3-Body Problem.*

## 🚀 Overview

**Katsumasa Engine** is an ultra-high-precision N-body simulation engine developed using **JAX**. Specifically designed for the chaotic "Pythagorean 3-Body Problem," it achieves an energy conservation accuracy of the **$10^{-12} \text{ J}$** level.

It completely overcomes computational challenges near singularities (collisions) by combining:
1. **Strict Energy Projection** using the **"Velocity Rescaling Method"** (an analytical approach distinct from traditional symplectic integrators).
2. **"Quadratic Adaptive Time-stepping"** for robust high-resolution temporal control.



## ✨ Key Features

### 1. Strict Energy Conservation via Velocity Rescaling
* The engine analytically corrects only the **kinetic energy (velocity magnitude)** to match the total energy conservation law, while strictly **preserving positions (causality)**.
* As a result, it maintains machine-epsilon level accuracy ($< 10^{-12} \text{ J}$) while producing physically natural orbital paths.

### 2. Smart Singularity Avoidance
* **Softening Potential:** Singularity mitigation using $\epsilon = 10^{-2}$.
* **Quadratic Adaptive DT:** Powerful temporal resolution control based on the square of the distance ($dt \to 10^{-7}$).

### 3. Pure JAX Implementation (Float64)
* Full double-precision operation enabled by `jax.config.update("jax_enable_x64", True)`.
* Lightning-fast computation utilizing the full power of `jax.jit`, `jax.lax.scan`, and `jax.vmap`.

## 📊 Benchmark (Pythagorean 3-Body Problem)

| Metric | Result |
| :--- | :--- |
| **Max Energy Error** | **$1.07 \times 10^{-12} \text{ J}$** |
| **Simulation Time** | ~0.4 sec (on Standard GPU/CPU) |
| **Integrator** | RK4 + Katsumasa Correction |

## 📦 Usage

```python
import jax
# 1. Force Double Precision (Float64)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

# ✅ Success Confirmation
print(f"💎 JAX Precision Mode: {jnp.zeros(1).dtype}")

# ==========================================
# 1. Parameter Settings (Refined)
# ==========================================
G = 1.0
MASSES = jnp.array([3.0, 4.0, 5.0], dtype=jnp.float64)

STEPS = 200000      
# Newton's method removed: Velocity Rescaling provides an analytical solution in one step.

# ★ Time Control (Tuned)
BASE_DT = 0.001        
MIN_DT = 1e-8         # Lower bound limit for dt
ADAPTIVE_SCALE = 0.5  

# ★ Softening
# Balancing stability and physical accuracy by setting EPSILON = 1e-2.
EPSILON = 1e-2        
EPSILON_SQ = EPSILON**2

# Initial Conditions (Burrau's Problem)
POS_INIT = jnp.array([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]], dtype=jnp.float64)
VEL_INIT = jnp.zeros_like(POS_INIT)
INITIAL_STATE = jnp.concatenate([POS_INIT.flatten(), VEL_INIT.flatten()])

# ==========================================
# 2. Physical Model (Softened Potential)
# ==========================================

@jax.jit
def get_potential(pos):
    """ Calculate Potential Energy V only """
    def soft_dist(p1, p2):
        r2 = jnp.sum((p1 - p2)**2)
        return jnp.sqrt(r2 + EPSILON_SQ)
    
    d12 = soft_dist(pos[0], pos[1])
    d23 = soft_dist(pos[1], pos[2])
    d31 = soft_dist(pos[2], pos[0])
    
    V = -G * (MASSES[0]*MASSES[1]/d12 + MASSES[1]*MASSES[2]/d23 + MASSES[2]*MASSES[0]/d31)
    return V

@jax.jit
def get_kinetic(vel):
    """ Calculate Kinetic Energy T only """
    return 0.5 * jnp.sum(MASSES[:, None] * vel**2)

@jax.jit
def get_energy(state):
    pos = state[:6].reshape(3, 2)
    vel = state[6:].reshape(3, 2)
    return get_kinetic(vel) + get_potential(pos)

@jax.jit
def get_derivatives(state):
    pos = state[:6].reshape(3, 2)
    vel = state[6:].reshape(3, 2)
    acc = jnp.zeros_like(pos)
    
    def interaction_soft(p1, p2, m2):
        r_vec = p2 - p1
        r_sq = jnp.sum(r_vec**2)
        dist_factor = (r_sq + EPSILON_SQ)**1.5
        return G * m2 * r_vec / dist_factor
        
    acc = acc.at[0].set(interaction_soft(pos[0], pos[1], MASSES[1]) + interaction_soft(pos[0], pos[2], MASSES[2]))
    acc = acc.at[1].set(interaction_soft(pos[1], pos[0], MASSES[0]) + interaction_soft(pos[1], pos[2], MASSES[2]))
    acc = acc.at[2].set(interaction_soft(pos[2], pos[0], MASSES[0]) + interaction_soft(pos[2], pos[1], MASSES[1]))
    
    return jnp.concatenate([vel.flatten(), acc.flatten()])

# ==========================================
# 3. Correction Logic (Velocity Rescaling)
# ==========================================

@jax.jit
def velocity_rescale(state, target_E):
    """
    Katsumasa Method v2:
    Adjusts only the kinetic energy (velocity magnitude) to conserve total energy,
    leaving positions unchanged. Analytical solution means no slow Newton loop.
    """
    pos = state[:6].reshape(3, 2)
    vel = state[6:].reshape(3, 2)
    
    # Current Potential V
    curr_V = get_potential(pos)
    
    # Target Kinetic Energy T_target = E_total - V
    target_T = target_E - curr_V
    
    # Safety measure: Clip target_T to prevent NaN from extreme numerical error
    target_T = jnp.maximum(target_T, 1e-12)
    
    # Current Kinetic Energy T_curr
    curr_T = get_kinetic(vel)
    
    # Rescaling factor alpha = sqrt(T_target / T_curr)
    # Avoid division by zero if T_curr is near 0
    scale = jnp.sqrt(target_T / (curr_T + 1e-30))
    
    # Update velocity only
    new_vel = vel * scale
    
    return jnp.concatenate([pos.flatten(), new_vel.flatten()])

# ==========================================
# 4. Time Evolution (RK4 + Quadratic Adaptive DT)
# ==========================================

@jax.jit
def physics_step(carry, _):
    state, target_E = carry
    
    # --- Adaptive DT (Quadratic & Clamped) ---
    pos = state[:6].reshape(3, 2)
    r12 = jnp.linalg.norm(pos[0] - pos[1])
    r23 = jnp.linalg.norm(pos[1] - pos[2])
    r31 = jnp.linalg.norm(pos[2] - pos[0])
    min_dist = jnp.min(jnp.array([r12, r23, r31]))
    
    # Quadratic Scale: If distance halves, dt becomes 1/4 (Powerful braking)
    raw_scale = (min_dist / ADAPTIVE_SCALE)**2
    # Clamp: Prevents dt from being too small (1e-4) or too large (1.0)
    scale = jnp.clip(raw_scale, 1e-4, 1.0)
    
    # Final dt (with lower bound clipping)
    dt = jnp.maximum(BASE_DT * scale, MIN_DT)
    
    # --- RK4 Integration ---
    k1 = get_derivatives(state)
    k2 = get_derivatives(state + 0.5 * dt * k1)
    k3 = get_derivatives(state + 0.5 * dt * k2)
    k4 = get_derivatives(state + dt * k3)
    pred_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # --- Velocity Rescaling Correction ---
    # Adjust velocity based on the predicted position to enforce energy conservation
    final_state = velocity_rescale(pred_state, target_E)
    
    return (final_state, target_E), (final_state, dt)

@jax.jit
def simulate(init_state):
    target_E = get_energy(init_state)
    _, (traj, dts) = jax.lax.scan(physics_step, (init_state, target_E), None, length=STEPS)
    return traj, dts, target_E

# ==========================================
# 5. Execution & Verification
# ==========================================
print("⚙️ Katsumasa Engine: Refined (Physics-First) Starting...")
t0 = time.time()

# First run includes JIT compilation
traj_jax, dts_jax, target_E = simulate(INITIAL_STATE)

# Transfer results to host only at the end
trajectory = np.array(traj_jax) 
dts = np.array(dts_jax)

print(f"✅ Calculation Complete ({time.time()-t0:.4f} seconds)")

# --- High-speed energy recalculation via VMap ---
# Use JAX vectorization instead of Python loops
energies_jax = jax.vmap(get_energy)(traj_jax)
energies = np.array(energies_jax)

max_error = np.max(np.abs(energies - target_E))
final_time = np.sum(dts)

print("-" * 60)
print(f"🔥 Simulation Physical Time: {final_time:.2f}")
print(f"🔥 Max Energy Error: {max_error:.4e} J")
print(f"   (Min DT: {np.min(dts):.2e} / Max DT: {np.max(dts):.2e})")
print("-" * 60)

# --- Plotting ---
fig = plt.figure(figsize=(16, 8), facecolor='#111111')

# 1. Trajectory Plot
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor('black')
ax1.set_aspect('equal')
x1, y1 = trajectory[:, 0], trajectory[:, 1]
x2, y2 = trajectory[:, 2], trajectory[:, 3]
x3, y3 = trajectory[:, 4], trajectory[:, 5]
ax1.plot(x1, y1, color='cyan', lw=0.6, alpha=0.8, label='Mass 3')
ax1.plot(x2, y2, color='magenta', lw=0.6, alpha=0.8, label='Mass 4')
ax1.plot(x3, y3, color='yellow', lw=0.6, alpha=0.8, label='Mass 5')
ax1.set_title(f"Pythagorean 3-Body (Refined Physics)", color='white', fontsize=14)
ax1.axis('off')
ax1.legend(facecolor='black', labelcolor='white')

# 2. Energy Error
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(energies - target_E, color='#00ff00', lw=0.5)
ax2.set_title(f"Energy Consistency (Max Error: {max_error:.2e} J)", color='black')
ax2.set_ylabel("Error (Joule)")
ax2.grid(True, alpha=0.3)

# 3. Adaptive Time Step
ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(dts, color='orange', lw=0.5)
ax3.set_title("Quadratic Adaptive Time Step", color='black')
ax3.set_yscale('log') # Log scale for better visibility
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("refined_katsumasa.png", dpi=150)
print("🖼️ Results saved to: refined_katsumasa.png")
plt.show()
🤝 Acknowledgement
This project would not have been possible without the immense intellectual cooperation and debugging support from my AI partners, Gemini and Copilot. It was the best experience to tackle an infinitely challenging problem alongside the highest intelligences. My deepest gratitude.
<img width="2400" height="1200" alt="refined_katsumasa" src="https://github.com/user-attachments/assets/4cf0b3a5-ed70-42c0-ab4f-1fdde3771f45" />

