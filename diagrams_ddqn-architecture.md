```mermaid
flowchart LR
  %% ========= DDQN 结构（在线Q/目标Q、重放与更新） =========
  subgraph INPUT[状态与动作空间]
    S[状态 s_t\n= GAT嵌入 + 物理量 + 通信统计(AoI/时延/丢包...)]
    A[离散动作空间\n例: RB×功率×边选择/开关]
  end

  S --> QON

  subgraph ONLINE[在线 Q 网络（Q_online）]
    direction TB
    QON[MLP: Dense→ReLU → Dense→ReLU → Dense|A|\n输出 Q(s_t, ·)]
    Argmax[贪心动作 a* = argmax_a Q_online(s_t,a)]
  end

  QON --> Argmax

  subgraph ACT[行为策略]
    EPS[ε-greedy 选择\n以 ε 概率随机，否则取 a*]
  end
  Argmax --> EPS
  A --> EPS

  EPS --> ENV[环境交互：执行动作 a_t\n获得 r_t, s_{t+1}, done]

  subgraph REPLAY[经验回放]
    MMB[存储 (s_t, a_t, r_t, s_{t+1}, done)\n优先经验回放（可选）]
  end

  ENV --> MMB
  MMB --> BATCH[采样小批量 transitions]

  BATCH --> TARGET

  subgraph TARGET[目标 Q 计算（DDQN）]
    direction TB
    QON_SP[Q_online(s_{t+1}, ·)]
    Argmax_SP[argmax_{a'} Q_online(s_{t+1}, a')]
    QTG[Q_target(s_{t+1}, a')]
    YT[y = r + γ·(1-done)·Q_target(s_{t+1}, argmax a')\n(Huber Loss)]
  end

  %% s' 经在线网选动作，再由目标网评估该动作的 Q
  BATCH --> QON_SP --> Argmax_SP --> QTG --> YT

  subgraph UPDATE[参数更新与软更新]
    LOSS[最小化 Huber(y, Q_online(s_t,a_t))]
    OPT[Adam/AdamW 学习率调度]
    TAU[目标网络软更新 τ≈0.005]
  end

  YT --> LOSS --> OPT --> QON
  QON -.周期软更新.-> TAU -.-> QTG

  %% 输出
  EPS -.在线推理动作.-> OUT[动作 a_t (用于资源调度/编队高层决策)]
```