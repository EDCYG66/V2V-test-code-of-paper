```mermaid
flowchart LR
  %% ========= 整体联动管线：GATv2 → 状态拼接 → DDQN =========
  subgraph GRAPH[图表示与GATv2编码]
    X[节点/边/通信特征] --> GAT[GATv2 Encoder (多头注意力×L)]
    GAT --> H[节点嵌入 h_i]
    H --> EDGE_CAT[拼接 h_u||h_v||e_uv → MLP_edge → 链路评分]
    H --> POOL[图级Pooling → 全局摘要]
  end

  subgraph FUSE[状态融合]
    CAT[拼接：\n节点/边嵌入摘要 + 物理量(速度/间距/误差)\n+ 通信统计(AoI/时延/丢包) + 全局摘要]
  end

  EDGE_CAT --> CAT
  POOL --> CAT

  subgraph RL[DDQN 决策]
    QNET[Q_online(s): MLP → Q(s,·)]
    TG[Q_target(s): 同结构]
    EPS[ε-greedy 执行动作]
    BUF[重放缓冲 PER]
    TD[DDQN 目标: y=r+γ·Q_target(s', argmax_a' Q_online(s',a'))]
    UPD[Huber 损失 + 软更新 τ]
  end

  CAT --> QNET --> EPS --> ACT[选择 RB×功率×链路 等动作]
  ACT --> ENV[通信/控制环境执行 → r,s']
  ENV --> BUF --> TD --> UPD --> QNET
  QNET -.-> TG
```