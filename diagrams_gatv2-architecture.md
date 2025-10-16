```mermaid
flowchart LR
  %% ========= GATv2 编码器（节点/边/图级输出） =========
  %% 颜色/布局仅供示意，渲染时由 Mermaid 自动布局
  subgraph G0[图输入]
    X[节点特征 X (N×F)]
    E[边索引 edge_index (2×M)]
    FE[可选：边特征 e_uv (M×F_e)]
  end

  X --> A1
  E --> A1
  FE --> A1

  subgraph ENC[图注意力编码器 · GATv2]
    direction LR
    A1[GATv2 多头注意力 (K heads)\nConcat/Mean → ELU]
    R1[残差 + LayerNorm + Dropout]
    A2[GATv2 多头注意力 (K heads)\nConcat/Mean → ELU]
    R2[残差 + LayerNorm + Dropout]
  end

  A1 --> R1 --> A2 --> R2

  %% 节点级表示
  R2 --> H[节点嵌入 h_i (N×D)]

  %% 边级表示（用于链路/通信任务）
  subgraph EDGE_HEAD[边级/链路头]
    direction TB
    C1[拼接: h_u || h_v || e_uv]
    M1[MLP_edge(隐藏层+ReLU/ELU)]
    Oe[边级输出: 链路质量/成功率/评分]
  end

  H --> C1
  FE --> C1
  C1 --> M1 --> Oe

  %% 图级汇聚（可选：用于整体车队/路段表示）
  subgraph GRAPH_HEAD[图级/全局头（可选）]
    P1[全局Pooling\n(Mean/Max/Attention)]
    M2[MLP_graph]
    Og[图级输出: 交通状态/总体评分]
  end

  H --> P1 --> M2 --> Og

  %% 与控制/决策侧的接口
  H -.节点特征拼接.-> S1[状态向量拼接\n(物理量/通信统计/AoI...)]
  Oe -.链路特征拼接.-> S1
  Og -.全局摘要拼接.-> S1

  S1 --> OUT[输出给后续策略网络（如 DDQN/SAC）]
```