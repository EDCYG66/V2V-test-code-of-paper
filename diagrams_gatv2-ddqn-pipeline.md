graph TD
  %% 1. 图编码模块：Graph encoding（上下布局，符合数据流向）
  subgraph GRAPH["Graph Encoding (图编码模块)"]
    direction TB
    X["Input: Nodes/Edges/Comm Stats (输入：节点/边/通信统计)"]
    ENC["GATv2 Encoder (L layers, K heads) (GATv2编码器)"]
    H["Node Embeddings h_i (节点嵌入)"]
    EDGE_HEAD["Edge Head (边预测头)"]
    POOL["Global Pooling (全局池化)"]
    LINK_SCORE["Output: Link Score (输出：链接分数)"]
    GLOBAL_SUM["Output: Global Summary (输出：全局摘要)"]
    
    %% 图编码内部流程
    X --> ENC
    ENC --> H
    H --> EDGE_HEAD --> LINK_SCORE
    H --> POOL --> GLOBAL_SUM
  end

  %% 2. 状态融合模块：State fusion（单独模块，突出特征拼接）
  subgraph FUSE["State Fusion (状态融合模块)"]
    direction TB
    CAT["Concat Features (特征拼接)"]
    S_T["Output: State s_t (输出：状态s_t)"]
    
    %% 拼接内容明确列出，增强可读性
    CAT_note["- Node Embeddings<br>- Link Scores<br>- Global Summary<br>- Physics: d, dv, v, a, tau<br>- Comm: SNR, delay, loss, AoI"]
    CAT --> CAT_note
    CAT --> S_T
  end

  %% 3. DDQN强化学习模块：RL（左右布局，体现闭环逻辑）
  subgraph RL["DDQN (强化学习模块)"]
    direction LR
    QON["Q_online(s_t) → Q(s,·) (在线Q网络)"]
    EPS["Epsilon-Greedy (ε-贪心策略)"]
    ACT["Action Selection (选动作：RB/Power/Link)"]
    BUF["Replay Buffer (经验回放池)"]
    TD["DDQN Target + Huber Loss (TD更新+Huber损失)"]
    QTG["Q_target(s_t) (目标Q网络)"]
    
    %% DDQN内部核心流程
    QON --> EPS --> ACT
    BUF --> TD --> QON
    QON -. "Soft Update (软更新)" .-> QTG
  end

  %% 4. 环境交互：Env（闭环关键节点）
  ENV["Env Step (环境步)"]
  ENV -->|Output: Reward r, Next State s'| BUF

  %% 各模块间连接（核心数据流）
  LINK_SCORE --> CAT
  GLOBAL_SUM --> CAT
  S_T --> QON
  ACT --> ENV
