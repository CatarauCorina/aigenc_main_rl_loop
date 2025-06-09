# AIGenC Main RL Loop

This repository implements the core reinforcement learning (RL) loop for the AIGenC framework—our computational model for functional creativity in RL agents, applied in the CREATE and Minigrid environment.

## 🔧 Installation

```bash
git clone https://github.com/CatarauCorina/aigenc_main_rl_loop.git
cd aigenc_main_rl_loop
pip install -r requirements.txt
```

How It Works

The agent encodes the current state into a structured graph.

It queries long-term memory using SETLE to retrieve relevant subgraphs.

Retrieved graphs are filtered (e.g., by attention) and added to working memory.

A deep RL agent takes enriched inputs and outputs an action.

The environment responds with a new state and reward.

The working memory is optionally updated, and the process continues.
