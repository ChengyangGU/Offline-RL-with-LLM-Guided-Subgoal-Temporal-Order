import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from collections import defaultdict
import random

class FourRoomEnv(gym.Env):
    """
    Custom FourRoom environment with fixed start in upper-left room and goal in bottom-right room.
    Each room is 5x5, total grid 11x11 with walls separating rooms and hallways (gaps) connecting them.
    Reward: +1 only when reaching the goal, 0 otherwise.
    """
    def __init__(self):
        self.grid_size = 11
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Discrete(self.grid_size ** 2)
        self.start_pos = (0, 0)  # Upper-left room
        self.goal_pos = (10, 10)  # Bottom-right room
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        # Define walls (impassable positions)
        self.walls = set()
        # Add all positions in row 5 as potential walls
        for c in range(self.grid_size):
            self.walls.add((5, c))
        # Add all positions in column 5 as potential walls
        for r in range(self.grid_size):
            self.walls.add((r, 5))

        # Remove the gap positions from the walls
        self.walls.remove((2, 5)) # Vertical gap top
        self.walls.remove((8, 5)) # Vertical gap bottom
        self.walls.remove((5, 2)) # Horizontal gap left
        self.walls.remove((5, 8)) # Horizontal gap right

        # Build graph for optimal policy (using networkx)
        self.positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in self.walls]
        self.G = nx.Graph()
        for pos in self.positions:
            self.G.add_node(pos)
        for pos in self.positions:
            r, c = pos
            for dr, dc in self.actions:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.positions:
                    self.G.add_edge(pos, (nr, nc))

        # Compute shortest path distances to goal
        self.dist_to_goal = nx.shortest_path_length(self.G, target=self.goal_pos)

        # Precompute optimal actions for each position
        self.optimal_actions = defaultdict(list)
        for pos in self.positions:
            if pos == self.goal_pos:
                continue
            # Ensure pos is in dist_to_goal before accessing (should be true if graph is connected)
            if pos in self.dist_to_goal:
                dist = self.dist_to_goal[pos]
                for a_idx, (dr, dc) in enumerate(self.actions):
                    nr, nc = pos[0] + dr, pos[1] + dc
                    if (nr, nc) in self.positions and self.dist_to_goal.get((nr, nc), float('inf')) == dist - 1:
                        self.optimal_actions[pos].append(a_idx)

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.current_pos = self.start_pos
        return self.pos_to_state(self.current_pos), {}

    def pos_to_state(self, pos):
        return pos[0] * self.grid_size + pos[1]

    def state_to_pos(self, state):
        return divmod(state, self.grid_size)

    def step(self, action):
        row, col = self.current_pos
        dr, dc = self.actions[action]
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size and (new_row, new_col) not in self.walls:
            self.current_pos = (new_row, new_col)
        terminated = self.current_pos == self.goal_pos
        reward = 1 if terminated else 0
        truncated = False  # No time limit
        return self.pos_to_state(self.current_pos), reward, terminated, truncated, {}

    def get_action(self, pos, epsilon=0.3):
        """
        Epsilon-greedy policy based on optimal actions.
        With prob epsilon, random action; else, random optimal action.
        """
        if random.random() < epsilon or not self.optimal_actions[pos]:
            return random.randint(0, 3)
        return random.choice(self.optimal_actions[pos])