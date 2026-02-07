"""
Joint Training for Hierarchical RL

Fine-tunes all three layers together with low learning rate
to improve coordination between layers.

Training stages:
1. Freeze strategic, train tactical with strategic rewards
2. Freeze strategic+tactical, train micro with tactical rewards
3. Unfreeze all, joint fine-tuning with low LR

Includes:
- Adaptive curriculum learning (difficulty scales with performance)
- Optional curiosity-driven exploration (RND)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

# Import layer modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.model import PolicyNetwork
from training.ppo import PPOAgent, PPOConfig
from tactical.model import TacticalNetwork
from tactical.train import TacticalPPOAgent, TacticalPPOConfig
from micro.model import MicroNetwork
from micro.train import MicroPPOAgent, MicroPPOConfig


class AdaptiveCurriculum:
    """
    Adaptive curriculum learning based on win rate.

    Automatically increases difficulty when agent performs well,
    and decreases when struggling. Prevents catastrophic forgetting
    by maintaining performance thresholds.
    """

    # Difficulty levels
    EASY = 0
    MEDIUM = 1
    HARD = 2
    BRUTAL = 3

    DIFFICULTY_NAMES = ['Easy', 'Medium', 'Hard', 'Brutal']

    def __init__(
        self,
        initial_difficulty: int = 0,
        window_size: int = 50,
        promote_threshold: float = 0.75,  # Win rate to increase difficulty
        demote_threshold: float = 0.35,   # Win rate to decrease difficulty
        min_games_before_change: int = 20  # Minimum games before difficulty can change
    ):
        self.difficulty = initial_difficulty
        self.window_size = window_size
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.min_games_before_change = min_games_before_change

        self.win_history = deque(maxlen=window_size)
        self.games_at_current_difficulty = 0
        self.difficulty_history = [(0, initial_difficulty)]  # (episode, difficulty)

    def update(self, won: bool, episode: int = 0) -> bool:
        """
        Update curriculum based on game outcome.

        Args:
            won: Whether the agent won
            episode: Current episode number (for logging)

        Returns:
            True if difficulty changed
        """
        self.win_history.append(1 if won else 0)
        self.games_at_current_difficulty += 1

        # Need enough games to make a decision
        if len(self.win_history) < self.min_games_before_change:
            return False

        if self.games_at_current_difficulty < self.min_games_before_change:
            return False

        win_rate = sum(self.win_history) / len(self.win_history)
        changed = False

        # Check for promotion
        if win_rate >= self.promote_threshold and self.difficulty < self.BRUTAL:
            self.difficulty += 1
            self.win_history.clear()
            self.games_at_current_difficulty = 0
            self.difficulty_history.append((episode, self.difficulty))
            changed = True
            print(f"[Curriculum] Difficulty INCREASED to {self.DIFFICULTY_NAMES[self.difficulty]} "
                  f"(win rate was {win_rate:.1%})")

        # Check for demotion
        elif win_rate <= self.demote_threshold and self.difficulty > self.EASY:
            self.difficulty -= 1
            self.win_history.clear()
            self.games_at_current_difficulty = 0
            self.difficulty_history.append((episode, self.difficulty))
            changed = True
            print(f"[Curriculum] Difficulty DECREASED to {self.DIFFICULTY_NAMES[self.difficulty]} "
                  f"(win rate was {win_rate:.1%})")

        return changed

    def get_ai_difficulty(self) -> int:
        """Get the game AI difficulty setting (0-2 for Easy/Medium/Hard)."""
        # Map our difficulty to game AI difficulty
        # BRUTAL uses Hard AI but with handicaps for us
        return min(self.difficulty, 2)

    def get_handicap(self) -> float:
        """Get handicap multiplier for BRUTAL difficulty (resources, etc.)."""
        if self.difficulty == self.BRUTAL:
            return 0.75  # We get 75% resources, enemy gets 125%
        return 1.0

    @property
    def current_difficulty_name(self) -> str:
        return self.DIFFICULTY_NAMES[self.difficulty]

    def get_stats(self) -> Dict:
        """Get curriculum statistics."""
        return {
            'difficulty': self.difficulty,
            'difficulty_name': self.current_difficulty_name,
            'win_rate': sum(self.win_history) / len(self.win_history) if self.win_history else 0.0,
            'games_at_difficulty': self.games_at_current_difficulty,
            'total_difficulty_changes': len(self.difficulty_history) - 1,
        }

    def save_state(self) -> Dict:
        """Save curriculum state for checkpointing."""
        return {
            'difficulty': self.difficulty,
            'win_history': list(self.win_history),
            'games_at_current_difficulty': self.games_at_current_difficulty,
            'difficulty_history': self.difficulty_history,
        }

    def load_state(self, state: Dict):
        """Load curriculum state from checkpoint."""
        self.difficulty = state['difficulty']
        self.win_history = deque(state['win_history'], maxlen=self.window_size)
        self.games_at_current_difficulty = state['games_at_current_difficulty']
        self.difficulty_history = state['difficulty_history']


@dataclass
class JointTrainingConfig:
    """Configuration for joint training."""

    # Learning rates (much lower for fine-tuning)
    strategic_lr: float = 1e-5
    tactical_lr: float = 5e-5
    micro_lr: float = 1e-4

    # Freeze settings
    freeze_strategic: bool = False
    freeze_tactical: bool = False
    freeze_micro: bool = False

    # Training params
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1  # Tighter clipping for fine-tuning
    entropy_coef: float = 0.005  # Lower entropy for exploitation

    # Reward weighting for hierarchical credit assignment
    # These weights control how rewards propagate upward through the hierarchy:
    # - micro rewards → tactical layer (scaled by micro_reward_weight)
    # - tactical rewards → strategic layer (scaled by tactical_reward_weight)
    #
    # Higher weights create stronger cross-layer learning signal.
    # Set to 0.0 for independent layer optimization (no propagation).
    strategic_reward_weight: float = 1.0   # Weight for strategic-layer terminal reward
    tactical_reward_weight: float = 0.5    # How much tactical success affects strategic learning
    micro_reward_weight: float = 0.3       # How much micro success affects tactical learning

    # Updates
    num_epochs: int = 3
    batch_size: int = 64

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_promote_threshold: float = 0.75
    curriculum_demote_threshold: float = 0.35

    # Curiosity-driven exploration (RND)
    use_curiosity: bool = False
    curiosity_coef: float = 0.5


class JointHierarchicalTrainer:
    """
    Trainer for joint fine-tuning of hierarchical RL layers.
    """

    def __init__(self,
                 strategic_agent: PPOAgent,
                 tactical_agent: TacticalPPOAgent,
                 micro_agent: MicroPPOAgent,
                 config: Optional[JointTrainingConfig] = None,
                 device: str = 'cpu'):

        self.config = config or JointTrainingConfig()
        self.device = torch.device(device)

        self.strategic = strategic_agent
        self.tactical = tactical_agent
        self.micro = micro_agent

        # LSTM hidden state tracking per unit (persists across episode steps)
        # Key: unit_id, Value: (hidden_state, cell_state) tuple
        self.micro_hidden_states: Dict[int, Tuple] = {}
        self.max_hidden_states = 256  # Limit memory usage

        # Override learning rates
        if not self.config.freeze_strategic:
            for param_group in self.strategic.optimizer.param_groups:
                param_group['lr'] = self.config.strategic_lr

        if not self.config.freeze_tactical:
            for param_group in self.tactical.optimizer.param_groups:
                param_group['lr'] = self.config.tactical_lr

        if not self.config.freeze_micro:
            for param_group in self.micro.optimizer.param_groups:
                param_group['lr'] = self.config.micro_lr

        # Override clip epsilon
        self.strategic.config.clip_epsilon = self.config.clip_epsilon
        self.tactical.config.clip_epsilon = self.config.clip_epsilon
        self.micro.config.clip_epsilon = self.config.clip_epsilon

        # Stats
        self.update_count = 0
        self.total_reward = 0.0

    def collect_episode(self, env) -> Tuple[float, Dict]:
        """
        Collect an episode with all three layers.

        Args:
            env: Environment that supports hierarchical control
                 (SimulatedHierarchicalEnv or real game env)

        Returns:
            (total_reward, info_dict)
        """
        # FIX: Clear LSTM hidden states at episode start for fresh temporal context
        self.micro_hidden_states.clear()

        strategic_state = env.reset()
        strategic_tensor = self._state_to_tensor(strategic_state, 'strategic')

        episode_reward = 0.0
        done = False
        step = 0
        info = {}

        # Get initial strategic action
        if not self.config.freeze_strategic:
            strategic_action, strategic_log_prob, strategic_value = \
                self.strategic.select_action(strategic_tensor)
        else:
            with torch.no_grad():
                strategic_action, _, _ = self.strategic.select_action(strategic_tensor)
            strategic_log_prob = torch.tensor(0.0)
            strategic_value = torch.tensor(0.0)

        while not done:
            step += 1

            # Tactical decisions for each team
            tactical_rewards = []
            team_states = env.get_team_states()

            for team_id, team_state in team_states.items():
                # Skip empty teams
                if team_state.get('empty'):
                    continue

                tactical_tensor = self._state_to_tensor(team_state, 'tactical')

                if not self.config.freeze_tactical:
                    tactical_action, tactical_log_prob, tactical_value = \
                        self.tactical.select_action(tactical_tensor)
                else:
                    with torch.no_grad():
                        tactical_action, _, _ = self.tactical.select_action(tactical_tensor)
                    tactical_log_prob = torch.tensor(0.0)
                    tactical_value = torch.tensor(0.0)

                # Micro decisions for units in team
                micro_rewards = []
                unit_states = env.get_unit_states(team_id)

                for unit_id, unit_state in unit_states.items():
                    micro_tensor = self._state_to_tensor(unit_state, 'micro')

                    if not self.config.freeze_micro:
                        # FIX: Restore hidden state for this unit if it exists
                        # Otherwise initialize new hidden state
                        if unit_id in self.micro_hidden_states:
                            self.micro.hidden = self.micro_hidden_states[unit_id]
                        else:
                            self.micro.reset_hidden(1)

                        micro_action, micro_log_prob, micro_value = \
                            self.micro.select_action(micro_tensor)

                        # FIX: Save hidden state for this unit after action
                        self.micro_hidden_states[unit_id] = self.micro.hidden

                        # Limit hidden state memory (LRU eviction)
                        if len(self.micro_hidden_states) > self.max_hidden_states:
                            oldest_key = next(iter(self.micro_hidden_states))
                            del self.micro_hidden_states[oldest_key]
                    else:
                        with torch.no_grad():
                            # Also maintain hidden state even when frozen
                            if unit_id in self.micro_hidden_states:
                                self.micro.hidden = self.micro_hidden_states[unit_id]
                            else:
                                self.micro.reset_hidden(1)
                            micro_action, _, _ = self.micro.select_action(micro_tensor)
                            self.micro_hidden_states[unit_id] = self.micro.hidden
                        micro_log_prob = torch.tensor(0.0)
                        micro_value = torch.tensor(0.0)

                    # Apply micro action
                    micro_reward = env.apply_micro_action(unit_id, self._action_to_dict(micro_action, 'micro'))
                    micro_rewards.append(micro_reward)

                    if not self.config.freeze_micro and len(self.micro.buffer) < 2048:
                        self.micro.store_transition(
                            micro_tensor, micro_action, micro_reward,
                            micro_value, micro_log_prob, done=False
                        )

                # Apply tactical action
                tactical_reward = env.apply_tactical_action(
                    team_id, self._action_to_dict(tactical_action, 'tactical')
                )
                # HIERARCHICAL CREDIT ASSIGNMENT:
                # Micro rewards propagate upward to tactical layer with weighted contribution.
                # This is intentional - tactical decisions that lead to good micro outcomes
                # (successful engagements, kiting, target selection) get reinforced.
                # The micro_reward_weight (default 0.3) controls the influence strength.
                # This creates hierarchical credit assignment: good tactical positioning
                # enables good micro, which feeds back to improve tactical learning.
                tactical_reward += self.config.micro_reward_weight * np.mean(micro_rewards) if micro_rewards else 0
                tactical_rewards.append(tactical_reward)

                if not self.config.freeze_tactical and len(self.tactical.buffer) < 2048:
                    self.tactical.store_transition(
                        tactical_tensor, tactical_action, tactical_reward,
                        tactical_value, tactical_log_prob, done=False
                    )

            # Step environment
            next_strategic_state, strategic_reward, done, info = env.step()
            # HIERARCHICAL CREDIT ASSIGNMENT:
            # Tactical rewards propagate upward to strategic layer.
            # This creates a multi-level credit assignment where strategic decisions
            # (what to build, when to attack) that enable good tactical execution
            # (team positioning, objective selection) get reinforced.
            # The tactical_reward_weight (default 0.5) controls influence strength.
            #
            # NOTE: If layers should be trained independently (no cross-layer feedback),
            # set tactical_reward_weight=0 and micro_reward_weight=0 in JointTrainingConfig.
            strategic_reward += self.config.tactical_reward_weight * np.mean(tactical_rewards) if tactical_rewards else 0

            next_strategic_tensor = self._state_to_tensor(next_strategic_state, 'strategic')
            episode_reward += strategic_reward

            if not self.config.freeze_strategic and len(self.strategic.buffer) < 2048:
                self.strategic.store_transition(
                    strategic_tensor, strategic_action, strategic_reward,
                    strategic_value, strategic_log_prob, done
                )

            strategic_tensor = next_strategic_tensor

            # Get new strategic action if continuing
            if not done:
                if not self.config.freeze_strategic:
                    strategic_action, strategic_log_prob, strategic_value = \
                        self.strategic.select_action(strategic_tensor)
                else:
                    with torch.no_grad():
                        strategic_action, _, _ = self.strategic.select_action(strategic_tensor)
                    strategic_log_prob = torch.tensor(0.0)
                    strategic_value = torch.tensor(0.0)

        self.total_reward += episode_reward
        return episode_reward, info

    def _action_to_dict(self, action, layer: str) -> Dict:
        """Convert action to dict for environment."""
        # If already a dict, just return it (agents return dicts)
        if isinstance(action, dict):
            return action

        if layer == 'tactical':
            if isinstance(action, torch.Tensor):
                if action.dim() == 0:
                    action_idx = action.item()
                else:
                    action_idx = action[0].item() if len(action) > 0 else 0
            else:
                action_idx = int(action)
            return {
                'action': int(action_idx) % 8,
                'target_x': 0.5,
                'target_y': 0.5,
                'attitude': 0.5
            }
        elif layer == 'micro':
            if isinstance(action, torch.Tensor):
                if action.dim() == 0:
                    action_idx = action.item()
                else:
                    action_idx = action[0].item() if len(action) > 0 else 0
            else:
                action_idx = int(action)
            return {
                'action': int(action_idx) % 11,
                'move_angle': 0.0,
                'move_distance': 0.3
            }
        return {}

    def update(self) -> Dict[str, Dict]:
        """
        Perform PPO updates on all unfrozen layers.

        Returns:
            Dict of stats for each layer
        """
        stats = {}

        # Get last values for GAE
        with torch.no_grad():
            _, _, strategic_last_value = self.strategic.select_action(
                torch.zeros(44)  # Dummy state
            )
            _, _, tactical_last_value = self.tactical.select_action(
                torch.zeros(64)
            )
            self.micro.reset_hidden(1)
            _, _, micro_last_value = self.micro.select_action(
                torch.zeros(32)
            )

        # Update layers
        if not self.config.freeze_strategic and len(self.strategic.buffer) > 0:
            stats['strategic'] = self.strategic.update(strategic_last_value)

        if not self.config.freeze_tactical and len(self.tactical.buffer) > 0:
            stats['tactical'] = self.tactical.update(tactical_last_value)

        if not self.config.freeze_micro and len(self.micro.buffer) > 0:
            stats['micro'] = self.micro.update(micro_last_value)

        self.update_count += 1
        return stats

    def _state_to_tensor(self, state: Dict, layer: str) -> torch.Tensor:
        """Convert state dict to tensor for specific layer."""
        if layer == 'strategic':
            from training.model import state_dict_to_tensor
            return state_dict_to_tensor(state)
        elif layer == 'tactical':
            # Handle SimulatedHierarchicalEnv team state format
            if 'strategy_embedding' in state:
                # Direct dict from sim env - convert to 64-dim tensor
                arr = []
                # Strategy embedding (8)
                arr.extend(state.get('strategy_embedding', [0.5] * 8))
                # Team composition - infantry (3)
                arr.extend([
                    state.get('team_count', 5) / 10.0,
                    state.get('team_health', 0.8),
                    1.0,  # ready
                ])
                # Team composition - vehicles (3)
                arr.extend([0.3, 0.8, 1.0])
                # Team composition - aircraft (3)
                arr.extend([0.0, 0.0, 0.0])
                # Team composition - mixed (3)
                arr.extend([0.0, 0.0, 0.0])
                # Team status (8)
                arr.extend([
                    state.get('team_health', 0.8),
                    1.0,  # ammo
                    state.get('cohesion', 0.8),
                    0.5,  # experience
                    0.5,  # dist_to_objective
                    0.3,  # dist_to_base
                    state.get('under_fire', 0.0),
                    0.0,  # has_transport
                ])
                # Situational (16)
                arr.extend(state.get('nearby_enemies', [0.1, 0.1, 0.1, 0.1]))
                arr.extend(state.get('nearby_allies', [0.2, 0.2, 0.2, 0.2]))
                arr.extend([
                    0.5,  # terrain_advantage
                    0.3,  # threat_level
                    0.5,  # target_value
                    0.5,  # supply_dist
                    1.0,  # retreat_path
                    0.0,  # reinforce_possible
                    0.0,  # special_ready
                    0.0,  # padding
                ])
                # Objective (8)
                arr.extend([
                    state.get('objective_type', 0) / 8.0,
                    state.get('objective_x', 0.5),
                    state.get('objective_y', 0.5),
                    0.5,  # priority
                    0.0,  # progress
                    0.0,  # time_on_objective
                    0.0, 0.0,  # padding
                ])
                # Temporal (4)
                arr.extend([0.5, 0.0, 0.5, 0.0])
                # Ensure exactly 64 dimensions
                while len(arr) < 64:
                    arr.append(0.0)
                return torch.tensor(arr[:64], dtype=torch.float32)
            else:
                from tactical.state import build_tactical_state
                tac_state = build_tactical_state(state, 0)
                return tac_state.to_tensor()
        elif layer == 'micro':
            # Handle SimulatedHierarchicalEnv unit state format
            if 'nearest_enemy_dist' in state:
                arr = [
                    state.get('unit_type', 0.5),
                    state.get('is_hero', 0.0),
                    state.get('veterancy', 0.0),
                    0.0,  # has_ability
                    state.get('health', 1.0),
                    0.0,  # shield
                    1.0,  # ammunition
                    0.0,  # cooldown
                    0.5,  # speed
                    0.5,  # range
                    state.get('dps', 0.5),
                    0.5,  # armor
                    state.get('nearest_enemy_dist', 1.0),
                    state.get('nearest_enemy_angle', 0.0),
                    state.get('nearest_enemy_health', 0.0),
                    0.3,  # enemy_threat
                    0.5,  # nearest_ally_dist
                    0.0,  # in_cover
                    state.get('under_fire', 0.0),
                    0.0,  # ability_ready
                    state.get('nearest_enemy_dist', 1.0),  # target_dist
                    state.get('nearest_enemy_health', 0.0),  # target_health
                    0.5,  # target_type
                    1.0,  # can_retreat
                    state.get('objective_type', 0.0),
                    state.get('objective_dir', 0.0),
                    state.get('team_role', 0.5),
                    state.get('priority', 0.5),
                    0.5,  # time_since_hit
                    0.0,  # time_since_shot
                    0.5,  # time_in_combat
                    0.0,  # movement_history
                ]
                return torch.tensor(arr[:32], dtype=torch.float32)
            else:
                from micro.state import build_micro_state
                mic_state = build_micro_state(state)
                return mic_state.to_tensor()
        else:
            raise ValueError(f"Unknown layer: {layer}")

    def save(self, path_prefix: str):
        """Save all models."""
        self.strategic.save(f"{path_prefix}_strategic.pt")
        self.tactical.save(f"{path_prefix}_tactical.pt")
        self.micro.save(f"{path_prefix}_micro.pt")

    def load(self, path_prefix: str):
        """Load all models."""
        self.strategic.load(f"{path_prefix}_strategic.pt")
        self.tactical.load(f"{path_prefix}_tactical.pt")
        self.micro.load(f"{path_prefix}_micro.pt")


def train_joint(
    strategic_checkpoint: str = None,
    tactical_checkpoint: str = None,
    micro_checkpoint: str = None,
    num_episodes: int = 1000,
    checkpoint_interval: int = 100,
    output_dir: str = 'checkpoints/joint',
    device: str = 'cpu',
    use_simulated: bool = True,
    use_curriculum: bool = True,
    use_curiosity: bool = False,
):
    """
    Joint fine-tuning of all three layers.

    Args:
        strategic_checkpoint: Path to strategic model (optional)
        tactical_checkpoint: Path to tactical model (optional)
        micro_checkpoint: Path to micro model (optional)
        num_episodes: Number of training episodes
        checkpoint_interval: Save checkpoint every N episodes
        output_dir: Directory for checkpoints
        device: Device to train on
        use_simulated: Use SimulatedHierarchicalEnv instead of real game
        use_curriculum: Enable adaptive curriculum learning
        use_curiosity: Enable RND curiosity-driven exploration
    """
    from .sim_env import SimulatedHierarchicalEnv

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Initializing agents...")

    # Initialize or load strategic
    strategic_config = PPOConfig(lr=1e-5, clip_epsilon=0.1, entropy_coef=0.005)
    strategic_agent = PPOAgent(strategic_config, device)
    if strategic_checkpoint and Path(strategic_checkpoint).exists():
        print(f"Loading strategic: {strategic_checkpoint}")
        strategic_agent.load(strategic_checkpoint)

    # Initialize or load tactical
    tactical_config = TacticalPPOConfig(lr=5e-5, clip_epsilon=0.1, entropy_coef=0.005)
    tactical_agent = TacticalPPOAgent(tactical_config, device)
    if tactical_checkpoint and Path(tactical_checkpoint).exists():
        print(f"Loading tactical: {tactical_checkpoint}")
        tactical_agent.load(tactical_checkpoint)

    # Initialize or load micro
    micro_config = MicroPPOConfig(lr=1e-4, clip_epsilon=0.1, entropy_coef=0.005)
    micro_agent = MicroPPOAgent(micro_config, device)
    if micro_checkpoint and Path(micro_checkpoint).exists():
        print(f"Loading micro: {micro_checkpoint}")
        micro_agent.load(micro_checkpoint)

    # Create joint trainer
    config = JointTrainingConfig(use_curriculum=use_curriculum, use_curiosity=use_curiosity)
    trainer = JointHierarchicalTrainer(
        strategic_agent, tactical_agent, micro_agent, config, device
    )

    # Initialize curriculum if enabled
    curriculum = None
    if use_curriculum:
        curriculum = AdaptiveCurriculum(
            initial_difficulty=0,
            promote_threshold=config.curriculum_promote_threshold,
            demote_threshold=config.curriculum_demote_threshold,
        )
        print(f"Curriculum learning enabled, starting at {curriculum.current_difficulty_name}")

    # Initialize curiosity if enabled
    curiosity = None
    if use_curiosity:
        from training.curiosity import RNDCuriosity
        curiosity = RNDCuriosity(
            state_dim=44,
            intrinsic_coef=config.curiosity_coef,
            device=device
        )
        print("Curiosity-driven exploration (RND) enabled")

    # Create environment
    if use_simulated:
        print("Using SimulatedHierarchicalEnv for training")
        env = SimulatedHierarchicalEnv(num_teams=3, units_per_team=5, episode_length=200)
    else:
        # Use real game environment for hierarchical training
        from .real_env import RealGameHierarchicalEnv
        print("Using RealGameHierarchicalEnv for training (requires game)")
        ai_difficulty = curriculum.get_ai_difficulty() if curriculum else 0
        env = RealGameHierarchicalEnv(
            headless=True,
            ai_difficulty=ai_difficulty,
            map_name="Alpine Assault",
        )

    print(f"Starting joint training for {num_episodes} episodes...")

    episode_rewards = []
    wins = 0

    for episode in range(num_episodes):
        # Update environment difficulty if using curriculum with real game
        if curriculum and not use_simulated:
            env.ai_difficulty = curriculum.get_ai_difficulty()

        reward, info = trainer.collect_episode(env)
        episode_rewards.append(reward)

        won = info.get('won', False)
        if won:
            wins += 1

        # Update curriculum
        if curriculum:
            curriculum.update(won, episode)

        # Update curiosity module
        if curiosity and len(trainer.strategic.buffer.states) > 0:
            states_tensor = torch.stack(trainer.strategic.buffer.states)
            curiosity.update(states_tensor)

        # Update all layers
        if (episode + 1) % 10 == 0:
            stats = trainer.update()

        # Log progress
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            win_rate = wins / (episode + 1) * 100

            log_msg = (f"Episode {episode+1}/{num_episodes} | Reward: {reward:.2f} | "
                       f"Avg: {avg_reward:.2f} | Win Rate: {win_rate:.1f}%")

            if curriculum:
                log_msg += f" | Difficulty: {curriculum.current_difficulty_name}"

            if curiosity:
                curiosity_stats = curiosity.get_stats()
                log_msg += f" | RND Loss: {curiosity_stats['rnd_avg_loss']:.4f}"

            print(log_msg)

        # Save checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = f"{output_dir}/joint_ep{episode+1}"
            trainer.save(checkpoint_path)

            # Save curriculum state
            if curriculum:
                import json
                with open(f"{checkpoint_path}_curriculum.json", 'w') as f:
                    json.dump(curriculum.save_state(), f)

            # Save curiosity module
            if curiosity:
                curiosity.save(f"{checkpoint_path}_curiosity.pt")

            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final models
    trainer.save(f"{output_dir}/joint_final")

    if curriculum:
        import json
        with open(f"{output_dir}/joint_final_curriculum.json", 'w') as f:
            json.dump(curriculum.save_state(), f)
        print(f"Final curriculum stats: {curriculum.get_stats()}")

    if curiosity:
        curiosity.save(f"{output_dir}/joint_final_curiosity.pt")
        print(f"Final curiosity stats: {curiosity.get_stats()}")

    print(f"\nTraining complete. Final models saved to {output_dir}/joint_final")

    return trainer


if __name__ == '__main__':
    print("Joint training module loaded.")
    print("\nTo use joint training:")
    print("1. Train strategic layer (already done, 81.5% win rate)")
    print("2. Train tactical layer: python -m tactical.train")
    print("3. Train micro layer: python -m micro.train")
    print("4. Run joint training with trained checkpoints")
    print("\nExample:")
    print("  trainer = train_joint(")
    print("      'checkpoints/strategic_best.pt',")
    print("      'checkpoints/tactical_best.pt',")
    print("      'checkpoints/micro_best.pt',")
    print("      num_episodes=500")
    print("  )")
