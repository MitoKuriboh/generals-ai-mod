"""
Imitation Learning for Micro Layer

Behavior cloning to initialize the MicroNetwork from expert demonstrations.
This provides a good starting policy before PPO fine-tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Tuple
from pathlib import Path

from .model import MicroNetwork, MICRO_STATE_DIM, MICRO_ACTION_DIM
from .rules import RuleBasedMicro, KitingExpert, AggressiveExpert, collect_expert_demonstrations


class MicroImitationLearner:
    """
    Behavior cloning trainer for MicroNetwork.

    Learns to mimic expert demonstrations before PPO fine-tuning.
    """

    def __init__(self,
                 model: Optional[MicroNetwork] = None,
                 lr: float = 3e-4,
                 batch_size: int = 64,
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = model or MicroNetwork()
        self.model = self.model.to(self.device)

        self.lr = lr
        self.batch_size = batch_size

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Loss functions
        self.action_loss_fn = nn.CrossEntropyLoss()
        self.continuous_loss_fn = nn.MSELoss()

    def train(self,
              states: np.ndarray,
              actions: np.ndarray,
              angles: np.ndarray,
              distances: np.ndarray,
              num_epochs: int = 100,
              validation_split: float = 0.1) -> Dict[str, list]:
        """
        Train model on expert demonstrations.

        Args:
            states: [N, 32] state vectors
            actions: [N] discrete actions
            angles: [N] normalized angles (0-1)
            distances: [N] movement distances (0-1)
            num_epochs: Number of training epochs
            validation_split: Fraction for validation

        Returns:
            Training history dict
        """
        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        moves_t = torch.tensor(np.stack([angles, distances], axis=1), dtype=torch.float32)

        # Split train/val
        n = len(states)
        n_val = int(n * validation_split)
        indices = np.random.permutation(n)

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        train_dataset = TensorDataset(
            states_t[train_idx], actions_t[train_idx], moves_t[train_idx]
        )
        val_dataset = TensorDataset(
            states_t[val_idx], actions_t[val_idx], moves_t[val_idx]
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        history = {
            'train_loss': [],
            'train_action_acc': [],
            'val_loss': [],
            'val_action_acc': [],
        }

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_losses = []
            train_correct = 0
            train_total = 0

            for batch_states, batch_actions, batch_moves in train_loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_moves = batch_moves.to(self.device)

                # Reset hidden state for each batch
                self.model.reset_hidden(batch_states.size(0), self.device)

                # Forward pass
                action_logits, move_alpha, move_beta, _, _ = self.model.forward(batch_states)

                # Action loss (cross entropy)
                action_loss = self.action_loss_fn(action_logits, batch_actions)

                # Continuous loss (MSE on Beta distribution mode)
                move_pred = (move_alpha - 1) / (move_alpha + move_beta - 2 + 1e-8)
                move_pred = move_pred.clamp(0, 1)
                continuous_loss = self.continuous_loss_fn(move_pred, batch_moves)

                # Combined loss
                loss = action_loss + 0.5 * continuous_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_losses.append(loss.item())
                train_correct += (action_logits.argmax(dim=-1) == batch_actions).sum().item()
                train_total += batch_actions.size(0)

            # Validation
            self.model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_states, batch_actions, batch_moves in val_loader:
                    batch_states = batch_states.to(self.device)
                    batch_actions = batch_actions.to(self.device)
                    batch_moves = batch_moves.to(self.device)

                    self.model.reset_hidden(batch_states.size(0), self.device)
                    action_logits, move_alpha, move_beta, _, _ = self.model.forward(batch_states)

                    action_loss = self.action_loss_fn(action_logits, batch_actions)
                    move_pred = (move_alpha - 1) / (move_alpha + move_beta - 2 + 1e-8)
                    move_pred = move_pred.clamp(0, 1)
                    continuous_loss = self.continuous_loss_fn(move_pred, batch_moves)
                    loss = action_loss + 0.5 * continuous_loss

                    val_losses.append(loss.item())
                    val_correct += (action_logits.argmax(dim=-1) == batch_actions).sum().item()
                    val_total += batch_actions.size(0)

            # Record history
            history['train_loss'].append(np.mean(train_losses))
            history['train_action_acc'].append(train_correct / train_total)
            history['val_loss'].append(np.mean(val_losses))
            history['val_action_acc'].append(val_correct / val_total)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: "
                      f"train_loss={history['train_loss'][-1]:.4f}, "
                      f"train_acc={history['train_action_acc'][-1]:.1%}, "
                      f"val_loss={history['val_loss'][-1]:.4f}, "
                      f"val_acc={history['val_action_acc'][-1]:.1%}")

        return history

    def save(self, path: str):
        """Save trained model."""
        self.model.save(path)

    def evaluate(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
        """Evaluate model accuracy on test data."""
        self.model.eval()

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)

        with torch.no_grad():
            self.model.reset_hidden(states_t.size(0), self.device)
            action_logits, _, _, _, _ = self.model.forward(states_t)
            predicted = action_logits.argmax(dim=-1)

        accuracy = (predicted == actions_t).float().mean().item()

        # Per-action accuracy
        action_accuracies = {}
        for action_id in range(MICRO_ACTION_DIM):
            mask = actions_t == action_id
            if mask.sum() > 0:
                action_accuracies[action_id] = (
                    (predicted[mask] == action_id).float().mean().item()
                )

        return {
            'overall_accuracy': accuracy,
            'action_accuracies': action_accuracies,
        }


def train_from_experts(
    num_samples: int = 10000,
    num_epochs: int = 100,
    save_path: Optional[str] = None,
    device: str = 'cpu',
) -> Tuple[MicroNetwork, Dict]:
    """
    Train MicroNetwork by imitating multiple expert policies.

    Combines demonstrations from different expert styles for diversity.
    """
    print("Collecting expert demonstrations...")

    # Collect from multiple experts
    experts = [
        RuleBasedMicro(seed=1),
        KitingExpert(seed=2),
        AggressiveExpert(seed=3),
    ]

    all_states = []
    all_actions = []
    all_angles = []
    all_distances = []

    samples_per_expert = num_samples // len(experts)

    for i, expert in enumerate(experts):
        states, actions, angles, distances = collect_expert_demonstrations(
            expert, num_samples=samples_per_expert, seed=i * 100
        )
        all_states.append(states)
        all_actions.append(actions)
        all_angles.append(angles)
        all_distances.append(distances)

    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)
    angles = np.concatenate(all_angles)
    distances = np.concatenate(all_distances)

    print(f"Collected {len(states)} total demonstrations")

    # Train
    print("\nTraining micro network...")
    model = MicroNetwork()
    learner = MicroImitationLearner(model=model, device=device)

    history = learner.train(
        states, actions, angles, distances,
        num_epochs=num_epochs
    )

    # Evaluate
    eval_results = learner.evaluate(states[:1000], actions[:1000])
    print(f"\nFinal accuracy: {eval_results['overall_accuracy']:.1%}")

    # Save
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        learner.save(save_path)
        print(f"Saved model to {save_path}")

    return model, history


if __name__ == '__main__':
    print("Testing imitation learning...")

    # Quick test with small sample
    model, history = train_from_experts(
        num_samples=2000,
        num_epochs=50,
        save_path=None,
        device='cpu'
    )

    print(f"\nFinal training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final action accuracy: {history['val_action_acc'][-1]:.1%}")

    # Test inference
    print("\nTesting inference...")
    model.reset_hidden(1)
    state = torch.randn(MICRO_STATE_DIM)
    action_dict, log_prob, value = model.get_action(state)

    from .model import MicroAction
    print(f"Action: {MicroAction.name(action_dict['action'].item())}")

    print("\nImitation learning test passed!")
