"""
Training monitoring and visualization.

This module provides the TrainingMonitor class for logging training progress
and generating visualization graphs during model training.
"""

import os
from typing import Dict, List
import matplotlib.pyplot as plt


class TrainingMonitor:
    """
    Monitors training progress and generates visualizations.
    
    Tracks losses, rewards, and other metrics during training,
    and generates PNG visualizations of training progress.
    
    Requirements:
        - 3.5: Include TrainingMonitor for logging and progress tracking
        - 3.10: Generate training progress graphs using TrainingMonitor
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize TrainingMonitor with log directory.
        
        Args:
            log_dir: Directory path where logs and plots will be saved
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Storage for step-level metrics
        self.step_metrics: Dict[str, List[float]] = {}
        self.step_numbers: List[int] = []
        
        # Storage for epoch-level metrics
        self.epoch_metrics: Dict[str, List[float]] = {}
        self.epoch_numbers: List[int] = []
    
    def log_step(self, metrics: Dict[str, float], step: int):
        """
        Log metrics for a training step.
        
        Args:
            metrics: Dictionary of metric names to values (e.g., {'loss': 0.5, 'reward': 2.3})
            step: Step number
        """
        self.step_numbers.append(step)
        
        for metric_name, value in metrics.items():
            if metric_name not in self.step_metrics:
                self.step_metrics[metric_name] = []
            self.step_metrics[metric_name].append(value)
    
    def log_epoch(self, metrics: Dict[str, float], epoch: int):
        """
        Log metrics for an epoch.
        
        Args:
            metrics: Dictionary of metric names to values (e.g., {'train_loss': 0.5, 'val_loss': 0.6})
            epoch: Epoch number
        """
        self.epoch_numbers.append(epoch)
        
        for metric_name, value in metrics.items():
            if metric_name not in self.epoch_metrics:
                self.epoch_metrics[metric_name] = []
            self.epoch_metrics[metric_name].append(value)
    
    def plot_metrics(self):
        """
        Generate and save metric plots as PNG files.
        
        Creates separate plots for step-level and epoch-level metrics.
        Each metric gets its own subplot in the visualization.
        """
        # Plot step-level metrics if available
        if self.step_metrics and self.step_numbers:
            self._plot_step_metrics()
        
        # Plot epoch-level metrics if available
        if self.epoch_metrics and self.epoch_numbers:
            self._plot_epoch_metrics()
    
    def _plot_step_metrics(self):
        """Generate plots for step-level metrics."""
        num_metrics = len(self.step_metrics)
        if num_metrics == 0:
            return
        
        # Create subplots
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
        if num_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for ax, (metric_name, values) in zip(axes, self.step_metrics.items()):
            ax.plot(self.step_numbers, values, label=metric_name)
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Step')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'step_metrics.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_epoch_metrics(self):
        """Generate plots for epoch-level metrics."""
        num_metrics = len(self.epoch_metrics)
        if num_metrics == 0:
            return
        
        # Create subplots
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
        if num_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for ax, (metric_name, values) in zip(axes, self.epoch_metrics.items()):
            ax.plot(self.epoch_numbers, values, marker='o', label=metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'epoch_metrics.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
