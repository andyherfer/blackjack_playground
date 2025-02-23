from typing import Dict, List, Union
from black_jack import simulate_episodes, BlackjackConfig, GameStats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def create_results_visualization(results: GameStats) -> tuple[Figure, Axes]:
    """Create visualization of simulation results"""
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # Create histogram of withdrawals
    ax.hist(results["final_withdrawals"], bins=30)
    ax.set_title("Distribution of Final Withdrawals")
    ax.set_xlabel("Amount Withdrawn")
    ax.set_ylabel("Count")

    # Add reference lines
    ax.axvline(x=0, color="r", linestyle="--", label="Wiped Out")
    ax.axvline(x=1000, color="g", linestyle="--", label="First Goal")

    return fig, ax


if __name__ == "__main__":
    # Create config
    config = BlackjackConfig(
        num_hands=5,
        initial_balance=1000,
        min_bet=20,
        max_bet=1000,
    )

    # Run simulation
    results = simulate_episodes(config, num_episodes=1000)

    # Create and show visualization
    fig, ax = create_results_visualization(results)

    # Add statistics text
    total_hands = results["hands_won"] + results["hands_lost"] + results["hands_pushed"]
    stats_text = f"""
    Win rate: {results['hands_won']/total_hands*100:.1f}%
    Loss rate: {results['hands_lost']/total_hands*100:.1f}%
    Push rate: {results['hands_pushed']/total_hands*100:.1f}%
    Blackjack rate: {results['blackjacks']/total_hands*100:.2f}%
    """
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.legend()
    plt.show()
