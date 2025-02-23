from typing import Dict, List, Union, Optional, TypedDict, Tuple
import random
from abc import ABC, abstractmethod
from policy import policy
from tqdm.auto import tqdm  # type: ignore  # Ignore missing stubs for tqdm
import matplotlib.pyplot as plt
import numpy as np


class GameStats(TypedDict):
    wiped_out: int
    first_goal: int
    beyond_first: int
    final_withdrawals: List[float]
    hands_won: int
    hands_lost: int
    hands_pushed: int
    blackjacks: int


class EpisodeStats(TypedDict):
    wins: int
    losses: int
    pushes: int
    blackjacks: int


class CardDeck:
    def __init__(self, config):
        self.config = config
        self.reset()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        if len(self.cards) < self.config.deck_penetration:
            self.reset()
        return self.cards.pop()

    def reset(self) -> None:
        """Reset and shuffle the deck"""
        # Each deck has 4 of each: A,2,3,4,5,6,7,8,9,10,J,Q,K
        self.cards = ([1] + list(range(2, 11)) + [10] * 3) * 4 * self.config.num_decks
        self.shuffle()


class Player(ABC):
    def __init__(self, name, balance, min_bet, max_bet):
        self.name = name
        self.hand = []
        self.total = 0
        self.balance = balance
        self.min_bet = min_bet
        self.max_bet = max_bet

    def draw_card(self, card):
        if card == 1:
            if self.total + 11 > 21:
                self.hand.append(1)
            else:
                self.hand.append(11)
        else:
            if self.total + card > 21:
                if 11 in self.hand:
                    self.hand.remove(11)
                    self.hand.append(1)
            self.hand.append(card)
        self.total = sum(self.hand)

    def reset(self):
        self.hand = []
        self.total = 0
        self.bet = 0

    def make_decision(self, dealer_card):
        """
        Determines the correct play based on the policy table.
        Args:
            dealer_card: The dealer's up card (2-11)
        Returns:
            str: The action to take ('Hit', 'Stand', 'Double', 'Split')
        """
        # NEVER hit on 21 or busted hands
        if self.total >= 21:
            return "Stand"

        hand_type = self.get_hand_type()

        # Convert policy codes to full action names
        action_map = {
            "H": "Hit",
            "S": "Stand",
            "D": "Double",
            "P": "Split",
        }

        try:
            if hand_type == "pair":
                # For pairs, use the individual card value
                card_value = self.hand[0]  # Both cards have same value
                action_code = policy[1][hand_type][card_value][dealer_card]
            else:
                # For other hands, use the total
                action_code = policy[1][hand_type][self.total][dealer_card]

            action = action_map[action_code]
        except KeyError:
            # If hand total is not in policy (e.g. bust), return Stand
            return "Stand"

        # Can't double after first two cards
        if len(self.hand) > 2 and action == "Double":
            action = "Hit"

        return action

    def get_hand_type(self):
        """
        Returns the type of hand the player has.
        Returns one of: 'hard', 'soft', or 'pair'
        """
        # Check for pairs first
        if len(self.hand) == 2 and self.hand[0] == self.hand[1]:
            return "pair"

        # Check for soft hands (contains an Ace counted as 11)
        if 11 in self.hand:
            return "soft"

        # Otherwise it's a hard hand
        return "hard"

    @abstractmethod
    def play(self, dealer_card, deck):
        """
        Play the hand against dealer's up card
        Args:
            dealer_card: The dealer's visible card
            deck: The card deck to draw from
        """
        pass

    @abstractmethod
    def make_bet(self):
        pass


class Dealer(Player):
    def __init__(self):
        super().__init__("Dealer", 0, 0, 0)

    def draw_card(self, card):
        """Override to handle Aces differently for dealer"""
        self.hand.append(card)
        # Recalculate total with Aces as 1 or 11
        total = 0
        aces = sum(1 for c in self.hand if c == 1)
        non_aces = sum(c for c in self.hand if c != 1)
        total = non_aces

        # Add aces, using them as 11 if possible
        for _ in range(aces):
            if total + 11 <= 21:
                total += 11
            else:
                total += 1

        self.total = total

    def play(self, dealer_card: int, deck: CardDeck) -> None:
        """Play dealer's hand according to fixed rules"""
        max_draws = 10  # Safety limit
        draws = 0

        while draws < max_draws:
            draws += 1
            # Stand on hard 17 or higher
            if self.total >= 17:
                if not (self.total == 17 and any(card == 11 for card in self.hand)):
                    break
            self.draw_card(deck.draw_card())

        # If we hit the safety limit, just keep the current hand

    def has_blackjack(self) -> bool:
        """Check if dealer has a natural blackjack"""
        return len(self.hand) == 2 and self.total == 21

    def make_bet(self):
        return 0

    def make_decision(self, dealer_card):
        return "Hit" if self.total < 17 else "Stand"


class ProgressivePlayer(Player):
    def __init__(self, name, balance, min_bet, max_bet, bet_multiplier=1.5):
        super().__init__(name, balance, min_bet, max_bet)
        self.bet_multiplier = bet_multiplier
        self.last_bet = 0
        self.last_result = None
        self.current_bet_size = self.min_bet
        self.base_bet_size = self.min_bet

    def make_bet(self):
        if self.last_result == "Win":
            self.current_bet_size = min(
                self.current_bet_size * self.bet_multiplier, self.max_bet
            )
        elif self.last_result == "Loss":
            self.current_bet_size = self.base_bet_size
        return self.current_bet_size


class MultiHandProgressivePlayer(ProgressivePlayer):
    def __init__(self, name, config):
        super().__init__(
            name,
            config.initial_balance,
            config.min_bet,
            config.max_bet,
            config.bet_multiplier,
        )
        self.config = config
        self.num_hands = config.num_hands
        self.hands = [[] for _ in range(config.num_hands)]
        self.hand_totals = [0] * self.num_hands
        self.initial_balance = config.initial_balance
        self.withdrawn = 0
        self.next_goal = config.initial_goal

        # Initialize progression state
        self.hand_progressions = []
        for _ in range(self.num_hands):
            self.hand_progressions.append(
                {"current_bet_size": config.min_bet, "last_result": None}
            )

    def reset_hands(self):
        """Reset all hands for a new round while preserving progression state"""
        # Only reset the cards and totals
        self.hands = [[] for _ in range(self.num_hands)]
        self.hand_totals = [0] * self.num_hands

    def make_bet_for_hand(self, hand_index):
        """Make bet for a specific hand based on its progression"""
        prog = self.hand_progressions[hand_index]

        # Base bet calculation based on last result
        if prog["last_result"] == "Win":
            # Increase bet after win
            new_bet = min(
                prog["current_bet_size"] * self.bet_multiplier,  # 1.5x after win
                self.max_bet,
                self.balance * self.config.max_bet_percent,  # Use config value
            )
            prog["current_bet_size"] = new_bet
        elif prog["last_result"] == "Loss" or prog["last_result"] is None:
            # Reset to minimum after loss or at start
            prog["current_bet_size"] = self.min_bet

        # Don't allow bet to exceed balance
        return min(prog["current_bet_size"], self.balance)

    def play_round(self, dealer_card, deck):
        """Play all hands against dealer's up card"""
        results = []
        bets = []  # Track bet size for each hand
        total_bet = 0
        max_hands = 20  # Safety limit for total hands after splits

        # Create dealer instance
        dealer = Dealer()
        dealer.draw_card(dealer_card)  # Give dealer their upcard

        # Make initial bets for all hands
        active_hands = 0
        for i in range(self.num_hands):
            bet_size = self.make_bet_for_hand(i)
            if bet_size > self.balance:
                # If can't afford bet, skip this hand
                continue
            self.balance -= bet_size
            total_bet += bet_size
            bets.append(bet_size)
            active_hands += 1

        # Deal initial cards
        for i in range(active_hands):
            self.hands[i] = []
            self.draw_card_to_hand(i, deck.draw_card())
            self.draw_card_to_hand(i, deck.draw_card())

        # Play each hand, with possible splits
        i = 0
        while i < len(self.hands) and len(self.hands) < max_hands:
            original_bet = (
                bets[i] if i < len(bets) else bets[-1]
            )  # Use last bet for splits
            result = play_hand(self, dealer, deck, i, original_bet, False)

            if result[0] == "Split":
                if self.balance >= original_bet:
                    # Create new hand from split
                    self.balance -= original_bet
                    total_bet += original_bet
                    bets.append(original_bet)

                    # Split the cards
                    split_card = self.hands[i].pop()
                    self.hand_totals[i] = self.hands[i][0]

                    # Create new hand
                    self.hands.append([split_card])
                    self.hand_totals.append(split_card)
                    self.hand_progressions.append(
                        {"current_bet_size": self.min_bet, "last_result": None}
                    )

                    # Draw new cards for both hands
                    self.draw_card_to_hand(i, deck.draw_card())
                    self.draw_card_to_hand(len(self.hands) - 1, deck.draw_card())
                    continue  # Stay on current hand
                else:
                    # If can't afford split, treat as regular hand
                    result = play_hand(self, dealer, deck, i, original_bet, False)

            results.append(result)
            i += 1

        return results, total_bet

    def draw_card_to_hand(self, hand_index, card):
        """Add a card to specific hand and update its total"""
        self.hands[hand_index].append(card)

        # Recalculate total with optimal Ace values
        total = 0
        aces = sum(1 for c in self.hands[hand_index] if c == 1)
        non_aces = sum(c for c in self.hands[hand_index] if c != 1)
        total = non_aces

        # Add aces, using them as 11 if possible
        for _ in range(aces):
            if total + 11 <= 21:
                total += 11
            else:
                total += 1

        self.hand_totals[hand_index] = total

    def make_decision_for_hand(self, hand_index, dealer_card):
        """Make decision for a specific hand"""
        self.hand = self.hands[hand_index]  # Temporarily set hand for decision making
        self.total = self.hand_totals[hand_index]
        decision = self.make_decision(dealer_card)
        return decision

    def check_goals(self):
        if self.balance >= self.next_goal:
            withdrawal = self.balance - self.initial_balance
            if self.balance > self.next_goal * self.config.withdrawal_threshold:
                withdrawal = withdrawal * self.config.withdrawal_amount_ratio
            self.withdrawn += withdrawal
            self.balance = self.initial_balance + (
                withdrawal * self.config.withdrawal_keep_ratio
            )
            self.next_goal += self.config.goal_increment
            return withdrawal
        return 0

    def play(self, dealer_card, deck):
        """
        Implementation of the abstract method from Player class.
        Delegates to play_round which handles multiple hands.
        """
        return self.play_round(dealer_card, deck)

    def update_hand_results(
        self, results: List[tuple[str, float]], dealer: Dealer
    ) -> None:
        """Update progression for each hand based on results"""
        for i, (result_code, bet) in enumerate(results):
            # First check for blackjack
            has_blackjack = len(self.hands[i]) == 2 and self.hand_totals[i] == 21
            if has_blackjack:
                if dealer.has_blackjack():
                    self.hand_progressions[i]["last_result"] = "Push"
                else:
                    self.hand_progressions[i]["last_result"] = "Win"
                continue

            # Regular hand comparison
            if result_code == "Bust":
                self.hand_progressions[i]["last_result"] = "Loss"
            elif dealer.total > 21:  # Check dealer bust FIRST
                self.hand_progressions[i]["last_result"] = "Win"
            else:
                # Only compare totals if dealer didn't bust
                player_total = self.hand_totals[i]
                if player_total > dealer.total:
                    self.hand_progressions[i]["last_result"] = "Win"
                elif player_total == dealer.total:
                    self.hand_progressions[i]["last_result"] = "Push"
                else:
                    self.hand_progressions[i]["last_result"] = "Loss"


class BlackjackConfig:
    def __init__(
        self,
        # Player parameters
        initial_balance=5000,
        min_bet=20,
        max_bet=1000,
        bet_multiplier=1.5,
        num_hands=3,
        # Betting strategy parameters
        aggressive_bet_chance=0.1,  # 10% chance of aggressive betting
        max_bet_percent=0.3,  # Don't bet more than 30% of current balance
        surplus_min_bet_multiplier=1.5,  # Bet 50% more than minimum when ahead
        aggressive_bet_high_multiplier=0.4,  # Use 40% of surplus when significantly ahead
        aggressive_bet_low_multiplier=0.2,  # Use 20% of surplus when moderately ahead
        significant_surplus_threshold=0.5,  # Consider 50% above initial as significant
        # Goal and withdrawal parameters
        initial_goal=2000,
        goal_increment=1000,
        withdrawal_threshold=1.2,  # Withdraw at 20% above goal
        withdrawal_keep_ratio=0.2,  # Keep 20% of withdrawal for continued play
        withdrawal_amount_ratio=0.8,  # Withdraw 80% when above threshold
        # Deck parameters
        num_decks=6,
        deck_penetration=20,  # Reshuffle when fewer than 20 cards remain
        hands_before_shuffle=52,
        min_balance_threshold=1.0,  # Stop if balance falls below this
    ):
        # Player parameters
        self.initial_balance = initial_balance
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.bet_multiplier = bet_multiplier
        self.num_hands = num_hands

        # Betting strategy parameters
        self.aggressive_bet_chance = aggressive_bet_chance
        self.max_bet_percent = max_bet_percent
        self.surplus_min_bet_multiplier = surplus_min_bet_multiplier
        self.aggressive_bet_high_multiplier = aggressive_bet_high_multiplier
        self.aggressive_bet_low_multiplier = aggressive_bet_low_multiplier
        self.significant_surplus_threshold = significant_surplus_threshold

        # Goal and withdrawal parameters
        self.initial_goal = initial_goal
        self.goal_increment = goal_increment
        self.withdrawal_threshold = withdrawal_threshold
        self.withdrawal_keep_ratio = withdrawal_keep_ratio
        self.withdrawal_amount_ratio = withdrawal_amount_ratio

        # Deck parameters
        self.num_decks = num_decks
        self.deck_penetration = deck_penetration
        self.hands_before_shuffle = hands_before_shuffle
        self.min_balance_threshold = min_balance_threshold


def simulate_episodes(
    config: BlackjackConfig, num_episodes: int = 10000, debug: bool = False
) -> GameStats:
    """Simulate multiple episodes of blackjack play."""
    results: GameStats = {
        "wiped_out": 0,
        "first_goal": 0,
        "beyond_first": 0,
        "final_withdrawals": [],
        "hands_won": 0,
        "hands_lost": 0,
        "hands_pushed": 0,
        "blackjacks": 0,
    }

    max_hands_per_episode = 10000
    max_balance_goal = 100_000

    for episode in tqdm(
        range(num_episodes), desc=f"Simulating {config.num_hands} hands"
    ):
        deck = CardDeck(config)
        dealer = Dealer()
        player = MultiHandProgressivePlayer("Player", config)
        hands_played = 0
        episode_stats = {"wins": 0, "losses": 0, "pushes": 0, "blackjacks": 0}
        total_withdrawn = 0  # Track only actual withdrawals, not final balance

        while (
            player.balance >= config.min_balance_threshold
            and hands_played < max_hands_per_episode
            and (total_withdrawn + player.balance) < max_balance_goal
        ):
            # Check if we've hit a goal
            withdrawal = player.check_goals()
            if withdrawal > 0:
                made_withdrawal = True
                total_withdrawn += withdrawal  # Only track actual withdrawals
                continue

            # Reset for new hand
            dealer.reset()
            player.reset_hands()
            hands_played += 1

            # Shuffle if needed
            if hands_played % config.hands_before_shuffle == 0:
                deck.reset()

            # Deal initial cards
            dealer_card = deck.draw_card()
            dealer.draw_card(dealer_card)

            # Track bets and results for this round
            round_bets = []
            round_results = []

            # Deal player's initial cards
            for hand_idx in range(player.num_hands):
                bet = player.make_bet_for_hand(hand_idx)
                if bet > player.balance:
                    continue

                # Deduct bet from balance
                player.balance -= bet
                round_bets.append(bet)

                # Deal initial cards
                player.draw_card_to_hand(hand_idx, deck.draw_card())
                player.draw_card_to_hand(hand_idx, deck.draw_card())

                # Play the hand
                result = play_hand(player, dealer, deck, hand_idx, bet, False)
                round_results.append(result)

            if not round_results:  # No bets were made
                break

            # Complete dealer's hand
            dealer.play(dealer_card, deck)

            # Process results
            player.update_hand_results(round_results, dealer)

            # Then process results for statistics
            for i, (result_code, bet) in enumerate(round_results):
                # First check for blackjack
                has_blackjack = (
                    len(player.hands[i]) == 2 and player.hand_totals[i] == 21
                )
                if has_blackjack:
                    if dealer.has_blackjack():
                        episode_stats["pushes"] += 1
                        player.balance += bet  # Return bet
                    else:
                        episode_stats["blackjacks"] += 1
                        episode_stats["wins"] += 1
                        player.balance += bet * 2.5  # Pay 3:2
                    continue

                # Then process regular hands
                if result_code == "Bust":
                    episode_stats["losses"] += 1
                    # Bet already deducted
                elif dealer.total > 21:  # Check dealer bust FIRST
                    episode_stats["wins"] += 1
                    player.balance += bet * 2
                else:
                    # Only compare totals if dealer didn't bust
                    player_total = player.hand_totals[i]
                    if player_total > dealer.total:
                        episode_stats["wins"] += 1
                        player.balance += bet * 2
                    elif player_total == dealer.total:
                        episode_stats["pushes"] += 1
                        player.balance += bet  # Return bet
                    else:
                        episode_stats["losses"] += 1
                        # Bet already deducted

        # Update overall results
        results["hands_won"] += episode_stats["wins"]
        results["hands_lost"] += episode_stats["losses"]
        results["hands_pushed"] += episode_stats["pushes"]
        results["blackjacks"] += episode_stats["blackjacks"]

        # Record final status and total withdrawn (not including final balance)
        if player.balance < config.min_balance_threshold and not made_withdrawal:
            results["wiped_out"] += 1
            results["final_withdrawals"].append(0)  # Wiped out = 0 withdrawals
        else:
            if player.withdrawn == config.initial_goal:
                results["first_goal"] += 1
            else:
                results["beyond_first"] += 1
            results["final_withdrawals"].append(
                total_withdrawn
            )  # Only actual withdrawals

    return results


HandResult = Tuple[str, float]  # Type alias for clarity


def play_hand(
    player: MultiHandProgressivePlayer,
    dealer: Dealer,
    deck: CardDeck,
    hand_idx: int,
    bet: float,
    debug: bool,
) -> HandResult:
    """Play a single hand and return the result and final bet amount"""
    max_decisions = 10
    decisions_made = 0
    split_count = 0
    max_splits = 3

    while decisions_made < max_decisions:
        decisions_made += 1
        action = player.make_decision_for_hand(hand_idx, dealer.hand[0])

        if action == "Split":
            if player.balance >= bet and split_count < max_splits:
                split_count += 1
                return ("Split", bet)
            else:
                # If can't split anymore, force a hit
                action = "Hit"

        if action == "Stand":
            return ("Stand", bet)

        elif action == "Hit":
            player.draw_card_to_hand(hand_idx, deck.draw_card())
            if player.hand_totals[hand_idx] > 21:
                return ("Bust", bet)
            continue

        elif action == "Double":
            if len(player.hands[hand_idx]) == 2 and player.balance >= bet:
                player.balance -= bet
                bet *= 2
                player.draw_card_to_hand(hand_idx, deck.draw_card())
                if player.hand_totals[hand_idx] > 21:
                    return ("Bust", bet)
                return ("Double", bet)
            else:
                action = "Hit"
                continue

    return ("Stand", bet)


if __name__ == "__main__":
    config = BlackjackConfig(
        aggressive_bet_chance=0.05,
        max_bet_percent=0.2,
        withdrawal_keep_ratio=0.3,
        withdrawal_threshold=1.3,
        bet_multiplier=1.5,
        min_balance_threshold=1.0,
        num_hands=5,
        initial_balance=5000,
    )

    # Run simulation with 10,000 episodes
    results = simulate_episodes(config, num_episodes=10000, debug=False)

    # Print results with proper formatting
    print("\nResults:")
    total_episodes = (
        results["wiped_out"] + results["first_goal"] + results["beyond_first"]
    )
    print(
        f"Wiped out: {results['wiped_out']} ({results['wiped_out']/total_episodes*100:.1f}%)"
    )
    print(
        f"Reached first goal: {results['first_goal']} ({results['first_goal']/total_episodes*100:.1f}%)"
    )
    print(
        f"Beyond first goal: {results['beyond_first']} ({results['beyond_first']/total_episodes*100:.1f}%)"
    )
    print(
        f"Mean withdrawal: ${sum(results['final_withdrawals'])/len(results['final_withdrawals']):.2f}"
    )
    print("\nHand Statistics:")
    total_hands = results["hands_won"] + results["hands_lost"] + results["hands_pushed"]
    print(f"Win rate: {results['hands_won']/total_hands*100:.1f}%")
    print(f"Loss rate: {results['hands_lost']/total_hands*100:.1f}%")
    print(f"Push rate: {results['hands_pushed']/total_hands*100:.1f}%")
    print(f"Blackjack rate: {results['blackjacks']/total_hands*100:.2f}%")

    # Create histogram of total withdrawals (not including final balances)
    withdrawals = np.array(results["final_withdrawals"])
    mean_withdrawal = np.mean(withdrawals)
    median_withdrawal = np.median(withdrawals)

    plt.figure(figsize=(12, 6))
    plt.hist(withdrawals, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
    plt.axvline(
        mean_withdrawal,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: ${mean_withdrawal:.2f}",
    )
    plt.axvline(
        median_withdrawal,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"Median: ${median_withdrawal:.2f}",
    )

    plt.title("Distribution of Total Withdrawals")  # Updated title
    plt.xlabel("Total Withdrawn ($)")  # Updated label
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
