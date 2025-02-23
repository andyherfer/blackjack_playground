# Progressive Multi-Hand Blackjack Strategy

This implementation simulates a progressive betting strategy for playing multiple hands of blackjack simultaneously. The strategy combines standard blackjack basic strategy with a sophisticated money management system.

## Core Strategy Components

### Basic Game Rules
- Uses 6 decks of cards
- Dealer stands on hard 17 and hits on soft 17
- Blackjack pays 3:2
- Double down allowed on any two cards
- Splitting allowed up to 3 times

### Betting Strategy

1. **Progressive Betting**
   - Each hand starts with a minimum bet
   - After a win: Bet increases by 50% (1.5x multiplier)
   - After a loss: Bet resets to minimum
   - Each hand's progression is tracked independently

2. **Bet Size Limitations**
   - Maximum bet cannot exceed 30% of current balance
   - Hard maximum bet limit of $1,000
   - Minimum bet of $20

### Money Management

1. **Goal System**
   - Initial goal set at $2,000 in profits
   - Each subsequent goal increases by $1,000
   - Goals trigger withdrawal evaluation

2. **Withdrawal Rules**
   - Triggers when balance exceeds current goal
   - If balance > 120% of goal (withdrawal threshold):
     - Withdraws 80% of profits above initial balance
     - Keeps 20% for continued play
   - Balance resets to initial amount after withdrawal
   - Withdrawn amounts are protected from future losses

3. **Risk Management**
   - Play stops if balance falls below minimum threshold
   - Maximum of 10,000 hands per session
   - Maximum balance goal of $100,000

### Playing Strategy

1. **Basic Strategy**
   - Follows standard basic strategy decision matrix for:
     - Hard totals
     - Soft totals
     - Pairs
   - Decisions include: Hit, Stand, Double, Split

2. **Multi-Hand Play**
   - Default configuration plays 5 hands simultaneously
   - Each hand tracked and managed independently
   - Hands can be split up to 3 times
   - Double down only allowed on initial two cards

### Session Management

1. **Deck Penetration**
   - Reshuffles when fewer than 20 cards remain
   - Forces reshuffle every 52 hands

2. **Session Termination Conditions**
   - Balance falls below minimum threshold
   - Maximum hands limit reached
   - Maximum balance goal achieved
   - All withdrawal goals met

## Default Configuration
```python
initial_balance = 5000
min_bet = 20
max_bet = 1000
bet_multiplier = 1.5
num_hands = 5
initial_goal = 2000
goal_increment = 1000
withdrawal_threshold = 1.2 # 120% of goal
withdrawal_keep_ratio = 0.2 # Keep 20%
withdrawal_amount_ratio = 0.8 # Withdraw 80%
```


## Performance Metrics

The strategy's performance is evaluated based on:
- Percentage of sessions reaching first goal
- Percentage of sessions reaching beyond first goal
- Percentage of sessions resulting in wipeout
- Average withdrawal amount
- Hand-level statistics (win/loss/push rates)
- Blackjack frequency

The simulation includes visualization of withdrawal distributions to analyze strategy effectiveness.