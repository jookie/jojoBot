<div align="center">

[Stable Base Line+DOW](StableBasdelineDowJones.md)
[Introduction](OVERVIEW) |
[SnapShot](READMECodeSnapShot.md) | 
[PaperTrading](READMExpAlpacaPaperTrading.md) | 
[FAQ](READMEfaq.md)
</div>

## Introduction

If you're tired staring at charts all day, and doing trades manually while exploring daily market data
just relax and let the bot do all the hard work.
This Trading-bot operates 24/7, ensuring no trading opportunities are missed. An AI JOJO Trading Bot offers the benefits of automated trading without needing constant attention, giving regular investors a chance for higher returns. 
The name of the AI trading bot is JojoFin. It is like having an automatic helper that trades for you 24/7 based on set rules, quickly making lots of small trades to profit from market changes, while traditional investing involves buying assets and holding them for a long time with less frequent trades and lower risk.
## Reinforcement Learning (RL)
Reinforcement Learning (RL) techniques are considered convenient for this task : 
In this experiment, we present an RL environment for the portfolio optimization based on state-of-the-art mathematical formulations. The environment aims to be easy-to-use, very customizable, and have integrations with modern RL frameworks.
Jojobot is a library that will allow you to easily create trading robots that are profitable in many different asset classes, including Stocks, Options, Futures, FOREX, and more. 
Check your trading strategies against historical data to make sure they are profitable before you invest in them. JojoBot makes it easy for you to do  (backtest) your trading strategies and easily convert them to algorithmic trading robots.
<br>
KeyWords: Portfolio optimization, Reinforcement learning, Simulation environment, Quantitative finance
<br>
## Overview
1. Pull 30 days of trading data for (Insert your stock or crypto) with Yahoo Finance Downloader API
2. Create a simulated trading environment using real trading data 
3. Train an neural network to predict that Stock Price using reinforcement learning inside this simulation
4. Once trained, backtest the predictions on the past 30 days data to compute potential returns 
5. If the expe

## Task Discription
This task is modeled as a Markov Decision Process (MDP), and the objective function is maximizing (expected) cumulative return.
### gym-style market environment 
It is built on top of the OpenAI Gym interface to train and evaluate DRL models. In this experiment, we use Dow Jones as a stock pool, and Stable baselines3 to train DRL agents.

## Market environment: 
30 consituent stocks of Dow Jones Industrial Average (DJIA) index. Accessed at the starting date of the testing period.
The data for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.

A **gym-style market environment** refers to the use of OpenAI's Gym framework to simulate financial markets in a reinforcement learning (RL) setup. Gym is a toolkit for developing and comparing RL algorithms, and it provides a standardized interface to interact with different types of environments. In a market environment, the idea is to represent financial markets as environments where an agent (such as a trading bot) can take actions (like buying, selling, or holding assets) and receive rewards based on market outcomes (like profit or loss).

In a gym-style market environment:

1. **States**: These represent the observable information about the market, like stock prices, technical indicators, or fundamental data. The state is updated as time progresses in the environment (e.g., a new trading day starts).

2. **Actions**: These are the decisions the agent can make, such as buying, selling, or holding a stock. In some cases, actions might also involve managing a portfolio or rebalancing.

3. **Rewards**: The agent receives rewards based on the actions it takes. In trading, rewards often relate to profit or loss, but they can also be linked to risk-adjusted returns, transaction costs, or other financial metrics.

4. **Episodes**: Similar to games in gym environments, an episode in a market environment could represent a certain period, such as a single day, week, or month of trading.

These environments allow developers and researchers to train reinforcement learning models for tasks like automated trading, portfolio optimization, or risk management.

## Results:
Setting the rolling window to 22 trading days, which is approximately one month, the training period is from January 1, 2009, to July 1, 2022, and the trading period is from July 1, 2022, to November 1, 2022. As shown in the above figure, the best intelligent agent is A2C, which has a return of 10.2%, while the return of DJI (Dow Jones Industrial Average) is 3%.

# ChatGPT_Trading_Bot
Here is a table based on the provided information:

| **Category** | **Metric**         | **Value** |
| ------------ | ------------------ | --------- |
| **Time**     | Episodes           | 8         |
|              | FPS                | 113       |
|              | Time Elapsed (sec) | 242       |
|              | Total Timesteps    | 27,520    |
| **Train**    | Actor Loss         | 30.5      |
|              | Critic Loss        | 118       |
|              | Learning Rate      | 0.001     |
|              | Number of Updates  | 27,419    |
|              | Reward             | -8.926152 |


This block of text appears to be output from training a reinforcement learning model, likely in a machine learning framework. Here's a breakdown of each part:

### Time Section:
- **episodes**: Number of training episodes completed (in this case, 8 episodes).
- **fps**: Frames per second, which indicates the speed of training or how fast the model is processing timesteps (113 frames per second).
- **time_elapsed**: Total time spent training so far, in seconds (242 seconds).
- **total_timesteps**: The cumulative number of timesteps processed during training (27,520 timesteps).

### Train Section:
- **actor_loss**: The loss value of the actor network in reinforcement learning, which is responsible for selecting actions (30.5).
- **critic_loss**: The loss value of the critic network, which evaluates the chosen actions (118). A high critic loss may indicate the need for further optimization.
- **learning_rate**: The rate at which the model adjusts its weights during training (0.001 in this case).
- **n_updates**: The number of updates to the model's parameters during training (27,419 updates).
- **reward**: The average reward obtained per episode, which reflects the model's performance. Here, the negative value (-8.93) suggests that the model is performing poorly or has not learned the task well yet.

In summary, the model is in the early stages of training (8 episodes), and the performance (reward) is still negative, indicating that more training and tuning may be needed to improve results.


day: 3439, episode: 70
begin_total_asset: 1000000.00
end_total_asset: 9410205.63
total_reward: 8410205.63
total_cost: 999.00
total_trades: 54825
Sharpe: 0.977

This block of text represents a summary of the performance of a financial reinforcement learning model, likely from a backtesting session or simulation. Here’s a breakdown of each component:

- **day: 3439, episode: 70**: 
  - **day 3439**: This could indicate that the simulation has run for 3439 days in the environment.
  - **episode 70**: The 70th training episode, indicating progress through multiple training iterations.

- **begin_total_asset: 1000000.00**: 
  - This is the initial amount of money or asset value at the start of the simulation, which is 1,000,000 units (could be dollars or another currency).

- **end_total_asset: 9410205.63**: 
  - This is the total asset value at the end of the episode. Here, it has grown to 9,410,205.63, suggesting significant growth over the course of the episode.

- **total_reward: 8410205.63**: 
  - This is the total reward the agent accumulated throughout the episode, calculated as the difference between the beginning and ending assets. In this case, it’s 8,410,205.63, reflecting the profit made.

- **total_cost: 999.00**: 
  - This is the total transaction cost incurred from buying and selling assets during the episode, which is relatively small (999 units).

- **total_trades: 54825**: 
  - The total number of trades executed by the agent during the episode (54,825 trades), indicating a high level of trading activity.

- **Sharpe: 0.977**: 
  - The **Sharpe ratio** is a measure of risk-adjusted return. A Sharpe ratio of 0.977 is reasonably good, indicating that the agent is generating returns with acceptable risk. A higher Sharpe ratio typically suggests better risk-adjusted performance.

### Summary:
The agent started with 1,000,000 units, ended with over 9.4 million units, made a significant profit (total reward), and executed a large number of trades with a relatively low transaction cost. The Sharpe ratio is decent, indicating the trades were made with a good balance between risk and return.


final return:  {'DJI': 0.04019851324835555, 'A2C': 0.07738397358608262, 'DDPG': 0.06724694765557149, 'TD3': 0.092468645831711, 'PPO': 0.027524547849764103, 'SAC': 0.035271144414289246}

This output represents the **final return** (percentage increase in asset value) from different algorithms or models used in a financial trading context. Each of these abbreviations corresponds to different reinforcement learning algorithms or market indices, and the number represents the percentage return over the testing or simulation period. Here's a breakdown:

- **DJI (Dow Jones Industrial Average)**: The return from following the performance of the Dow Jones Industrial Average. In this case, the return is 0.0402, or approximately **4.02%**.
  
- **A2C (Advantage Actor-Critic)**: This is a reinforcement learning algorithm. Its final return is **7.74%**.
  
- **DDPG (Deep Deterministic Policy Gradient)**: Another RL algorithm, commonly used in continuous action spaces. It has a final return of **6.72%**.
  
- **TD3 (Twin Delayed DDPG)**: A more advanced variant of DDPG, known for improving stability in training. It has a return of **9.25%**, the highest among the algorithms listed.
  
- **PPO (Proximal Policy Optimization)**: A popular policy gradient method. Its final return is **2.75%**, the lowest in this set.
  
- **SAC (Soft Actor-Critic)**: A reinforcement learning algorithm known for its stability and exploration. It shows a return of **3.53%**.

### Summary:
The **TD3** algorithm performed the best, with a **9.25%** return, while **PPO** had the lowest performance at **2.75%**. The **DJI** return represents the performance of a well-known stock index, showing how it compares to the reinforcement learning algorithms.

## PLOT RESULTS

![Stock Trading](results/stock_trading.0.png)
![Stock Trading](results/stock_trading.0.png)
![Stock Trading](results/stock_trading.0.png)

To explain the plot of all agents' results, it would typically contain lines or bars representing the performance (returns or rewards) of different agents or algorithms over time. Here's how you can interpret a typical plot of results for various reinforcement learning agents in a trading environment:

### What to Look For:

1. **X-axis (Time / Days / Episodes)**: 
   - Usually, this axis represents time in the simulation. It could be in terms of **days** if you're simulating trading over a period, or **episodes** if you're looking at performance across multiple training runs.

2. **Y-axis (Total Assets / Cumulative Reward / Return)**:
   - This axis could represent different performance metrics depending on the plot. In trading environments, it often shows **total assets**, **cumulative rewards**, or **percentage returns**. 
   - A higher value on the Y-axis indicates better performance.

3. **Lines or Bars (Agents' Performance)**:
   - Each line (or bar) represents the performance of a specific reinforcement learning algorithm (agent).
   - For example, the plot may contain lines for:
     - **A2C** (Advantage Actor-Critic)
     - **DDPG** (Deep Deterministic Policy Gradient)
     - **TD3** (Twin Delayed DDPG)
     - **PPO** (Proximal Policy Optimization)
     - **SAC** (Soft Actor-Critic)
     - **DJI** (Baseline, e.g., Dow Jones Industrial Average)

### Key Metrics to Observe:

1. **Performance over Time**:
   - Compare how the agents’ performance evolves as time progresses. Does one algorithm consistently outperform others?
   - For example, if **TD3** consistently stays above the other algorithms, it indicates stronger performance in terms of returns or rewards over time.

2. **Stability and Volatility**:
   - If a line shows frequent ups and downs, the agent is volatile, meaning its performance fluctuates a lot.
   - A smoother upward line typically indicates a more stable learning process or investment strategy, while jagged lines could indicate higher risk or instability.

3. **Crossovers**:
   - If the lines for two agents cross, this means one agent's performance surpassed the other at a specific point in time. It can be useful to see which algorithms perform well in different market conditions or training phases.

4. **Final Returns**:
   - The endpoint of each line gives the final return or total assets after the last day or episode. You can compare which agent achieved the highest return by looking at the ending value of each line.

### Example Plot Analysis:

- **TD3** might have the steepest incline, indicating it earns higher returns faster and more consistently compared to others like **PPO** or **A2C**.
- **DJI** (the stock index baseline) might have a slower, steadier incline, which can be compared to the RL agents’ performance to see if the agents outperform the traditional market.
- If **PPO** remains lower on the plot, it indicates that its strategy or learning approach is not as effective as the others.

### Conclusion:
A plot of all agents' results will allow you to visually assess which agent is performing best over time, which ones are volatile, and whether the reinforcement learning agents are able to outperform a baseline such as **DJI**. You can use this to further tune your models or choose the best-performing algorithm for live trading.


## Overview

1. Pull 30 days of trading data for (Insert your stock or crypto) with Yahoo Finance Downloader API
2. Create a simulated trading environment using real trading data with FinRL
3. Train an neural network to predict that Stock Price using reinforcement learning inside this simulation with FinRL
4. Once trained, backtest the predictions on the past 30 days data to compute potential returns with FinRL
5. If the expectd returns are above a certain threshold, buy, else hold. If they're below a certain threshold, sell. (using Alpaca API)

In order to have this to run automatically once a day, we can deploy it to a hosting platform like Vercel with a seperate file that repeatedly executes it. 

#### from config file, TRAIN , TEST and TRADE days
```python
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TRADE_START_DATE = '2021-10-01'
TRADE_END_DATE = '2023-03-01'
```
#### Yahoo donloader for data frames collection from Start Train to End Tradedate
```python
df = YahooDownloader(start_date = TRAIN_START_DATE,
                      end_date = TRADE_END_DATE,
                      ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
```     
#### Features Included DOW_30_TICKER - Technical, VIX and Turbelance INDICATORS, 
```python
fe = FeatureEngineer(
                      use_technical_indicator=True,
                      tech_indicator_list = INDICATORS,
                      use_vix=True,
                      use_turbulence=True,
                      user_defined_feature = False)
```  
#### Envionment Aeguments
```python
 env_kwargs = {
      "hmax": 100,
      "initial_amount": 1000000,
      "num_stock_shares": num_stock_shares,
      "buy_cost_pct": buy_cost_list,
      "sell_cost_pct": sell_cost_list,
      "state_space": state_space,
      "stock_dim": stock_dimension,
      "tech_indicator_list": INDICATORS,
      "action_space": stock_dimension,
      "reward_scaling": 1e-4
  }
  ```
#### Taining Agents Ensamble
```python
  print(insample_risk_indicator.vix.quantile(0.996))
  print(insample_risk_indicator.turbulence.quantile(0.996))
  e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold =70,  risk_indicator_col='vix', **env_kwargs)
  
  models = {
      "a2c": trained_a2c,
      "ddpg": trained_ddpg,
      "ppo": trained_ppo,
      "td3": trained_td3,
      "sac": trained_sac
  }

  results = predict_with_models(models, e_trade_gym)
  # Access results for each model
  df_account_value_a2c = results["a2c"]["account_value"]
  df_actions_a2c = results["a2c"]["actions"]

  df_account_value_ddpg = results["ddpg"]["account_value"]
  df_actions_ddpg = results["ddpg"]["actions"]

  df_account_value_ppo = results["ppo"]["account_value"]
  df_actions_ppo = results["ppo"]["actions"]

  df_account_value_td3 = results["td3"]["account_value"]
  df_actions_td3 = results["td3"]["actions"]

  df_account_value_sac = results["sac"]["account_value"]
  df_actions_sac = results["sac"]["actions"]
#### Taining Agents Ensamble
```
#### predict_with_models Ensamble
```python
  def predict_with_models(models, environment):
      """
      Perform predictions using multiple trained models in the specified environment.

      Parameters:
      - models: A dictionary of trained models with names as keys.
      - environment: The trading environment to be used for predictions.

      Returns:
      - results: A dictionary containing DataFrames of account values and actions for each model.
      """
      results = {}

      for model_name, trained_model in models.items():
          df_account_value, df_actions = DRLAgent.DRL_prediction(
              model=trained_model,
              environment=environment
          )
          results[model_name] = {
              "account_value": df_account_value,
              "actions": df_actions
          }

      return results

```