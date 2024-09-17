<div align="center">
<h2>

[Introduction](../../README.md) |
[DOW Stable Base Line](StableBasdelineDowJones.md) |
[OverView](OverView.md) |
[PaperTrading](READMExpAlpacaPaperTrading.md) | 
[Trading Experiments](READMExperiment.md) |
[TECH](/docs/MD/README.TECH.md) |
[FAQ](READMEfaq.md) | 
[SnapShot](READMECodeSnapShot.md) 

</h2>
</div>

# Stable baselines3 using Dow Jones Stock pool for Stock trading 

Library that uses deep reinforcement learning (DRL) for financial trading decision-making. Supports several DRL libraries, e.g., Stable Baselines3, and ElegantRL. Stable Baselines3 is a DRL library implemented in Python. 
It is built on top of the OpenAI Gym and provides a simple interface to train and evaluate DRL models. Hers, we use Dow Jones as a stock pool, and Stable baselines3 to train DRL agents.

## 1 Task Discription

We train a DRL agent for stock trading. This task is modeled as a Markov Decision Process (MDP), and the objective function is maximizing (expected) cumulative return.

We specify the state-action-reward as follows:

### State s: 
The state space represents an agent’s perception of the market environment. Just like a human trader analyzing various information, here our agent passively observes many features and learns by interacting with the market environment by replaying historical data.

### Action a: 
The action space includes allowed actions that an agent can take at each state. For example, a ∈ {−1, 0, 1}, where −1, 0, 1 represent selling, holding, and buying. When an action operates multiple shares, a ∈{−k, …, −1, 0, 1, …, k}, e.g.. “Buy 10 shares of AAPL” or “Sell 10 shares of AAPL” are 10 or −10, respectively

### Reward function r(s, a, s′): 
Reward is an incentive for an agent to learn a better policy. For example, it can be the change of the portfolio value when taking a at state s and arriving at new state s’, i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio values at state s′ and s, respectively

### Market environment: 
30 consituent stocks of Dow Jones Industrial Average (DJIA) index. Accessed at the starting date of the testing period.

The data is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.

## 2 Install and import

### 2.1 Install Python Packages

```python
## install required packages
!pip install swig
!pip install wrds
!pip install pyportfolioopt
## install finrl library
!pip install -q condacolab
import condacolab
condacolab.install()
!apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
```

2.2 Import packages and functions

```python
from finrl import config
from finrl import config_tickers
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import DATA_SAVE_DIR
from finrl.config import INDICATORS
from finrl.config import RESULTS_DIR
from finrl.config import TENSORBOARD_LOG_DIR
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config import TRAINED_MODEL_DIR
from finrl.config_tickers import DOW_30_TICKER
from finrl.main import check_and_make_directories
from finrl.meta.data_processor import DataProcessor
from finrl.meta.data_processors.func import calc_train_trade_data
from finrl.meta.data_processors.func import calc_train_trade_starts_ends_if_rolling
from finrl.meta.data_processors.func import date2str
from finrl.meta.data_processors.func import str2date
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.plot import backtest_plot
from finrl.plot import backtest_stats
from finrl.plot import get_baseline
from finrl.plot import get_daily_return
from finrl.plot import plot_return
from finrl.applications.stock_trading.stock_trading import stock_trading
import sys
sys.path.append("../FinRL")

import itertools
```

### 3. Set parameters and run

```python
We set parameters for the function stock_trading, and run it. The code is here.

    train_start_date = "2009-01-01"
    train_end_date = "2022-09-01"
    trade_start_date = "2022-09-01"
    trade_end_date = "2023-11-01"
    if_store_actions = True
    if_store_result = True
    if_using_a2c = True
    if_using_ddpg = True
    if_using_ppo = True
    if_using_sac = True
    if_using_td3 = True

    stock_trading(
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        trade_start_date=trade_start_date,
        trade_end_date=trade_end_date,
        if_store_actions=if_store_actions,
        if_store_result=if_store_result,
        if_using_a2c=if_using_a2c,
        if_using_ddpg=if_using_ddpg,
        if_using_ppo=if_using_ppo,
        if_using_sac=if_using_sac,
        if_using_td3=if_using_td3,
    )
### 4 Result

