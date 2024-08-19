<img align="left" src=jojo/StateReward.webp width="55%"/>
<div >
 
  <h1 align="right">
    InfoSoft Finance
  </h1>
</div>
<div align="left">

Reinforcement learning (RL) is a machine learning technique that focuses on training an algorithm following the cut-and-try approach. The algorithm (agent) evaluates a current situation (state), takes an action, and receives feedback (reward) from the environment after each act. Positive feedback is a reward (in its usual meaning for us), and negative feedback is punishment for making a mistake.

## Deep Reinforcement Learning for Automated Stock Trading
We introduce an ensemble strategy using deep reinforcement learning to maximize returns. This involves a learning agent employing three actor-critic algorithms: Proximal Policy Optimization (PPO), Advantage Actor Critic (A2C), and Deep Deterministic Policy Gradient (DDPG). This ensemble approach combines the strengths of each algorithm, adapting well to market changes. To manage memory usage when training with continuous action spaces, we use a load-on-demand data processing technique. We tested our strategy on 30 Dow Jones stocks and compared its performance to the Dow Jones Industrial Average and the traditional min-variance portfolio allocation. Our ensemble strategy outperformed individual algorithms and baselines in risk-adjusted returns, as measured by the Sharpe ratio.

Keywords: Deep Reinforcement Learning, Markov Decision Process, Automated Stock Trading, Ensemble Strategy, Actor-Critic Framework

## Outline

- [Deep Reinforcement Learning for Automated Stock Trading](#deep-reinforcement-learning-for-automated-stock-trading)
- [Outline](#outline)
- [Overview](#overview)
- [Reinforcement, supervised, and unsupervised learning](#reinforcement-supervised-and-unsupervised-learning)
- [Multi-level deep Q-networks for Bitcoin trading strategies](#multi-level-deep-q-networks-for-bitcoin-trading-strategies)
- [File Structure](#file-structure)
- [Supported Data Sources](#supported-data-sources)
- [Deep Q-network](#deep-q-network)
- [News articles](#news-articles)
- [Installation](#installation)
- [Status Update](#status-update)
- [Tutorials](#tutorials)
- [Publications](#publications)
- [News](#news)
- [Citing](#citing)
- [Join and Contribute](#join-and-contribute)
  - [Contributors](#contributors)
- [LICENSE](#license)
## Overview

Jojo has three layers: market environments, agents, and applications.  For a trading task (on the top), an agent (in the middle) interacts with a market environment (at the bottom), making sequential decisions.

## Reinforcement, supervised, and unsupervised learning

<div align="center">
<img align="center" src=figs/finrl_framework.png>
</div>

In supervised learning, an agent “knows” what task to perform and which set of actions is correct. Data scientists train the agent on historical data with target variables (desired answers with predictive analysis) AKA labeled data. The agent receives direct feedback. As a result of training, an agent can forecast whether there will be target variables in new data or not. Supervised learning allows for solving classification and regression tasks.

Reinforcement learning doesn’t rely on labeled datasets: The agent isn’t told which actions to take or the optimal way of performing a task. RL uses rewards and penalties instead of labels associated with each decision in datasets to signal whether a taken action is good or bad. So, the agent only gets feedback once it completes the task. That’s how time-delayed feedback and the trial-and-error principle differentiate reinforcement learning from supervised learning.

Since one of the goals of RL is to find a set of consecutive actions that maximize a reward, sequential decision making is another significant difference between these algorithm training styles. Each agent’s decision can affect its future actions.

Reinforcement learning vs unsupervised learning. In unsupervised learning, the algorithm analyzes unlabeled data to find hidden interconnections between data points and structures them by similarities or differences. RL aims at defining the best action model to get the biggest long-term reward, differentiating it from unsupervised learning in terms of the key goal.

Reinforcement and deep learning. Most of reinforcement learning implementations employ deep learning models. They involve the use of deep neural networks as the core method for agent training. Unlike other machine learning methods, deep learning fits best for recognizing complex patterns in images, sounds, and texts. Additionally, neural networks allow data scientists to fit all processes into a single model without breaking down the agent’s architecture into multiple modules.

## Multi-level deep Q-networks for Bitcoin trading strategies

We propose a multi-level deep Q-network (M-DQN) that leverages historical Bitcoin price data and Twitter sentiment analysis. In addition, an innovative preprocessing pipeline is introduced to extract valuable insights from the data, which are then input into the M-DQN model. In the experiments, this integration led to a noteworthy increase in investment value from the initial amount and a Sharpe Ratio in excess of 2.7 in measuring risk-adjusted return.

Reinforcement learning is applicable in numerous industries, including internet advertising and eCommerce, finance, robotics, and manufacturing. Let’s take a closer look at these use cases.



<div>
## File Structure

The main folder **finrl** has three subfolders **applications, agents, meta**. We employ a **train-test-trade** pipeline with three files: train.py, test.py, and trade.py.
</div>

```
FinRL
├── finrl (main folder)
│   ├── applications
│   	├── Stock_NeurIPS2018
│   	├── imitation_learning
│   	├── cryptocurrency_trading
│   	├── high_frequency_trading
│   	├── portfolio_allocation
│   	└── stock_trading
│   ├── agents
│   	├── elegantrl
│   	├── rllib
│   	└── stablebaseline3
│   ├── meta
│   	├── data_processors
│   	├── env_cryptocurrency_trading
│   	├── env_portfolio_allocation
│   	├── env_stock_trading
│   	├── preprocessor
│   	├── data_processor.py
│       ├── meta_config_tickers.py
│   	└── meta_config.py
│   ├── config.py
│   ├── config_tickers.py
│   ├── main.py
│   ├── plot.py
│   ├── train.py
│   ├── test.py
│   └── trade.py
│
├── examples
├── unit_tests (unit tests to verify codes on env & data)
│   ├── environments
│   	└── test_env_cashpenalty.py
│   └── downloaders
│   	├── test_yahoodownload.py
│   	└── test_alpaca_downloader.py
├── setup.py
├── requirements.txt
└── README.md
```

## Supported Data Sources

|Data Source |Type |Range and Frequency |Request Limits|Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|[Akshare](https://alpaca.markets/docs/introduction/)| CN Securities| 2015-now, 1day| Account-specific| OHLCV| Prices&Indicators|
|[Alpaca](https://alpaca.markets/docs/introduction/)| US Stocks, ETFs| 2015-now, 1min| Account-specific| OHLCV| Prices&Indicators|
|[Baostock](http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3)| CN Securities| 1990-12-19-now, 5min| Account-specific| OHLCV| Prices&Indicators|
|[Binance](https://binance-docs.github.io/apidocs/spot/en/#public-api-definitions)| Cryptocurrency| API-specific, 1s, 1min| API-specific| Tick-level daily aggegrated trades, OHLCV| Prices&Indicators|
|[CCXT](https://docs.ccxt.com/en/latest/manual.html)| Cryptocurrency| API-specific, 1min| API-specific| OHLCV| Prices&Indicators|
|[EODhistoricaldata](https://eodhistoricaldata.com/financial-apis/)| US Securities| Frequency-specific, 1min| API-specific | OHLCV | Prices&Indicators|
|[IEXCloud](https://iexcloud.io/docs/api/)| NMS US securities|1970-now, 1 day|100 per second per IP|OHLCV| Prices&Indicators|
|[JoinQuant](https://www.joinquant.com/)| CN Securities| 2005-now, 1min| 3 requests each time| OHLCV| Prices&Indicators|
|[QuantConnect](https://www.quantconnect.com/docs/home/home)| US Securities| 1998-now, 1s| NA| OHLCV| Prices&Indicators|
|[RiceQuant](https://www.ricequant.com/doc/rqdata/python/)| CN Securities| 2005-now, 1ms| Account-specific| OHLCV| Prices&Indicators|
[Sinopac](https://sinotrade.github.io/zh_TW/tutor/prepare/terms/) | Taiwan securities | 2023-04-13~now, 1min | Account-specific | OHLCV | Prices&Indicators|
|[Tushare](https://tushare.pro/document/1?doc_id=131)| CN Securities, A share| -now, 1 min| Account-specific| OHLCV| Prices&Indicators|
|[WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/)| US Securities| 2003-now, 1ms| 5 requests each time| Intraday Trades|Prices&Indicators|
|[YahooFinance](https://pypi.org/project/yfinance/)| US Securities| Frequency-specific, 1min| 2,000/hour| OHLCV | Prices&Indicators|


OHLCV: open, high, low, and close prices; volume. adjusted_close: adjusted close price

Technical indicators: 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'. Users also can add new features.

## [Deep Q-network](https://www.nature.com/articles/s41598-024-51408-w#:~:text=Building%20upon%20the,large%2Dscale%20problems)
Building upon the foundations of Q-learning, DQN is an extension that combines reinforcement learning with deep learning techniques15. It uses a deep neural network as an approximator to estimate the action-value function Q(s, a). DQN addresses the main challenges of traditional Q-learning, such as learning stability. Moreover, by employing deep learning, DQN can handle high-dimensional state spaces, such as those encountered in image-based tasks or large-scale problems

## News articles
Returns latest news articles across stocks and crypto. By default returns latest 10 news articles.
News recommendation. Machine learning has made it possible for businesses to personalize customer interactions at scale through the analysis of data on their preferences, background, and online behavior patterns.

However, recommending such content type as online news is still a complex task. News features are dynamic by nature and become rapidly irrelevant. User preferences in topics change as well.
A Deep Reinforcement Learning Framework for News Recommendation discuss three main challenges related to news recommendation methods. 
We used the Deep Q-Learning based recommendation framework that considers current reward and future reward simultaneously in addition to user return as feedback.

## Installation
+ [Install description for all operating systems (MAC OS, Ubuntu, Windows 10)](./docs/source/start/installation.rst)
+ [FinRL for Quantitative Finance: Install and Setup Tutorial for Beginners](https://ai4finance.medium.com/finrl-for-quantitative-finance-install-and-setup-tutorial-for-beginners-1db80ad39159)

## Status Update
<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>

* 2022-06-25
	0.3.5: Formal release of FinRL, neo_finrl is chenged to FinRL-Meta with related files in directory: *meta*.
* 2021-08-25
	0.3.1: pytorch version with a three-layer architecture, apps (financial tasks), drl_agents (drl algorithms), neo_finrl (gym env)
* 2020-12-14
  	Upgraded to **Pytorch** with stable-baselines3; Remove tensorflow 1.0 at this moment, under development to support tensorflow 2.0
* 2020-11-27
  	0.1: Beta version with tensorflow 1.5
</div>
</details>

## Tutorials

+ [Towardsdatascience] [Deep Reinforcement Learning for Automated Stock Trading](https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02)

## Publications

<details>
<summary>Click to view publications</summary>

| Title | Conference/Journal | Link | Citations | Year |
|-------|-------------------|------|-----------|------|
| Dynamic Datasets and Market Environments for Financial Reinforcement Learning | Machine Learning - Springer Nature | [paper](https://arxiv.org/abs/2304.13174) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta) | 7 | 2024 |
| **FinRL-Meta**: FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning | NeurIPS 2022 | [paper](https://arxiv.org/abs/2211.03107) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta) | 37 | 2022 |
| **FinRL**: Deep reinforcement learning framework to automate trading in quantitative finance | ACM International Conference on AI in Finance (ICAIF) | [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3955949) | 49 | 2021 |
| **FinRL**: A deep reinforcement learning library for automated stock trading in quantitative finance | NeurIPS 2020 Deep RL Workshop | [paper](https://arxiv.org/abs/2011.09607) | 87 | 2020 |
| Deep reinforcement learning for automated stock trading: An ensemble strategy | ACM International Conference on AI in Finance (ICAIF) | [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/2-Advance/FinRL_Ensemble_StockTrading_ICAIF_2020/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb) | 154 | 2020 |
| Practical deep reinforcement learning approach for stock trading | NeurIPS 2018 Workshop on Challenges and Opportunities for AI in Financial Services | [paper](https://arxiv.org/abs/1811.07522) [code](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/FinRL/tree/master/examples)) | 164 | 2018 |

</details>

## News
Returns latest news articles across stocks and crypto. 
By default returns latest 10 news articles.
Example using symbol BTCUSD-BitCoin start:2024-08-04 end 2024-11-04:

{
  "news": [
    {
      "author": "Bibhu Pattnaik",
      "content": "<p>Crypto analyst <strong>Kevin Svenson</strong> predicts that <strong>Bitcoin</strong> (CRYPTO: <a class=\"ticker\" href=\"https://www.benzinga.com/quote/btc/usd\">BTC</a>) could see an upsurge of up to 86%, <a href=\"https://www.benzinga.com/markets/cryptocurrency/23/12/36274982/crypto-analyst-forecasts-monumental-bitcoin-rally-by-2026\">potentially hitting the $100,000 mark</a>.</p>\n\n\n\n<p><strong>What Happened</strong>: Svenson said that Bitcoin has formed a bullish divergence pattern on the daily chart. This pattern emerges when the asset&#8217;s price is trading down or sideways, while an oscillator like the relative strength index (RSI) is in an uptrend, indicating increasing bullish momentum.</p>\n\n\n\n<p>In a video <a href=\"https://www.youtube.com/watch?v=dU_qfQRtxks\">post</a>, Svenson said, &#8220;We got actually a slightly higher low in the RSI – you could also call it flat support, horizontal support. And upon that flat support, we had lower lows in price. That is a bullish divergence.&#8221;</p>\n\n\n\n<p>He also observed a broadening pattern in the Bitcoin chart, characterized by lower highs and even lower lows, which could be interpreted as a bullish continuation pattern if the asset breaks its diagonal resistance.</p>\n\n\n\n<p>&#8220;And so what is happening on the Bitcoin chart? Well, what we see is a broadening pattern of sorts – lower highs but even lower lows,\" he added. </p>\n\n\n\n<p>Also Read: <a href=\"https://www.benzinga.com/markets/cryptocurrency/24/06/39241558/analyst-predicts-bitcoin-to-reach-groundbreaking-100-000-milestone\">Analyst Predicts Bitcoin To Reach Groundbreaking $100,000 Milestone</a></p>\n\n\n\n<p>

## Citing
[DRL based trading agents: Risk driven learning for financial rules-based policy](https://www.sciencedirect.com/journal/expert-systems-with-applications)

[A General Portfolio Optimization Environment](https://sol.sbc.org.br/index.php/bwaif/article/view/24959/24780)

## Join and Contribute

Welcome to **JojoFinance** community!

### Contributors

Thank you!

## LICENSE

MIT License

**Disclaimer: We are sharing codes for academic purpose under the MIT education license. Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**