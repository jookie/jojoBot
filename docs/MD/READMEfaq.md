
<br>
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

### How It Works:
1. **Load the Trained Model:** Replace the `...` with the method you use to load your trained model (e.g., with `pickle` or TensorFlow).
2. **Retrieve Market Data:** The Alpaca API is used to fetch the latest market data for the specific stock.
3. **Predict Actions:** The model predicts the action (buy/sell/hold) based on the current market data.
4. **Execute Trades:** The predicted action is executed via the Alpaca API.

### Automating Execution
- **Scheduler:** You can automate the execution of this script using a task scheduler like `cron` on Unix-based systems or Task Scheduler on Windows.
- **Daily Updates:** Make sure to run this script at the start of each trading day to decide the actions for that day.

This is a basic example. Depending on your strategy, you may need to handle more complexities like position sizing, multiple assets, risk management, etc.


Inputs and datasets
========================================================================

    
### Q: What is the best data source for free daily data?***

	*Yahoo Finance (through the yfinance library)*

#### Q: What is the best data source for minute data? ***

	*Yahoo Finance (only up to last 7 days), through the yfinance library. It is the only option besides scraping (or paying for a service provider)*

#### Q: Does the softwre support trading with leverage? ***

	*No, as this is more of an execution strategy related to risk control. You can use it as part of your system, adding the risk control part as a separate component*

#### Q: Can a sentiment feature be added to improve the model's performance? ***

	*Yes, you can add it. Remember to check on the code that this additional feature is being fed to the model (state)*

#### Q: Is there a good free source for market sentiment to use as a feature?  ***

	*No, you'll have to use a paid service or library/code to scrape news and obtain the sentiment from them (normally, using deep learning and NLP)*

Code and implementation
========================================================================

### Q: Does the softwre supports GPU training?  ***

	*Yes, it does*

#### Q: The code works for daily data but gives bad results on intraday frequency.***

	*Yes, because the current parameters are defined for daily data. You'll have to tune the model for intraday trading*

#### Q: Are there different reward functions available? ***

	*Not many yet, but we're working on providing different reward functions and an easy way to set your own reward function*

#### Q: Can I use a pre-trained model?  ***

	*Yes, but none is available at the moment. Sometimes in the literature you'll find this referred to as transfer learning*

#### Q: What is the most important hyperparameter to tune on the models?  ***

	*Each model has its own hyperparameters, 
    but the most important is the total_timesteps (think of it as epochs in a neural network: 
    even if all the other hyperparameters are optimal, 
    with few epochs the model will have a bad performance). The other important hyperparameters, 
    in general, are: learning_rate, batch_size, ent_coef, buffer_size, policy, and reward scaling

#### Q: What are some libraries I could use to better tune the models? ***

	*There are several, such as: Ray Tune and Optuna. You can start from our examples in the tutorials

#### Q: What DRL algorithms can I use with the softwre?  ***

	*We suggest using ElegantRL or Stable Baselines 3. We tested the following models with success: 
    A2C, A3C, DDPG, PPO, SAC, TD3, TRPO. You can also create your own algorithm,
    with an OpenAI Gym-style market environment*

#### Q: The model is presenting strange results OR is not training.   ***
    
    check if the hyperparameters used were not outside a normal range (ex: learning rate too high), and run the code again. If you still have problems, please check Section 2 (What to do when you experience problems)*

    - :raw-html: `<font color="#A52A2A">What to do when you experience problems? ***

    *1. Check if it is not already answered on this FAQ 2. Check if it is posted on the GitHub repo* `issues <https://github.com/AI4Finance-LLC/the softwre-Library/issues>`_. If not, welcome to submit an issue on GitHub 3. Use the correct channel on the AI4Finance slack or Wechat group.*

    - :raw-html: `<font color="#A52A2A">Does anyone know if there is a trading environment for a single stock? There is one in the docs, but the collab link seems to be broken. ***

        *We did not update the single stock for long time. The performance for single stock is not very good, since the state space is too small so that the agent extract little information from the environment. Please use the multi stock environment, and after training only use the single stock to trade.*


.. _Section-3:

3-Model evaluation
========================================================================

#### Q: The model did not beat buy and hold (BH) with my data. Is the model or code wrong?  ***

	*Not exactly. Depending on the period, the asset, the model chosen, and the hyperparameters used, BH may be very difficult to beat (it's almost never beaten on stocks/periods with low volatility and steady growth). Nevertheless, update the library and its dependencies (the github repo has the most recent version), and check the example notebook for the specific environment type (single, multi, portfolio optimization) to see if the code is running correctly*

#### Q: How does backtesting works in the library?  ***

	*We use the Pyfolio backtest library from Quantopian ( https://github.com/quantopian/pyfolio ), especially the simple tear sheet and its charts. In general, the most important metrics are: annual returns, cumulative returns, annual volatility, sharpe ratio, calmar ratio, stability, and max drawdown*

#### Q: Which metrics should I use for evaluting the model?  ***

	*There are several metrics, but we recommend the following, as they are the most used in the market: annual returns, cumulative returns, annual volatility, sharpe ratio, calmar ratio, stability, and max drawdown*

#### Q: Which models should I use as a baseline for comparison?  ***

	*We recommend using buy and hold (BH), as it is a strategy that can be followed on any market and tends to provide good results in the long run. You can also compare with other DRL models and trading strategies such as the minimum variance portfolio*

4-Miscellaneous
========================================================================
### Q: Can I use the softwre for crypto? ***
	*Not yet. We're developing this functionality*
### Q: Can I use the softwre for live trading?  ***
	*Not yet. We're developing this functionality*
### Q: Can I use the softwre for forex? ***
	*Not yet. We're developing this functionality*
### Q: Can I use the softwre for futures? ***
	*Not yet*


Common issues/bugs
====================================
