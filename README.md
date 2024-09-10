# ChatGPT_Trading_Bot


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