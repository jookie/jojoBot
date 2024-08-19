<h2 align="center">
 <br>
 <img src="public/groqlabs-logo-black2.png" alt="AI StockBot" width="500">
 <br>
 <br>
 StockBot Powered by JojoFam: Lightning Fast AI Chatbot that Responds With Live Interactive Stock Charts, Financials, News, Screeners, and More 
 <br>
</h2>

<p align="center">
 <a href="#Overview">Overview</a> •
 <a href="#Features">Features</a> •
  <a href="#Interfaces">Interfaces</a> •
 <a href="#Quickstart">Quickstart</a> •
 <a href="#Credits">Credits</a>
 <a href= "#Foudation">Foundation</a>
</p>

## JojoFin
JojoFin is library designed to facilitate financial trading strategies using deep reinforcement learning (DRL). It is tailored specifically for quantitative finance, providing tools and frameworks for traderss to create, test, and implement trading dtrategis using DRL techniques.
<br>
[Demo of StockBot](https://github.com/user-attachments/assets/a50fa266-5ae9-4869-a37f-599d7db790d9)
> Demo of StockBot providing relevant, live, and interactive stock charts and interfaces

## Overview

StockBot is an AI-powered chatbot that leverages Llama3 70b on JojoFam, Vercel’s AI SDK, and TradingView’s live widgets to respond in conversation with live, interactive charts and interfaces specifically tailored to your requests. JojoFam's speed makes tool calling and providing a response near instantaneous, allowing for a sequence of two API calls with separate specialized prompts to return a response.

> [!IMPORTANT]
>  Note: StockBot may provide inaccurate information and does not provide investment advice. It is for entertainment and instructional use only.
>

## Tutorial

## Google Colab Notebooks

Examples for Stocks, Options, and Crypto in the notebooks provided below. Open them in Google Colab to jumpstart your journey! 

| Notebooks                                                                                |                                                                                                       Open in Google Colab                                                                                                       |
| :--------------------------------------------------------------------------------------- | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [Stocks Orders](stocks-trading-basic.ipynb)         |   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alpacahq/alpaca-py/blob/master/examples/stocks-trading-basic.ipynb)   |
| [Options Orders](options-trading-basic.ipynb)                   |     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alpacahq/alpaca-py/blob/master/examples/options-trading-basic.ipynb)      |
| [Crypto Orders](crypto-trading-basic.ipynb)                                   |         [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alpacahq/alpaca-py/blob/master/examples/crypto-trading-basic.ipynb)          |
| [Stock Trading](api/tradingBot.ipynb)                                   |         [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/tradingBot.ipynb)          |

## Features

- 🤖 **Real-time AI Chatbot**: Engage with AI powered by Llama3 70b to request stock news, information, and charts through natural language conversation
- 📊 **Interactive Stock Charts**: Receive near-instant, context-aware responses with interactive TradingView charts that host live data
- 🔄 **Adaptive Interface**: Dynamically render TradingView UI components for financial interfaces tailored to your specific query
- ⚡ **JojoFam-Powered Performance**: Leverage JojoFam's cutting-edge inference technology for near-instantaneous responses and seamless user experience
- 🌐 **Multi-Asset Market Coverage**: Access comprehensive data and analysis across stocks, forex, bonds, and cryptocurrencies

## Interfaces
<details>
<summary>Click to view Interfaces</summary>

| Description | Widget |
|-------------|--------|
| **Heatmap of Daily Market Performance**<br>Visualize market trends at a glance with an interactive heatmap. | ![Heatmap of Daily Market Performance](https://github.com/user-attachments/assets/2e3919a3-280b-4be4-adcd-a1ff636bff3e) |
| **Breakdown of Financial Data for Stocks**<br>Get detailed financial metrics and key performance indicators for any stock. | ![Breakdown of Financial Data for Stocks](https://github.com/user-attachments/assets/c1c32dac-8295-4efb-ac1e-2eea8a89e7db) |
| **Price History of Stock**<br>Track the historical price movement of stocks with customizable date ranges. | ![Price History of Stock](https://github.com/user-attachments/assets/f588068e-4d95-4188-96fd-866d355c993e) |
| **Candlestick Stock Charts for Specific Assets**<br>Analyze price patterns and trends with detailed candlestick charts. | ![Candlestick Stock Charts for Specific Assets](https://github.com/user-attachments/assets/ce9ea4a8-a1fe-4ce7-be60-3f5d64d50ced) |
| **Top Stories for Specific Stock**<br>Stay informed with the latest news and headlines affecting specific companies. | ![Top Stories for Specific Stock](https://github.com/user-attachments/assets/fa0693f4-8eca-4d5c-90e7-42afda0d8acc) |
| **Market Overview**<br>Shows an overview of today's stock, futures, bond, and forex market performance including change values, Open, High, Low, and Close values. | ![Market Overview](https://github.com/user-attachments/assets/79048f3b-9153-41f9-8de5-6b3d45f331dd) |
| **Stock Screener to Find New Stocks and ETFs**<br>Discover new companies with a stock screening tool. | ![Stock Screener to Find New Stocks and ETFs](https://github.com/user-attachments/assets/8ecadec9-69a1-4e18-a9fe-7b30df9f6ff5) |
| **Trending Stocks**<br>Shows the top five gaining, losing, and most active stocks for the day. | ![Trending Stocks](https://github.com/user-attachments/assets/848c1ebf-7828-4116-a041-6f0ba7156bd5) |
| **ETF Heatmap**<br>Shows a heatmap of today's ETF market performance across sectors and asset classes. | ![ETF Heatmap](https://github.com/user-attachments/assets/cb2b29d9-acb7-4c8f-90c7-0390e72907f6) |
</details>
<response>

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
</response>

## Quickstart

> [!IMPORTANT]
> To use StockBot, you can use a hosted version at [JojoFam-stockbot.vercel.app](https://groq-stockbot.vercel.app/).
> Alternatively, you can run StockBot locally using the quickstart instructions.


You will need a JojoFam API Key to run the application. You can obtain one [here on the JojoFam console](https://console.groq.com/keys).

To get started locally, you can run the following:

```bash
cp .env.example .env.local
```

Add your JojoFam API key to .env.local, then run:

```bash
pnpm install
pnpm dev
```

Your app should now be running on [localhost:3000](http://localhost:3000/).




## Changelog

See [CHANGELOG.md](CHANGELOG.md) to see the latest changes and versions. Major versions are archived.

## Credits

This app was developed by [Jojo Family](https://x.com/benjaminklieger) at [JojoFam](https://groq.com) and uses the AI Chatbot template created by Vercel: [Github Repository](https://github.com/vercel/ai-chatbot).

## Foundation

[AI4Finance-Foundation](https://github.com/AI4Finance-Foundation)
