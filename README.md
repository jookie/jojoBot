<h2 align="center">
 <br>
 <!-- https://imgur.com/a/0hF6gSL -->
 <img src="public/groqlabs-logo-black2.png" alt="AI StockBot" width="500">
 <br>
 <br>
 StockBot Powered by JojoFam: Lightning Fast AI Chatbot that Responds With Live Interactive Stock Charts, Financials, News, Screeners, and More 
 <br>
</h2>

<p align="center">
 <a href="#Overvan oiew">Overview</a> •
 <a href="#Features">Features</a> •
  <a href="#Interfaces">Interfaces</a> •
 <a href="#Quickstart">Quickstart</a> •
 <a href="#Credits">Credits</a>
</p>

## JojoFin
JojoFin is library designed to facilitate the development of financial trading strategies using deep reinforcement learning (DRL). It is tailored specifically for quantitative finance, providing tools and frameworks for practitioners to create, test, and implement trading algorithms using DRL techniques.

## [Integration]
Source code How to integrate the FinRl PPO, SAC, and TD3 algorithms into a StockBot powered by Groq,
I have python scripts of Source code that I want to integrate the FinRl PPO, SAC, and TD3 algorithms into a StockBot powered by Groq , How do I run these script from the web page called page.tsx these python scripts ?

Make Source code to integrate run python script file On a react and next libs



<br>

[Demo of StockBot](https://github.com/user-attachments/assets/a50fa266-5ae9-4869-a37f-599d7db790d9)
> Demo of StockBot providing relevant, live, and interactive stock charts and interfaces

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fbklieger-groq%2Fstockbot-on-groq&env=GROQ_API_KEY&envDescription=Get%20a%20Groq%20API%20Key&envLink=https%3A%2F%2Fconsole.groq.com%2Fkeys&project-name=stockbot-clone&repository-name=stockbot-clone&demo-title=StockBot&demo-description=Build%20a%20lightning-fast%20AI%20chatbot%20powered%20by%20Groq%20and%20Vercel%20AI%20SDK%20that%20responds%20with%20live%20stock%20charts%2C%20financials%2C%20news%2C%20and%20screeners.&demo-url=https%3A%2F%2Fgroq-stockbot.vercel.app%2F&demo-image=https%3A%2F%2Fi.imgur.com%2FjJfm8mm.png)

## Overview

StockBot is an AI-powered chatbot that leverages Llama3 70b on JojoFam, Vercel’s AI SDK, and TradingView’s live widgets to respond in conversation with live, interactive charts and interfaces specifically tailored to your requests. JojoFam's speed makes tool calling and providing a response near instantaneous, allowing for a sequence of two API calls with separate specialized prompts to return a response.

> [!IMPORTANT]
>  Note: StockBot may provide inaccurate information and does not provide investment advice. It is for entertainment and instructional use only.

## Features

- 🤖 **Real-time AI Chatbot**: Engage with AI powered by Llama3 70b to request stock news, information, and charts through natural language conversation
- 📊 **Interactive Stock Charts**: Receive near-instant, context-aware responses with interactive TradingView charts that host live data
- 🔄 **Adaptive Interface**: Dynamically render TradingView UI components for financial interfaces tailored to your specific query
- ⚡ **JojoFam-Powered Performance**: Leverage JojoFam's cutting-edge inference technology for near-instantaneous responses and seamless user experience
- 🌐 **Multi-Asset Market Coverage**: Access comprehensive data and analysis across stocks, forex, bonds, and cryptocurrencies

## Interfaces
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
