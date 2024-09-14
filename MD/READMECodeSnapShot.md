<br>
<div align="center">

[Stable Base Line+DOW](StableBasdelineDowJones.md)
[Introduction](OVERVIEW) |
[Experiments](READMExperiment.md) |
[PaperTrading](READMExpAlpacaPaperTrading.md) | 
[FAQ](READMEfaq.md)
</div>

## JojoFin
JojoFin is library designed to facilitate financial trading strategies using deep reinforcement learning (DRL). It is tailored specifically for quantitative finance, providing tools and frameworks for traderss to create, test, and implement trading dtrategis using DRL techniques.
<br>
[Experiments of StockBot](READMExperiment.md)
> Experiment of StockBot providing relevant, live, and interactive stock charts and interfaces

## Overview

StockBot is an AI-powered chatbot that leverages Llama3 70b on JojoFam, Vercel’s AI SDK, and TradingView’s live widgets to respond in conversation with live, interactive charts and interfaces specifically tailored to your requests. JojoFam's speed makes tool calling and providing a response near instantaneous, allowing for a sequence of two API calls with separate specialized prompts to return a response.

> [!IMPORTANT]
>  Note: StockBot may provide inaccurate information and does not provide investment advice. It is for entertainment and instructional use only.
>

## Tutorial

## Google Colab Notebooks

Examples for Stocks, Options, and Crypto in the notebooks provided below. Open them in Google Colab to jumpstart your journey! 

| Notebooks                                     |                                                                                    Open in Google Colab                                                                                    |
| :-------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [Stocks Orders](stocks-trading-basic.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alpacahq/alpaca-py/blob/master/examples/stocks-trading-basic.ipynb)  |
| [Options Orders](options-trading-basic.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alpacahq/alpaca-py/blob/master/examples/options-trading-basic.ipynb) |
| [Crypto Orders](crypto-trading-basic.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alpacahq/alpaca-py/blob/master/examples/crypto-trading-basic.ipynb)  |
| [Stock Trading](api/tradingBot.ipynb)         |                                                 [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](api/tradingBot.ipynb)                                                 |

## Features

- 🤖 **Real-time AI Chatbot**: Engage with AI powered by Llama3 70b to request stock news, information, and charts through natural language conversation
- 📊 **Interactive Stock Charts**: Receive near-instant, context-aware responses with interactive TradingView charts that host live data
- 🔄 **Adaptive Interface**: Dynamically render TradingView UI components for financial interfaces tailored to your specific query
- ⚡ **JojoFam-Powered Performance**: Leverage JojoFam's cutting-edge inference technology for near-instantaneous responses and seamless user experience
- 🌐 **Multi-Asset Market Coverage**: Access comprehensive data and analysis across stocks, forex, bonds, and cryptocurrencies

## Interfaces
<details>
<summary>Click to view Interfaces</summary>

| Description                                                                                                                                                        | Widget                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **Heatmap of Daily Market Performance**<br>Visualize market trends at a glance with an interactive heatmap.                                                        | ![Heatmap of Daily Market Performance](https://github.com/user-attachments/assets/2e3919a3-280b-4be4-adcd-a1ff636bff3e)          |
| **Breakdown of Financial Data for Stocks**<br>Get detailed financial metrics and key performance indicators for any stock.                                         | ![Breakdown of Financial Data for Stocks](https://github.com/user-attachments/assets/c1c32dac-8295-4efb-ac1e-2eea8a89e7db)       |
| **Price History of Stock**<br>Track the historical price movement of stocks with customizable date ranges.                                                         | ![Price History of Stock](https://github.com/user-attachments/assets/f588068e-4d95-4188-96fd-866d355c993e)                       |
| **Candlestick Stock Charts for Specific Assets**<br>Analyze price patterns and trends with detailed candlestick charts.                                            | ![Candlestick Stock Charts for Specific Assets](https://github.com/user-attachments/assets/ce9ea4a8-a1fe-4ce7-be60-3f5d64d50ced) |
| **Top Stories for Specific Stock**<br>Stay informed with the latest news and headlines affecting specific companies.                                               | ![Top Stories for Specific Stock](https://github.com/user-attachments/assets/fa0693f4-8eca-4d5c-90e7-42afda0d8acc)               |
| **Market Overview**<br>Shows an overview of today's stock, futures, bond, and forex market performance including change values, Open, High, Low, and Close values. | ![Market Overview](https://github.com/user-attachments/assets/79048f3b-9153-41f9-8de5-6b3d45f331dd)                              |
| **Stock Screener to Find New Stocks and ETFs**<br>Discover new companies with a stock screening tool.                                                              | ![Stock Screener to Find New Stocks and ETFs](https://github.com/user-attachments/assets/8ecadec9-69a1-4e18-a9fe-7b30df9f6ff5)   |
| **Trending Stocks**<br>Shows the top five gaining, losing, and most active stocks for the day.                                                                     | ![Trending Stocks](https://github.com/user-attachments/assets/848c1ebf-7828-4116-a041-6f0ba7156bd5)                              |
| **ETF Heatmap**<br>Shows a heatmap of today's ETF market performance across sectors and asset classes.                                                             | ![ETF Heatmap](https://github.com/user-attachments/assets/cb2b29d9-acb7-4c8f-90c7-0390e72907f6)                                  |
</details>
<response>

## Publications
<details>
<summary>Click to view publications</summary>

| Title                                                                                                           | Conference/Journal                                                                 | Link                                                                                                                                                                                                                                                 | Citations | Year |
| --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---- |
| Dynamic Datasets and Market Environments for Financial Reinforcement Learning                                   | Machine Learning - Springer Nature                                                 | [paper](https://arxiv.org/abs/2304.13174) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta)                                                                                                                                                | 7         | 2024 |
| **FinRL-Meta**: FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning | NeurIPS 2022                                                                       | [paper](https://arxiv.org/abs/2211.03107) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta)                                                                                                                                                | 37        | 2022 |
| **FinRL**: Deep reinforcement learning framework to automate trading in quantitative finance                    | ACM International Conference on AI in Finance (ICAIF)                              | [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3955949)                                                                                                                                                                                 | 49        | 2021 |
| **FinRL**: A deep reinforcement learning library for automated stock trading in quantitative finance            | NeurIPS 2020 Deep RL Workshop                                                      | [paper](https://arxiv.org/abs/2011.09607)                                                                                                                                                                                                            | 87        | 2020 |
| Deep reinforcement learning for automated stock trading: An ensemble strategy                                   | ACM International Conference on AI in Finance (ICAIF)                              | [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/2-Advance/FinRL_Ensemble_StockTrading_ICAIF_2020/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb) | 154       | 2020 |
| Practical deep reinforcement learning approach for stock trading                                                | NeurIPS 2018 Workshop on Challenges and Opportunities for AI in Financial Services | [paper](https://arxiv.org/abs/1811.07522) [code](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/FinRL/tree/master/examples))                                                              | 164       | 2018 |
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

<response>
# Integrate RDL Trainer into JojoBot's Backend with Next.js and Python

Here's a structured approach to integrate an RDL trainer into your Next.js serverless backend for your financial data analysis application:

## 1. Serverless Next.js Architecture

You will use serverless functions with Vercel to run your backend code. Ensure that your Next.js project is configured for serverless deployment.

## 2. React Code in `app` and `components` Folders

### Folder Structure

```
/jojobot
  /app
    /train
      page.tsx
  /components
    DataDisplay.tsx
  /public
  /styles
  /node_modules
  package.json
  next.config.js
```

### React Component (`DataDisplay.tsx`)

```typescript
// components/DataDisplay.tsx
import React, { useEffect, useState } from 'react';

const DataDisplay: React.FC = () => {
  const [data, setData] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch('/api/train');
      const result = await response.json();
      setData(result.result);
    };

    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every 1 minute

    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  return <div>{data}</div>;
};

export default DataDisplay;
```

### Page Component (`page.tsx`)

```typescript
// app/train/page.tsx
import React from 'react';
import DataDisplay from '../../components/DataDisplay';

const TrainPage: React.FC = () => {
  return (
    <div>
      <h1>Financial Data Analysis</h1>
      <DataDisplay />
    </div>
  );
};

export default TrainPage;
```

## 3. Python Script for Financial Data and Trading

### `train.py`

```python
# train.py
import finrl
from finrl.algos import PPO
from alpaca_trade_api import REST
import datetime
import pytz

# Initialize Alpaca API
api = REST('APCA_API_KEY_ID', 'APCA_API_SECRET_KEY', base_url='https://paper-api.alpaca.markets')

def main():
    # Retrieve historical data
    data = finrl.get_data()
    
    # Initialize PPO agent and train
    agent = PPO()
    agent.train(data)

    # Example action decision
    action = agent.predict(data)

    # Place an order based on the action
    symbol = 'AAPL'
    if action == 'buy':
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='buy',
            type='limit',
            time_in_force='gtc',
            limit_price=150
        )
    elif action == 'sell':
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='sell',
            type='limit',
            time_in_force='gtc',
            limit_price=155
        )
    # Implement 'hold' action if needed

if __name__ == "__main__":
    main()
```

## 4. Running Python Script Programmatically Every 1 Minute

### Schedule Python Script Using `cron` (On Vercel)

If using Vercel, you might need to schedule the script externally (e.g., via a cron job on a separate server or using a cloud function with scheduling capability).

Alternatively, you can use a local scheduler if running the script in a local or cloud VM environment.

```bash
# Example cron job to run every minute (on a Unix-based system)
* * * * * /usr/bin/python3 /path/to/train.py
```

## 🌐 Sources
1. [workik.com - AI for Backend & Frontend Code Development](https://workik.com/)
2. [nextjs.org - Server and Client Components](https://nextjs.org/learn/react-foundations/server-and-client-components)
3. [docs.alpaca.markets - Orders at Alpaca](https://docs.alpaca.markets/docs/orders-at-alpaca)
4. [medium.com - Proximal Policy Optimization(PPO) for trading environment](https://medium.com/@sthanikamsanthosh1994/reinforcement-learning-part-8-proximal-policy-optimization-ppo-for-trading-9f1c3431f27d)
</response>

make the above code to results in the form of text and images that being saved to get saved locally and get displayed on the fromt end page 

To modify the code so that the results of running the Python script are saved locally as text and images, and then displayed on the front end, follow the steps below.

### 1. Modify the Serverless Function to Save Results Locally

Update the serverless function to save the output as text and images on the server.

```typescript
// pages/api/triggerScript.ts
import { NextApiRequest, NextApiResponse } from 'next';
import { exec } from 'child_process';
import path from 'path';
import fs from 'fs';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const outputDir = path.join(process.cwd(), 'public', 'results');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const scriptPath = '/path/to/train.py';
  exec(`python3 ${scriptPath}`, (error, stdout, stderr) => {
    if (error || stderr) {
      const errorMessage = error ? error.message : stderr;
      console.error(`Error: ${errorMessage}`);
      return res.status(500).json({ success: false, message: errorMessage });
    }

    // Save the text output
    const outputFilePath = path.join(outputDir, 'output.txt');
    fs.writeFileSync(outputFilePath, stdout);

    // Assume the Python script saves images in the outputDir
    const images = fs.readdirSync(outputDir).filter(file => file.endsWith('.png'));

    res.status(200).json({ success: true, message: 'Script executed', images });
  });
}
```

### 2. Modify the Python Script to Save Text and Images

Ensure that your Python script saves its output as text and images in the `public/results` directory. Here's a simple example:

```python
# train.py
import matplotlib.pyplot as plt

# Sample data for demonstration purposes
data = [1, 2, 3, 4, 5]
output_text = "This is the result of the training process.\n"

# Save the output text
with open('public/results/output.txt', 'w') as f:
    f.write(output_text)

# Save a plot as an image
plt.plot(data)
plt.title("Sample Plot")
plt.savefig('public/results/plot.png')
```

### 3. Update the Frontend to Display Text and Images

Modify the `DataDisplay` component to fetch and display the saved text and images.

```typescript
// components/DataDisplay.tsx
import React, { useState } from 'react';

const DataDisplay: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [output, setOutput] = useState<string | null>(null);
  const [images, setImages] = useState<string[]>([]);

  const startSchedule = () => {
    setIsRunning(true);
    const intervalId = setInterval(async () => {
      const response = await fetch('/api/triggerScript');
      const result = await response.json();

      if (result.success) {
        // Fetch the text output
        const textResponse = await fetch('/results/output.txt');
        const textData = await textResponse.text();
        setOutput(textData);

        // Update image paths
        const imagePaths = result.images.map((img: string) => `/results/${img}`);
        setImages(imagePaths);
      }
    }, 60000);

    (window as any).intervalId = intervalId;
  };

  const stopSchedule = () => {
    setIsRunning(false);
    clearInterval((window as any).intervalId);
  };

  return (
    <div>
      <h1>Python Script Runner</h1>
      <button onClick={startSchedule} disabled={isRunning}>
        Start Running Every 1 Minute
      </button>
      <button onClick={stopSchedule} disabled={!isRunning}>
        Stop Running
      </button>

      {output && (
        <div>
          <h2>Output:</h2>
          <pre>{output}</pre>
        </div>
      )}

      {images.length > 0 && (
        <div>
          <h2>Generated Images:</h2>
          {images.map((src, idx) => (
            <img key={idx} src={src} alt={`Result ${idx + 1}`} />
          ))}
        </div>
      )}
    </div>
  );
};

export default DataDisplay;
```

### 4. Use the `DataDisplay` Component in Your Page

Include the `DataDisplay` component in the desired page.

```typescript
// app/train/page.tsx
import React from 'react';
import DataDisplay from '../../components/DataDisplay';

const TrainPage: React.FC = () => {
  return (
    <div>
      <h1>Financial Data Analysis</h1>
      <DataDisplay />
    </div>
  );
};

export default TrainPage;
```

### Summary
- The Python script runs every 1 minute when triggered by a button.
- The results, including text and images, are saved to the `public/results` directory.
- The front-end component fetches and displays the saved results on the page.

## 🌐 Sources
1. [stackoverflow.com - How to execute a function every 1 minute](https://stackoverflow.com/questions/75767160/how-to-execute-a-function-every-1-minute)
2. [advsyscon.com - Python Job Scheduling with Cron](https://www.advsyscon.com/blog/python-job-scheduling/)
3. [discuss.python.org - Help me to initialize or trigger a python file to run every 2 hours](https://discuss.python.org/t/help-me-to-initialize-or-trigger-a-python-file-to-run-in-every-2-hrs/51347)
</response>

To set up your Next.js project to run a Python script, save the results locally, and display them on the front end, follow these steps:

### 1. **Run the Python Script in Next.js:**
   - Use Node.js’s `child_process` module to execute the Python script from within an API route in your Next.js project.
   - Example code (TypeScript):

   ```typescript
   import { exec } from 'child_process';
   import type { NextApiRequest, NextApiResponse } from 'next';

   export default function handler(req: NextApiRequest, res: NextApiResponse) {
     exec('python3 ./../../scripts/train.py', (error, stdout, stderr) => {
       if (error) {
         res.status(500).json({ error: stderr });
       } else {
         // Save stdout to a file or return the output directly
         res.status(200).json({ message: 'Script executed successfully', output: stdout });
       }
     });
   }
   ```

### 2. **Save Results Locally:**
   - You can save the output of the Python script to a file using Node.js’s `fs` module.
   - Example:

   ```typescript
   import { exec } from 'child_process';
   import fs from 'fs';
   import type { NextApiRequest, NextApiResponse } from 'next';

   export default function handler(req: NextApiRequest, res: NextApiResponse) {
     exec('python3 ./../../scripts/train.py', (error, stdout, stderr) => {
       if (error) {
         res.status(500).json({ error: stderr });
       } else {
         // Save the output to a file
         fs.writeFileSync('./results/output.txt', stdout);
         res.status(200).json({ message: 'Script executed and result saved successfully' });
       }
     });
   }
   ```

### 3. **Display the Results on the Front End:**
   - Create a React component to fetch and display the saved results.

   ```typescript
   import { useEffect, useState } from 'react';

   function DataDisplay() {
     const [textResult, setTextResult] = useState('');

     useEffect(() => {
       fetch('/api/get-results')  // This API route should return the saved results
         .then((res) => res.text())
         .then((data) => setTextResult(data));
     }, []);

     return (
       <div>
         <h2>Results</h2>
         <p>{textResult}</p>
       </div>
     );
   }

   export default DataDisplay;
   ```

### Summary
- **Run the Python script** using a custom API route.
- **Save the script's output** locally in a file.
- **Fetch and display the saved output** in a front-end component.

This setup will allow your Next.js application to integrate Python scripts, process their results, and present them to users effectively.

## 🌐 Sources
- [stackoverflow.com - Running python scripts in Next.js](https://stackoverflow.com/questions/68013605/running-python-scripts-in-next-js)
- [docs.deque.com - Save Results Locally](https://docs.deque.com/devtools-mobile/2023.8.16/en/android-save-result)
- [docs.formvibes.com - Display Data on Frontend](https://docs.formvibes.com/article/121-display-data-on-the-frontend)