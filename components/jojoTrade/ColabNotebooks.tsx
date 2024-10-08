// RunScriptButton.tsx by chatGPT
import React from 'react'
const notebooks = [
  {
    name: 'Stock Order',
    file: 'stocks-trading-basic.ipynb',
    link: 'https://colab.research.google.com/github/alpacahq/alpaca-py/blob/master/examples/stocks-trading-basic.ipynb'
  },
  {
    name: 'Option Order',
    file: 'options-trading-basic.ipynb',
    link: 'https://colab.research.google.com/github/alpacahq/alpaca-py/blob/master/examples/options-trading-basic.ipynb'
  },
  {
    name: 'Crypto Order',
    file: 'crypto-trading-basic.ipynb',
    link: 'https://colab.research.google.com/github/alpacahq/alpaca-py/blob/master/examples/crypto-trading-basic.ipynb'
  },
  {
    name: 'Stock Ensemble',
    file: 'FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb',
    link: 'https://colab.research.google.com/github/AI4Finance-Foundation/FinRL-Tutorials/blob/master/2-Advance/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb'
  }
]  
import { cn } from '@/lib/utils'

export function ColabNotebooks({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      className={cn(
        'px-2 text-center text-xs leading-normal text-muted-foreground',
        className)} >
      <table>
        <thead>
          <tr>
            <th>Notebooks</th>
            <th>Open in Google Colab</th>
          </tr>
        </thead>
        <tbody>
          {notebooks.map(notebook => (
            <tr key={notebook.name}>
              <td>
                <a
                  href={notebook.link}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {notebook.name}
                </a>
              </td>
              <td>
                <a
                  href={notebook.link}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <img
                    src="https://colab.research.google.com/assets/colab-badge.svg"
                    alt={`Open ${notebook.name} in Google Colab`}
                  />
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

