// ToggleSwitch.js (React)
"use client";

import React, { useState } from 'react'

export function ToggleSwitch() {
  const [isToggled, setIsToggled] = useState(false)
  const [output, setOutput] = useState('')

  const handleToggle = async () => {
    setIsToggled(!isToggled)

    if (!isToggled) {
      // Fetch backtest result from Python backend
      const response = await fetch('/run-backtest')
      const data = await response.json()
      setOutput(data.output) // Display backtest result in the browser
    } else {
      setOutput('')
    }
  }

  return (
    <div>
      <label>
        <input type="checkbox" checked={isToggled} onChange={handleToggle} />
        Run Backtest
      </label>
      <div>
        <pre>{output}</pre>
      </div>
    </div>
  )
}

