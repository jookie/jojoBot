// Run Script.tsx
import React from 'react'

// RunScrButton component
export function RunScrButton() {
  const runScript = async () => {
    try {
      const response = await fetch('/api/run-script', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scriptName: 'run-script' })
      })
      const data = await response.json()
      console.log(data.output)
    } catch (error) {
      console.error('Error running script:', error)
    }
  }

  return (
    <button onClick={runScript}>Run Training Script</button>
  )
}

