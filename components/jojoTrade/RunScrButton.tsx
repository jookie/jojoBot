// Run Script.tsx
import React from 'react'

// RunScrButton component
export function RunScrButton() {
  const runScript = async () => {
    try {
      // Fetch the run-script API endpoint and send a POST request with the scriptName parameter set to 'run-script'   
      // /Users/dovpeles/workspace/WorkPlace/app/api/run-script.ts
      const response = await fetch(
        '/Users/dovpeles/workspace/WorkPlace/app/api/',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scriptName: 'run-script' })
        }
      )
      const data = await response.json()
      console.log(data.output)
    } catch (error) {
      console.error('Error running script:', error)
    }
  }

  return (
    <button onClick={runScript}>Run Stock Training Script</button>
  )
}

