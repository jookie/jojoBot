// /jojobot/app/train/page.tsx
import { useState } from 'react'

function RunScriptButton() {
  const [status, setStatus] = useState('')

  const runScript = async () => {
    const res = await fetch('/api/run-script')
    const data = await res.json()
    setStatus(data.message || data.error)
  }

  return (
    <div>
      <button onClick={runScript}>Run Script</button>
      <p>{status}</p>
    </div>
  )
}

export default RunScriptButton
