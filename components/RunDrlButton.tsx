// app/components/RunDrlButton.tsx
'use client'

import { useState } from 'react'

export default function RunDrlButton() {
  const [result, setResult] = useState<string | null>(null)

  const handleClick = async () => {
    const response = await fetch('/api/run-drl')
    const data = await response.json()
    setResult(data.result || data.error)
  }

  return (
    <div>
      <button onClick={handleClick}>Run AI DRL Script</button>
      <br />
      <textarea
        value={result || ''}
        placeholder="Script output will appear here..."
        readOnly
        rows={10}
        cols={50}
      />
    </div>
  )
}
