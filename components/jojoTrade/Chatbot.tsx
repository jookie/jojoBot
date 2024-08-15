import React, { useState } from 'react'

export function Chatbot() {
  const [input, setInput] = useState('')
  const [result, setResult] = useState('')

  const handleSubmit = async () => {
    const res = await fetch('/api/run-drl', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ input })
    })

    const data = await res.json()
    setResult(data.result || 'Error running script')
  }

  return (
    <div>
      <input
        type="text"
        value={input}
        onChange={e => setInput(e.target.value)}
        placeholder="Enter your message"
      />
      <button onClick={handleSubmit}>Run Script</button>
      <div>{result ? `Result: ${result}` : 'Waiting for result...'}</div>
    </div>
  )
}
