// RunScriptButton.tsx by chatGPT

import React from 'react';

export const RunScriptButton: React.FC = () => {
  const runScript = async () => {
    try {
      const response = await fetch('/api/run-script', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scriptName: 'path/to/your_script.py' }),
      });
      const data = await response.json();
      console.log(data.output);
    } catch (error) {
      console.error('Error running script:', error);
    }
  };

  return <button onClick={runScript}>Run Python Script</button>;
};

// export default RunScriptButton;
