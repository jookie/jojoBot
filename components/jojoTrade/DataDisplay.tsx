// /jojobot/components/DataDisplay.tsx
'use client'
import { useEffect, useState } from 'react';

const DataDisplay: React.FC = () => {
  const [textResult, setTextResult] = useState<string>('')
  const [imageSrc, setImageSrc] = useState<string>('')

  useEffect(() => {
    // Fetch the text result
    fetch('/results/result.txt')
      .then(res => res.text())
      .then(data => setTextResult(data))

    // Set the image source
    setImageSrc('/results/result.png')
  }, [])

  return (
    <div>
      <h2>Results</h2>
      <p>{textResult}</p>
      {imageSrc && <img src={imageSrc} alt="Result Plot" />}
    </div>
  )
}

export default DataDisplay;
