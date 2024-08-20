// /jojobot/components/DataDisplay.tsx
'use client'
import { useEffect, useState } from 'react';

function DataDisplay() {
  const [textResult, setTextResult] = useState('');
  const [imageSrc, setImageSrc] = useState('');

  useEffect(() => {
    // Fetch the text result
    fetch('/results/result.txt')
      .then((res) => res.text())
      .then((data) => setTextResult(data));

    // Set the image source
    setImageSrc('/results/result.png');
  }, []);

  return (
    <div>
      <h2>Results</h2>
      <p>{textResult}</p>
      {imageSrc && <img src={imageSrc} alt="Result Plot" />}
    </div>
  );
}

export default DataDisplay;
