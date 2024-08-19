// components/DataDisplay.tsx
import React, { useEffect, useState } from 'react'


const DataDisplay: React.FC = () => {
  const [data, setData] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch('/api/train')
      const result = await response.json()
      setData(result.result)
    }

    fetchData()
    const interval = setInterval(fetchData, 60000) // Refresh every 1 minute

    return () => clearInterval(interval) // Cleanup on unmount
  }, [])

  return <div>{data}</div>
}

export default DataDisplay
