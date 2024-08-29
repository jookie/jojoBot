// /Users/dovpeles/workspace/jojobot/components/MessageFetcher.tsx
"use client";

import React, { useEffect, useState } from "react";

const MessageFetcher: React.FC = () => {
  const [time, setTime] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTime = async () => {
      try {
        const response = await fetch("/api/runPythonScript");
        if (!response.ok) {
          throw new Error(
            `Network response was not ok. Status: ${response.status}`
          );
        }
        const data = await response.json();
        if (data.error) {
          throw new Error(`API error: ${data.error}`);
        }
        setTime(data.output);;
      } catch (err) {
        setError((err as Error).message);
      }
    };

    fetchTime();
  }, []);

  return (
    <div>
      {error ? (
        <p>Error: {error}</p>
      ) : time ? (
        <p>Current server time: {time}</p>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
};

export default MessageFetcher;
