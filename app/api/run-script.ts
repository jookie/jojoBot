// /jojobot/pages/api/run-script.ts
import { exec } from 'child_process';
import type { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  exec('python3 ./../../scripts/train.py', (error, stdout, stderr) => {
    if (error) {
      res.status(500).json({ error: stderr });
    } else {
      res.status(200).json({ message: 'Script executed successfully' });
    }
  });
}
