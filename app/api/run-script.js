// /jojobot/pages/api/run-script.js
import { exec } from 'child_process';

export default function handler(req, res) {
  exec('python3 ./../../scripts/train.py', (error, stdout, stderr) => {
    if (error) {
      res.status(500).json({ error: stderr });
    } else {
      res.status(200).json({ message: 'Script executed successfully' });
    }
  });
}
