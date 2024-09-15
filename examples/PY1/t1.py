# import { NextResponse } from 'next/server'
# import { exec } from 'child_process'
# import { promisify } from 'util'
# import path from 'path'

# const PYscriptPath = path.join(process.cwd(), 'py', 'greet.py')

# const execPromise = promisify(exec)

# export async function GET() {
#   try {
#     const { stdout, stderr } = await execPromise('python3 ' + PYscriptPath)
#     if (stderr) {
#       throw new Error(stderr)
#     }
#     return NextResponse.json({ output: stdout })
#   } catch (error: any) {
#     return NextResponse.json({ error: error.message }, { status: 500 })
#   }
# }

# from http.server import BaseHTTPRequestHandler
 
# class handler(BaseHTTPRequestHandler):
 
#     def do_GET(self):
#         self.send_response(200)
#         self.send_header('Content-type','text/plain')
#         self.end_headers()
#         self.wfile.write('Hello, world!'.encode('utf-8'))
#         return

# // /jojobot/pages/api/run-script.ts
# import { exec } from 'child_process';
# import type { NextApiRequest, NextApiResponse } from 'next';

# export default function handler(req: NextApiRequest, res: NextApiResponse) {
#   exec('python ./../../scripts/train.py', (error, stdout, stderr) => {
#     if (error) {
#       res.status(500).json({ error: stderr });
#     } else {
#       res.status(200).json({ message: 'Script executed successfully' });
#     }
#   });
# }
