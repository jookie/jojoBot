// Users/dovpeles/workspace/jojobot/app/api/runPythonScript/route.ts
import { NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import path from 'path'

const PYscriptPath = path.join(process.cwd(), 'py', 'greet.py')

const execPromise = promisify(exec)

export async function GET() {
  try {
    const { stdout, stderr } = await execPromise('python3 ' + PYscriptPath)
    if (stderr) {
      throw new Error(stderr)
    }
    return NextResponse.json({ output: stdout })
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
