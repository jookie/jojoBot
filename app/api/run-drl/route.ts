// app/api/run-drl/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';

export async function GET(request: NextRequest) {
    exec('python3 path/to/your/script.py', (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${error.message}`);
            return NextResponse.json({ error: error.message });
        }
        if (stderr) {
            console.error(`Stderr: ${stderr}`);
            return NextResponse.json({ stderr });
        }
        console.log(`Stdout: ${stdout}`);
        return NextResponse.json({ result: stdout });
    });
}
/* 
Answer
Here’s a basic example of how to create a Next.js component that runs an AI Deep Reinforcement Learning (DRL) Python script when a button is pressed. The Python script will be executed via an API route using Node.js's child_process module.

Step 1: Create an API Route to Run the Python Script
First, create an API route in your Next.js project that executes the Python script.

typescript

Answer
Here’s a basic example of how to create a Next.js component that runs an AI Deep Reinforcement Learning (DRL) Python script when a button is pressed. The Python script will be executed via an API route using Node.js's child_process module.

Step 1: Create an API Route to Run the Python Script
First, create an API route in your Next.js project that executes the Python script.

typescript

 */
