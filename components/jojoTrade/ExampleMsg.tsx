// Description: This file contains utility functions that are used across the project.

// Importing the useState and useEffect hooks from React.
"use client";

import * as React from 'react'
import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { PromptForm } from '@/components/prompt-form'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { IconShare } from '@/components/ui/icons'
import { FooterText } from '@/components/footer'
import { useAIState, useActions, useUIState } from 'ai/rsc'
import type { AI } from '@/lib/chat/actions'
import { nanoid } from 'nanoid'
import { ColabNotebooks } from '@/components/jojoTrade/ColabNotebooks'

export function ExampleMessage() {

    interface ExampleMessage {
        heading: string
        subheading: string
        message: string
    }

    const exampleMessages = [
        {
            heading: 'What is the price',
            subheading: 'of Apple Inc.?',
            message: 'What is the price of Apple stock?'
        },
        {
            heading: 'Show me a stock chart',
            subheading: 'for $GOOGL',
            message: 'Show me a stock chart for $GOOGL'
        },
        {
            heading: 'What are some recent',
            subheading: `events about Amazon?`,
            message: `What are some recent events about Amazon?`
        },
        {
            heading: `What are Microsoft's`,
            subheading: 'latest financials?',
            message: `What are Microsoft's latest financials?`
        },
        {
            heading: 'How is the stock market',
            subheading: 'performing today by sector?',
            message: `How is the stock market performing today by sector?`
        },
        {
            heading: 'Show me a screener',
            subheading: 'to find new stocks',
            message: 'Show me a screener to find new stocks'
        }
    ]

    const [randExamples, setRandExamples] = useState<ExampleMessage[]>([])

    useEffect(() => {
        const shuffledExamples = [...exampleMessages].sort(
            () => 0.5 - Math.random()
        )
        setRandExamples(shuffledExamples)
    }, [])
    
    return (
        <div>
            {/* {randExamples.map((example, index) => (
                <div key={index}>
                    <h4>{example.heading}</h4>
                    <p>{example.subheading}</p>
                    <p>{example.message}</p>
                </div>
            ))} */}
        </div>
    )
}

