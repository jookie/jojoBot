import { UseChatHelpers } from 'ai/react'
import { Button } from '@/components/ui/button'
import { ExternalLink } from '@/components/external-link'
import { IconArrowRight } from '@/components/ui/icons'

export function EmptyScreen() {
  return (
    <div className="mx-auto max-w-2xl px-4">
      <div className="flex flex-col gap-2 border bg-background p-8">
        <h1 className="text-lg font-semibold">
          Welcome to the Jojo's TraderBot powered by Jojo-Trades!
        </h1>
        <p className="leading-normal text-sm">
          Render TradingView
          stock market widgets.{' Jojo '}
          <span className="font-muted-foreground">
            Built with{' JojoInfoSoft'}
            {/* <ExternalLink href="https://github.com/jookie/jojoBot">
              Jojo AI SDK{' '}
            </ExternalLink>
            <ExternalLink href="https://tradingview.com">
              , TradingView Widgets
            </ExternalLink>
            , and powered by{' '}
            <ExternalLink href="https://groq.com">Jojo</ExternalLink> */}
          </span>
        </p>
      </div>
    </div>
  )
}
