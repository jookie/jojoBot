// /jojobot/app/train/page.tsx
import RunScriptButton from './../../components/jojoTrade/RunScriptButton'
import DataDisplay from     './../../components/jojoTrade/DataDisplay'

export function TrainPage() {
  return (
    <div>
      <h1>Run Training</h1>
      <RunScriptButton />
      <DataDisplay />
    </div>
  )
}
