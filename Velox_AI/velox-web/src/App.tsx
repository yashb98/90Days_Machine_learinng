// src/App.tsx

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import AgentFlowBuilder from './pages/AgentFlowBuilder'

// Home page component
function HomePage() {
  return (
    <div className="min-h-screen bg-background p-8">
      <div className="mx-auto max-w-6xl space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold tracking-tight">Velox AI</h1>
            <p className="text-muted-foreground">Enterprise AI Voice Agent Platform</p>
          </div>
          <Badge variant="secondary">Tailwind v4 ✨</Badge>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl font-bold">24</CardTitle>
              <CardDescription>Active Agents</CardDescription>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl font-bold">1,234</CardTitle>
              <CardDescription>Total Conversations</CardDescription>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl font-bold">$459.32</CardTitle>
              <CardDescription>Total Cost</CardDescription>
            </CardHeader>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Latest Stack</CardTitle>
            <CardDescription>Built with the newest technologies</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <Badge>Vite 7</Badge>
              <Badge>React 19</Badge>
              <Badge>TypeScript 5</Badge>
              <Badge variant="secondary">Tailwind CSS 4</Badge>
              <Badge>TanStack Query</Badge>
              <Badge>Zustand</Badge>
            </div>
            <div className="flex gap-2">
              <Button asChild>
                <Link to="/agents/demo/flow">Open Flow Builder</Link>
              </Button>
              <Button variant="outline">Learn More</Button>
              <Button variant="secondary">Documentation</Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/agents/:agentId/flow" element={<AgentFlowBuilder />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App