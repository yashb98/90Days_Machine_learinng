// src/App.tsx


import { BrowserRouter, Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import AgentFlowBuilder from './pages/AgentFlowBuilder'
import Playground from './pages/Playground'
import Billing from './pages/Billing'



function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/agents/:agentId/flow" element={<AgentFlowBuilder />} />
        <Route path="/agents/:agentId/playground" element={<Playground />} />
        <Route path="/billing" element={<Billing />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App

