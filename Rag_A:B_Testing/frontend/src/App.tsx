import { useState } from 'react' // Keep if you plan to use state, otherwise remove it
import reactLogo from './assets/react.svg' // Remove if not used
import viteLogo from '/vite.svg' // Remove if not used
import './App.css' // Keep for styling

// Corrected import: Omit the .tsx extension and remove the extra 'import React from 'react';'
import Abtest from './components/ABtest'; 


const App: React.FC = () => {
  // If you are not using useState, you should remove the import from the top
  // const [count, setCount] = useState(0) 
  
  return (
    <div className="App">
      <header className="App-header">
        {/* If you wanted to use the logos, you'd place them here: 
        <img src={viteLogo} className="logo" alt="Vite logo" /> 
        */}
        <h1>My A/B Testing Setup</h1>
      </header>
      
      {/* This part is correct, assuming the file path is right */}
      <Abtest /> 
      
      <main>
        {/* Other main content */}
      </main>
    </div>
  );
};

export default App;