import { useState, useEffect } from 'react';
import { InventoryDashboard } from './components/InventoryDashboard';
import { RecipeAdvisor } from './components/RecipeAdvisor';
import { api } from './api';

function App() {
  const [inventory, setInventory] = useState<any>({ items: [] });
  const [activeTab, setActiveTab] = useState<'fridge' | 'recipes'>('fridge');

  useEffect(() => {
    api.getInventory().then(setInventory).catch(console.error);
  }, []);
  
  return (
    <div className="app-wrapper">
      <header className="app-header">
        <h1>{activeTab === 'fridge' ? 'Smart Fridge' : 'Recipes'}</h1>
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'fridge' ? 'active' : ''}`}
            onClick={() => setActiveTab('fridge')}
          >
            My Fridge
          </button>
          <button 
            className={`tab ${activeTab === 'recipes' ? 'active' : ''}`}
            onClick={() => setActiveTab('recipes')}
          >
            Recipe Advisor
          </button>
        </div>
      </header>

      <main className="panel">
        {activeTab === 'fridge' && (
          <InventoryDashboard inventory={inventory} onInventoryUpdate={(data) => setInventory(data)} />
        )}
        
        {activeTab === 'recipes' && (
          <RecipeAdvisor />
        )}
      </main>
    </div>
  );
}

export default App;
