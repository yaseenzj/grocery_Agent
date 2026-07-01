import React, { useState, useEffect } from 'react';
import { ReceiptUploader } from './components/ReceiptUploader';
import { InventoryDashboard } from './components/InventoryDashboard';
import { RecipeAdvisor } from './components/RecipeAdvisor';

function App() {
  const [inventory, setInventory] = useState<any>({ items: [] });

  // Initial load would typically fetch from a GET /inventory endpoint.
  // Since we don't have one in this API, the inventory will populate 
  // either when we upload a receipt or consume an item.
  
  return (
    <div className="container">
      <div style={{ textAlign: 'center', marginBottom: '48px' }} className="animate-fade-in">
        <h1 style={{ fontSize: '3rem', margin: '0 0 16px 0', background: 'linear-gradient(to right, var(--accent-primary), #d946ef)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Grocery Agent
        </h1>
        <p className="text-secondary" style={{ fontSize: '1.2rem', maxWidth: '600px', margin: '0 auto' }}>
          Your AI-powered sous chef. Upload receipts, track your fridge, and generate recipes before food goes bad.
        </p>
      </div>

      <div className="grid-layout">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
          <ReceiptUploader onUploadComplete={(data) => setInventory(data)} />
          <InventoryDashboard inventory={inventory} onInventoryUpdate={(data) => setInventory(data)} />
        </div>
        
        <div>
          <RecipeAdvisor />
        </div>
      </div>
    </div>
  );
}

export default App;
