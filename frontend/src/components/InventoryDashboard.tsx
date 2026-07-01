import React, { useState } from 'react';
import { Package, Utensils, AlertTriangle, Send, Trash2 } from 'lucide-react';
import { api } from '../api';

export function InventoryDashboard({ inventory, onInventoryUpdate }: { inventory: any, onInventoryUpdate: (data: any) => void }) {
  const [statement, setStatement] = useState('');
  const [loading, setLoading] = useState(false);

  const handleConsume = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!statement.trim()) return;
    
    setLoading(true);
    try {
      const data = await api.consumeItem(statement);
      onInventoryUpdate(data);
      setStatement('');
    } catch (err) {
      alert("Failed to record consumption.");
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (itemName: string) => {
    if (!window.confirm(`Are you sure you want to remove "${itemName}" from your fridge?`)) return;
    setLoading(true);
    try {
      const data = await api.removeItem(itemName);
      onInventoryUpdate(data);
    } catch (err) {
      alert("Failed to remove item.");
    } finally {
      setLoading(false);
    }
  };

  const items = inventory?.items || [];

  
  // Calculate days until expiration to determine badge color
  const getExpirationBadge = (dateString: string) => {
    const exp = new Date(dateString);
    const now = new Date();
    const days = Math.ceil((exp.getTime() - now.getTime()) / (1000 * 3600 * 24));
    
    if (days < 0) return <span className="badge badge-danger">Expired</span>;
    if (days <= 3) return <span className="badge badge-danger">Expiring Soon ({days}d)</span>;
    if (days <= 7) return <span className="badge badge-warning">Expires in {days}d</span>;
    return <span className="badge badge-success">Good ({days}d left)</span>;
  };

  return (
    <div className="glass-panel animate-fade-in" style={{ animationDelay: '0.1s' }}>
      <h2><Package size={24} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 8 }}/> Fridge Inventory</h2>
      
      {/* Consumption Input */}
      <form onSubmit={handleConsume} style={{ display: 'flex', gap: '8px', marginBottom: '24px' }}>
        <input
          type="text"
          className="input-glass"
          placeholder="e.g. I drank 1 cup of milk"
          value={statement}
          onChange={(e) => setStatement(e.target.value)}
          disabled={loading}
        />
        <button type="submit" className="btn-primary" disabled={loading || !statement.trim()}>
          <Send size={18} />
        </button>
      </form>

      {/* Inventory List */}
      {items.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-secondary)' }}>
          <Utensils size={48} style={{ opacity: 0.2, marginBottom: '16px' }} />
          <p>Your fridge is completely empty!</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {items.map((item: any, idx: number) => (
            <div 
              key={idx} 
              style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                padding: '16px',
                background: 'rgba(255,255,255,0.05)',
                borderRadius: '8px',
                borderLeft: item.count <= 0 ? '4px solid var(--danger)' : '4px solid var(--accent-primary)'
              }}
            >
              <div>
                <h4 style={{ margin: 0, textTransform: 'capitalize' }}>{item.item_name}</h4>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  {item.count <= 0 ? (
                    <span className="text-danger" style={{ fontWeight: 'bold' }}>Out of stock</span>
                  ) : (
                    <>{item.count} {item.unit}</>
                  )}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ textAlign: 'right' }}>
                  {item.count > 0 && getExpirationBadge(item.expiration_date)}
                </div>
                <button 
                  onClick={() => handleRemove(item.item_name)} 
                  disabled={loading}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--danger, #ef4444)',
                    cursor: 'pointer',
                    padding: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: '4px',
                    transition: 'background 0.2s'
                  }}
                  title="Remove item"
                  onMouseOver={(e) => e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)'}
                  onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>

          ))}
        </div>
      )}
    </div>
  );
}
