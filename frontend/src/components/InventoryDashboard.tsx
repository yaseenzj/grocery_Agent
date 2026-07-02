import React, { useState } from 'react';
import { Send, Trash2, Plus } from 'lucide-react';
import { api } from '../api';
import { ReceiptUploader } from './ReceiptUploader';

const getIconForProduct = (name: string) => {
  const lower = name.toLowerCase();
  if (lower.includes('apple')) return '🍎';
  if (lower.includes('orange')) return '🍊';
  if (lower.includes('egg')) return '🥚';
  if (lower.includes('milk')) return '🥛';
  if (lower.includes('bread')) return '🍞';
  if (lower.includes('butter')) return '🧈';
  if (lower.includes('chicken')) return '🍗';
  if (lower.includes('tomato')) return '🍅';
  if (lower.includes('garlic')) return '🧄';
  if (lower.includes('flour')) return '🌾';
  if (lower.includes('potato')) return '🥔';
  if (lower.includes('onion')) return '🧅';
  if (lower.includes('cheese')) return '🧀';
  if (lower.includes('beef') || lower.includes('meat')) return '🥩';
  if (lower.includes('fish')) return '🐟';
  if (lower.includes('rice')) return '🍚';
  if (lower.includes('water')) return '💧';
  if (lower.includes('oil')) return '🍾';
  return '📦'; // Default
};

export function InventoryDashboard({ inventory, onInventoryUpdate }: { inventory: any, onInventoryUpdate: (data: any) => void }) {
  const [statement, setStatement] = useState('');
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [showAdd, setShowAdd] = useState(false);

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
    if (!window.confirm(`Are you sure you want to completely throw away "${itemName}"?`)) return;
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
  const filteredItems = items.filter((item: any) => 
    item.item_name.toLowerCase().includes(search.toLowerCase())
  );

  const getExpirationClass = (dateString: string) => {
    const exp = new Date(dateString);
    const now = new Date();
    const days = Math.ceil((exp.getTime() - now.getTime()) / (1000 * 3600 * 24));
    
    if (days < 0) return 'danger';
    if (days <= 3) return 'danger';
    if (days <= 7) return 'warning';
    return 'success';
  };

  return (
    <div className="animate-fade-in">
      
      {/* Quick Actions */}
      <div style={{ display: 'flex', gap: '16px', marginBottom: '32px' }}>
        <input
          type="text"
          className="input-field"
          placeholder="🔍 Search fridge..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ flex: 1 }}
        />
        <button 
          className="btn-primary" 
          style={{ width: 'auto' }}
          onClick={() => setShowAdd(!showAdd)}
        >
          {showAdd ? 'Close' : <><Plus size={20} /> Add Items</>}
        </button>
      </div>

      {showAdd && (
        <div style={{ marginBottom: '40px', padding: '24px', background: 'var(--card-bg)', borderRadius: '20px', border: '1px solid var(--card-border)', boxShadow: 'var(--card-shadow)' }}>
          <ReceiptUploader onUploadComplete={(data) => {
            onInventoryUpdate(data);
            setShowAdd(false);
          }} />
        </div>
      )}

      {/* Consumption Logging */}
      <div style={{ marginBottom: '40px', padding: '24px', background: '#f8fafc', borderRadius: '20px', border: '1px solid var(--card-border)' }}>
        <div className="panel-title" style={{ fontSize: '1.2rem', marginBottom: '12px' }}>Did you eat something?</div>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', fontSize: '0.9rem' }}>
          Tell the AI what you used (e.g. "I drank half the milk" or "used 2 eggs").
        </p>
        <form onSubmit={handleConsume} style={{ display: 'flex', gap: '12px' }}>
          <input
            type="text"
            className="input-field"
            placeholder="Log your consumption..."
            value={statement}
            onChange={(e) => setStatement(e.target.value)}
            disabled={loading}
            style={{ background: '#fff' }}
          />
          <button type="submit" className="btn-primary" style={{ width: 'auto' }} disabled={loading || !statement.trim()}>
            <Send size={20} />
          </button>
        </form>
      </div>

      <div className="panel-title">Your Current Ingredients</div>
      
      {items.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-tertiary)' }}>
          <div style={{ fontSize: '3rem', marginBottom: '16px' }}>🛒</div>
          <p>Your fridge is completely empty!</p>
        </div>
      ) : (
        <div className="product-grid">
          {filteredItems.map((item: any, idx: number) => (
            <div key={idx} className="product-card">
              <button 
                className="product-delete" 
                onClick={() => handleRemove(item.item_name)}
                disabled={loading}
                title="Throw away completely"
              >
                <Trash2 size={16} />
              </button>
              
              <div className={`product-badge ${getExpirationClass(item.expiration_date)}`}></div>
              
              <div className="product-icon">
                {getIconForProduct(item.item_name)}
              </div>
              <div className="product-name" title={item.item_name}>
                {item.item_name}
              </div>
              <div className="product-qty">
                {item.count <= 0 ? (
                  <span className="text-danger">Out of stock</span>
                ) : (
                  <>{item.count} {item.unit}</>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
