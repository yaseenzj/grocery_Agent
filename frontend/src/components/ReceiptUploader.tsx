import React, { useState } from 'react';
import { Upload, Loader2, CheckCircle } from 'lucide-react';
import { api } from '../api';

export function ReceiptUploader({ onUploadComplete }: { onUploadComplete: (data: any) => void }) {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    
    setLoading(true);
    setSuccess(false);
    try {
      const data = await api.uploadReceipt(text);
      setSuccess(true);
      setText('');
      onUploadComplete(data);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      alert("Failed to parse receipt. Please check server logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel animate-fade-in">
      <h2><Upload size={24} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 8 }}/> Add Groceries</h2>
      <p className="text-secondary" style={{ marginBottom: '16px' }}>
        Paste a grocery receipt, or just type items in any format. Our AI handles the rest.
      </p>
      
      <form onSubmit={handleSubmit}>
        <textarea
          className="input-glass"
          rows={6}
          placeholder={`Any format works, e.g.:\n• 1 litre milk, 10 tomatoes, 10 potatoes\n• Chicken Breast 2 lbs\n  Whole Milk 1 gallon\n• Bought 2 cartons of eggs and some spinach`}
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
          style={{ marginBottom: '16px', resize: 'vertical' }}
        />
        
        <button 
          type="submit" 
          className="btn-primary" 
          disabled={loading || !text.trim()}
          style={{ width: '100%' }}
        >
          {loading ? (
            <><Loader2 className="animate-spin" /> Processing AI...</>
          ) : success ? (
            <><CheckCircle /> Successfully Added</>
          ) : (
            'Process Receipt'
          )}
        </button>
      </form>
    </div>
  );
}
