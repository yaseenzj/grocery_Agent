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
    <div className="animate-fade-in">
      <div className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Upload size={20} /> Paste a Receipt or List
      </div>
      <p className="text-secondary" style={{ marginBottom: '24px', fontSize: '0.9rem' }}>
        Type items in any format. Our AI handles the rest.
      </p>
      
      <form onSubmit={handleSubmit}>
        <textarea
          className="input-field"
          rows={6}
          placeholder={`Any format works, e.g.:\n• 1 litre milk, 10 tomatoes\n• Chicken Breast 2 lbs\n• Bought 2 cartons of eggs`}
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
          style={{ marginBottom: '24px', resize: 'vertical' }}
        />
        
        <button 
          type="submit" 
          className="btn-primary" 
          disabled={loading || !text.trim()}
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
