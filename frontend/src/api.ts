const API_BASE = "http://localhost:8000";

export const api = {
  uploadReceipt: async (receiptText: string) => {
    const res = await fetch(`${API_BASE}/receipt/upload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ receipt_text: receiptText })
    });
    if (!res.ok) throw new Error("Failed to upload receipt");
    return res.json();
  },
  
  consumeItem: async (statement: string) => {
    const res = await fetch(`${API_BASE}/inventory/consume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ statement })
    });
    if (!res.ok) throw new Error("Failed to consume item");
    return res.json();
  },
  
  removeItem: async (itemName: string) => {
    const res = await fetch(`${API_BASE}/inventory/item/${encodeURIComponent(itemName)}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Failed to remove item");
    return res.json();
  },
  
  getRecipes: async () => {
    const res = await fetch(`${API_BASE}/recipes/recommend`);
    if (!res.ok) throw new Error("Failed to get recipes");
    return res.json();
  }
};

