import React, { useState } from 'react';
import { ChefHat, Loader2, PlayCircle, ExternalLink } from 'lucide-react';
import { api } from '../api';

export function RecipeAdvisor() {
  const [recipes, setRecipes] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchRecipes = async () => {
    setLoading(true);
    try {
      const data = await api.getRecipes();
      setRecipes(data.recipes || []);
    } catch (err) {
      alert("Failed to fetch recipes.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel animate-fade-in" style={{ animationDelay: '0.2s' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h2><ChefHat size={24} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 8 }}/> Recipe Advisor</h2>
        <button className="btn-primary" onClick={fetchRecipes} disabled={loading}>
          {loading ? <Loader2 className="animate-spin" size={18} /> : <PlayCircle size={18} />}
          {loading ? 'Generating...' : 'Generate Recipes'}
        </button>
      </div>

      {recipes.length === 0 && !loading && (
        <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-secondary)' }}>
          <p>Click generate to see what you can cook with your ingredients!</p>
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {recipes.map((recipe, idx) => (
          <div key={idx} style={{ background: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
            <h3 style={{ color: 'var(--accent-primary)', marginBottom: '16px' }}>{recipe.recipe_name}</h3>
            
            <div style={{ marginBottom: '16px' }}>
              <h4 style={{ fontSize: '0.9rem', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '8px' }}>Ingredients Used</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {recipe.ingredients.map((ing: any, i: number) => (
                  <span key={i} className="badge" style={{ background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)' }}>
                    {ing.quantity} {ing.unit} {ing.item_name}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h4 style={{ fontSize: '0.9rem', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '8px' }}>Instructions</h4>
              <ol style={{ paddingLeft: '20px', color: 'var(--text-primary)', margin: 0 }}>
                {recipe.steps.map((step: string, i: number) => (
                  <li key={i} style={{ marginBottom: '8px' }}>{step}</li>
                ))}
              </ol>
            </div>
            
            {recipe.source && (
              <a href={recipe.source} target="_blank" rel="noreferrer" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: 'var(--accent-primary)', textDecoration: 'none', marginTop: '16px', fontSize: '0.9rem', fontWeight: 600 }}>
                <ExternalLink size={16} /> View Source
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
