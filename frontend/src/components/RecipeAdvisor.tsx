import { useState, useEffect } from 'react';
import { ChefHat, CheckCircle, X, ShoppingCart } from 'lucide-react';
import { api } from '../api';

export function RecipeAdvisor() {
  const [recipes, setRecipes] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRecipe, setSelectedRecipe] = useState<any | null>(null);

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

  useEffect(() => {
    fetchRecipes();
  }, []);

  const getGradient = (name: string) => {
    let hash = 0;
    for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
    const h1 = Math.abs(hash) % 360;
    const h2 = (h1 + 40) % 360;
    return `linear-gradient(135deg, hsl(${h1}, 70%, 90%) 0%, hsl(${h2}, 70%, 80%) 100%)`;
  };

  return (
    <div className="animate-fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, margin: 0 }}>Cook with what you have</h2>
        <button 
          onClick={fetchRecipes} 
          disabled={loading}
          style={{ 
            background: 'none', border: 'none', color: 'var(--accent-hover)', 
            fontWeight: 600, cursor: 'pointer', fontSize: '1rem' 
          }}
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {recipes.length === 0 && !loading ? (
        <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-tertiary)' }}>
          <ChefHat size={48} style={{ opacity: 0.5, margin: '0 auto 16px auto' }} />
          <p>No recipes available. Add items to your fridge!</p>
        </div>
      ) : (
        <div className="recipe-list">
          {recipes.map((recipe, idx) => {
            const matchScore = recipe.inventory_match_score || 0;
            const isReady = recipe.restock_recommendations?.length === 0 || matchScore === 1;
            
            return (
              <div key={idx} className="recipe-card" onClick={() => setSelectedRecipe(recipe)}>
                <div className="recipe-image" style={{ background: getGradient(recipe.recipe_name) }}>
                  🍲
                </div>
                <div className="recipe-content">
                  <div className="recipe-tags" style={{ marginBottom: '6px' }}>
                    <span className="recipe-tag accent">
                      {recipe.difficulty_level || 'Medium'}
                    </span>
                  </div>
                  <div className="recipe-title" title={recipe.recipe_name}>{recipe.recipe_name}</div>
                  
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-tertiary)', marginBottom: '8px' }}>
                    ~25 min • ~350 kcal
                  </div>

                  <div className="recipe-ready">
                    <CheckCircle size={16} />
                    {isReady ? 'Ready to cook' : `${Math.round(matchScore * 100)}% ingredients ready`}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Centered Desktop Modal */}
      {selectedRecipe && (
        <div className="modal-overlay" onClick={() => setSelectedRecipe(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close-btn" onClick={() => setSelectedRecipe(null)}>
              <X size={20} />
            </button>
            
            <div className="modal-header">
              <h2 style={{ fontSize: '2rem', marginBottom: '8px', paddingRight: '40px' }}>{selectedRecipe.recipe_name}</h2>
              <div className="recipe-tags">
                <span className="recipe-tag accent">{selectedRecipe.difficulty_level || 'Medium'}</span>
                <span className="recipe-ready" style={{ marginLeft: '12px' }}>
                  <CheckCircle size={16} /> 
                  {selectedRecipe.restock_recommendations?.length === 0 ? 'Ready to cook' : `${Math.round((selectedRecipe.inventory_match_score || 0) * 100)}% Match`}
                </span>
              </div>
            </div>
            
            <div className="modal-body">
              <div className="detail-section" style={{ marginTop: 0 }}>
                <h3>Ingredients from Fridge</h3>
                <ul className="detail-list">
                  {selectedRecipe.ingredients.map((ing: any, i: number) => (
                    <li key={i}>
                      <span style={{ flex: 1, textTransform: 'capitalize', fontWeight: 500 }}>{ing.item_name}</span>
                      <strong style={{ color: 'var(--text-secondary)' }}>{ing.quantity} {ing.unit}</strong>
                    </li>
                  ))}
                </ul>
              </div>

              {selectedRecipe.restock_recommendations && selectedRecipe.restock_recommendations.length > 0 && (
                <div className="detail-section">
                  <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--danger)' }}>
                    <ShoppingCart size={18} /> You need to buy
                  </h3>
                  <ul className="detail-list" style={{ borderLeft: '4px solid var(--danger)' }}>
                    {selectedRecipe.restock_recommendations.map((ing: any, i: number) => (
                      <li key={i}>
                        <span style={{ flex: 1, textTransform: 'capitalize', fontWeight: 500 }}>{ing.item_name}</span>
                        <strong style={{ color: 'var(--text-secondary)' }}>{ing.quantity_needed} {ing.unit}</strong>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="detail-section">
                <h3>How to cook</h3>
                <ul className="detail-list" style={{ border: 'none', background: 'transparent', padding: 0 }}>
                  {selectedRecipe.steps.map((step: string, i: number) => (
                    <li key={i} style={{ borderBottom: 'none', padding: '16px 0' }}>
                      <div className="step-number">{i + 1}</div>
                      <div style={{ flex: 1, lineHeight: '1.6', color: 'var(--text-primary)' }}>{step}</div>
                    </li>
                  ))}
                </ul>
              </div>
              
              {selectedRecipe.source && (
                <div style={{ marginTop: '32px', textAlign: 'left' }}>
                  <a href={selectedRecipe.source} target="_blank" rel="noreferrer" style={{ color: 'var(--accent-hover)', textDecoration: 'none', fontWeight: 600 }}>
                    View Original Source ↗
                  </a>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
