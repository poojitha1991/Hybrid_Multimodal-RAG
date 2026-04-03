import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Title
ax.text(5, 13.5, 'Multi-Modal RAG System Architecture', 
        fontsize=20, weight='bold', ha='center')

# Color scheme
color_input = '#E3F2FD'
color_process = '#FFF3E0'
color_storage = '#F3E5F5'
color_retrieval = '#E8F5E9'
color_output = '#FCE4EC'

# 1. INPUT LAYER
ax.add_patch(FancyBboxPatch((0.5, 11.5), 2, 1.2, boxstyle="round,pad=0.1", 
                            facecolor=color_input, edgecolor='#1976D2', linewidth=2))
ax.text(1.5, 12.3, 'PDF Upload', fontsize=11, weight='bold', ha='center')
ax.text(1.5, 11.9, '(Streamlit UI)', fontsize=9, ha='center', style='italic')

# 2. INGESTION PIPELINE
ax.add_patch(FancyBboxPatch((3.5, 10.5), 3, 2.5, boxstyle="round,pad=0.1", 
                            facecolor=color_process, edgecolor='#F57C00', linewidth=2))
ax.text(5, 12.7, 'Ingestion Pipeline', fontsize=11, weight='bold', ha='center')

# Sub-components
ax.text(5, 12.2, '1. PyMuPDF: Extract text', fontsize=9, ha='center')
ax.text(5, 11.8, '2. PDFPlumber: Extract tables', fontsize=9, ha='center')
ax.text(5, 11.4, '3. PyMuPDF: Extract images', fontsize=9, ha='center')
ax.text(5, 11.0, '4. Chunking (900 chars, 200 overlap)', fontsize=9, ha='center')

# 3. EMBEDDING & STORAGE
ax.add_patch(FancyBboxPatch((7.5, 10.5), 2, 2.5, boxstyle="round,pad=0.1", 
                            facecolor=color_storage, edgecolor='#7B1FA2', linewidth=2))
ax.text(8.5, 12.7, 'Vector Store', fontsize=11, weight='bold', ha='center')
ax.text(8.5, 12.2, 'Sentence-BERT', fontsize=9, ha='center')
ax.text(8.5, 11.8, 'Embeddings', fontsize=9, ha='center', style='italic')
ax.text(8.5, 11.3, 'ChromaDB', fontsize=9, ha='center', weight='bold')
ax.text(8.5, 10.9, '(Persistent)', fontsize=8, ha='center')

# 4. USER QUERY
ax.add_patch(FancyBboxPatch((0.5, 8.5), 2, 1, boxstyle="round,pad=0.1", 
                            facecolor=color_input, edgecolor='#1976D2', linewidth=2))
ax.text(1.5, 9.2, 'User Query', fontsize=11, weight='bold', ha='center')
ax.text(1.5, 8.8, '"What is RAG?"', fontsize=9, ha='center', style='italic')

# 5. HYBRID RETRIEVAL ALGORITHM
ax.add_patch(FancyBboxPatch((3, 6), 6.5, 3.2, boxstyle="round,pad=0.1", 
                            facecolor=color_retrieval, edgecolor='#388E3C', linewidth=3))
ax.text(6.25, 9, 'Hybrid Retrieval Algorithm', fontsize=12, weight='bold', ha='center')

# Step 1: Query Expansion
ax.add_patch(FancyBboxPatch((3.3, 7.8), 2.5, 0.9, boxstyle="round,pad=0.05", 
                            facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=1.5))
ax.text(4.55, 8.5, '1. Query Expansion', fontsize=9, weight='bold', ha='center')
ax.text(4.55, 8.15, '3 variations', fontsize=8, ha='center', style='italic')

# Step 2: Multi-Query Search
ax.add_patch(FancyBboxPatch((6.2, 7.8), 2.8, 0.9, boxstyle="round,pad=0.05", 
                            facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=1.5))
ax.text(7.6, 8.5, '2. Vector Search', fontsize=9, weight='bold', ha='center')
ax.text(7.6, 8.15, 'top_k × 2 per query', fontsize=8, ha='center', style='italic')

# Step 3: Hybrid Scoring
ax.add_patch(FancyBboxPatch((3.3, 6.5), 2.5, 1, boxstyle="round,pad=0.05", 
                            facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=1.5))
ax.text(4.55, 7.2, '3. Hybrid Scoring', fontsize=9, weight='bold', ha='center')
ax.text(4.55, 6.9, 'Semantic: 70%', fontsize=8, ha='center')
ax.text(4.55, 6.65, 'Keyword: 30%', fontsize=8, ha='center')

# Step 4: Re-ranking
ax.add_patch(FancyBboxPatch((6.2, 6.5), 2.8, 1, boxstyle="round,pad=0.05", 
                            facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=1.5))
ax.text(7.6, 7.2, '4. Re-ranking', fontsize=9, weight='bold', ha='center')
ax.text(7.6, 6.9, 'Dedup + Sort', fontsize=8, ha='center')
ax.text(7.6, 6.65, 'Return top_k', fontsize=8, ha='center')

# 6. LLM GENERATION
ax.add_patch(FancyBboxPatch((3, 3.8), 6.5, 1.8, boxstyle="round,pad=0.1", 
                            facecolor=color_process, edgecolor='#F57C00', linewidth=2))
ax.text(6.25, 5.3, 'LLM Generation', fontsize=11, weight='bold', ha='center')
ax.text(6.25, 4.9, 'Context + Query → LLM', fontsize=9, ha='center')
ax.text(6.25, 4.5, '(Groq/Ollama/OpenAI)', fontsize=9, ha='center', style='italic')
ax.text(6.25, 4.1, 'Grounded Answer', fontsize=9, ha='center', weight='bold')

# 7. OUTPUT
ax.add_patch(FancyBboxPatch((3, 1.5), 3, 2, boxstyle="round,pad=0.1", 
                            facecolor=color_output, edgecolor='#C2185B', linewidth=2))
ax.text(4.5, 3.2, 'Answer + References', fontsize=11, weight='bold', ha='center')
ax.text(4.5, 2.8, '• Text citations', fontsize=9, ha='center')
ax.text(4.5, 2.5, '• Tables', fontsize=9, ha='center')
ax.text(4.5, 2.2, '• Images', fontsize=9, ha='center')
ax.text(4.5, 1.9, '• Page numbers', fontsize=9, ha='center')

# 8. AUDIO SUMMARY (Optional)
ax.add_patch(FancyBboxPatch((6.5, 1.5), 3, 2, boxstyle="round,pad=0.1", 
                            facecolor=color_output, edgecolor='#C2185B', linewidth=2))
ax.text(8, 3.2, 'Audio Summary', fontsize=11, weight='bold', ha='center')
ax.text(8, 2.8, '1. Extract content', fontsize=9, ha='center')
ax.text(8, 2.5, '2. LLM summarize', fontsize=9, ha='center')
ax.text(8, 2.2, '3. Text-to-Speech', fontsize=9, ha='center')
ax.text(8, 1.9, '4. MP3 output', fontsize=9, ha='center')

# ARROWS - Flow
# Upload to Ingestion
arrow1 = FancyArrowPatch((2.5, 12.1), (3.5, 12.1), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='#1976D2')
ax.add_patch(arrow1)

# Ingestion to Storage
arrow2 = FancyArrowPatch((6.5, 11.7), (7.5, 11.7), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='#F57C00')
ax.add_patch(arrow2)

# Query to Retrieval
arrow3 = FancyArrowPatch((2.5, 9), (3, 8.5), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='#1976D2')
ax.add_patch(arrow3)

# Storage to Retrieval (bidirectional)
arrow4 = FancyArrowPatch((8.5, 10.5), (7.6, 9.2), 
                        arrowstyle='<->', mutation_scale=20, linewidth=2, color='#7B1FA2', linestyle='dashed')
ax.add_patch(arrow4)

# Retrieval to LLM
arrow5 = FancyArrowPatch((6.25, 6), (6.25, 5.6), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='#388E3C')
ax.add_patch(arrow5)

# LLM to Output
arrow6 = FancyArrowPatch((4.5, 3.8), (4.5, 3.5), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='#F57C00')
ax.add_patch(arrow6)

# LLM to Audio
arrow7 = FancyArrowPatch((7.5, 4.5), (8, 3.5), 
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='#F57C00', linestyle='dashed')
ax.add_patch(arrow7)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='#1976D2', label='Input Layer'),
    mpatches.Patch(facecolor=color_process, edgecolor='#F57C00', label='Processing'),
    mpatches.Patch(facecolor=color_storage, edgecolor='#7B1FA2', label='Storage'),
    mpatches.Patch(facecolor=color_retrieval, edgecolor='#388E3C', label='Retrieval'),
    mpatches.Patch(facecolor=color_output, edgecolor='#C2185B', label='Output'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

# Key Features Box
ax.add_patch(FancyBboxPatch((0.3, 0.2), 2.2, 1, boxstyle="round,pad=0.05", 
                            facecolor='#FFFDE7', edgecolor='#F57F17', linewidth=1.5))
ax.text(1.4, 1, 'Key Features', fontsize=9, weight='bold', ha='center')
ax.text(1.4, 0.75, '✓ Multi-modal (text/table/image)', fontsize=7, ha='center')
ax.text(1.4, 0.55, '✓ Hybrid retrieval', fontsize=7, ha='center')
ax.text(1.4, 0.35, '✓ Audio summaries', fontsize=7, ha='center')

plt.tight_layout()
plt.savefig('rag_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved as 'rag_architecture.png'")
plt.show()
