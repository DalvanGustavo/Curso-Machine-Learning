#%%
import pandas as pd
df = pd.read_parquet('Dados/dados_clones.parquet')
df['General Jedi encarregado'].unique()
df.columns

# %%
features = ['Massa(em kilos)','Estatura(cm)', 'Distância Ombro a ombro', 'Tamanho do crânio', 'Tamanho dos pés', 'Tempo de existência(em meses)']
target = 'Status '
x = df[features]
y = df[target]
x = x.replace({
    'Tipo 1': 1, 'Tipo 2': 2, 'Tipo 3': 3, 'Tipo 4': 4, 'Tipo 5': 5,
    #'Yoda': 1, 'Shaak Ti': 2, 'Obi-Wan Kenobi': 3, 'Aayla Secura': 4, 'Mace Windu': 5
})
# %%
from sklearn import tree
model = tree.DecisionTreeClassifier(random_state=42)
model.fit(x, y)
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True, fontsize=10, max_depth=3)
