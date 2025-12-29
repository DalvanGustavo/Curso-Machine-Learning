#%%
import pandas as pd
df = pd.read_excel('Dados/dados_cerveja.xlsx')
df.head()

#%%
features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'
x = df[features]
y = df[target]
x = x.replace({
    'mud': 1, 'pint': 2,
    'sim': 1, 'não': 0,
    'clara': 0, 'escura': 1
})

#%%
from sklearn import tree
model = tree.DecisionTreeClassifier(random_state=42)
model.fit(x, y)

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True, fontsize=10)
