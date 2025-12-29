#%%
import pandas as pd
df = pd.read_excel('Dados/dados_frutas.xlsx')
df

#%%
from sklearn import tree
arvore = tree.DecisionTreeClassifier(random_state=42)

#%%
y = df['Fruta']
caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
x = df[caracteristicas]

#%%
arvore.fit(x, y)

#%%
arvore.predict([[1,1,1,1]])

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
tree.plot_tree(arvore, feature_names=caracteristicas, class_names=arvore.classes_, filled=True,fontsize=10)

#%%
proba = arvore.predict_proba([[1,1,1,0]])[0]
pd.Series(proba, index=arvore.classes_)