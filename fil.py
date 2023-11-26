import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import tensorflow as tf
from tensorflow import keras

data = pd.read_csv('CC GENERAL.csv').dropna()


clustering_data = data[['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']]


scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)


kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(clustering_data_scaled)
data['CREDIT_CARD_SEGMENTS'] = clusters

fig = px.scatter_3d(data, x='BALANCE', y='PURCHASES', z='CREDIT_LIMIT', color='CREDIT_CARD_SEGMENTS',
                    symbol='CREDIT_CARD_SEGMENTS', size_max=10, opacity=0.7,
                    title='Credit Card Segments Visualization')


fig.update_layout(scene=dict(xaxis=dict(title='BALANCE'), yaxis=dict(title='PURCHASES'), zaxis=dict(title='CREDIT_LIMIT')),
                  showlegend=False)

fig.show()


plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='BALANCE', y='PURCHASES', hue='CREDIT_CARD_SEGMENTS', palette='viridis', s=100)
plt.title('Credit Card Segments Visualization (Seaborn)')
plt.show()

G = nx.Graph()
edges = [(i, j) for i, j in zip(clusters, range(len(clusters)))]
G.add_edges_from(edges)


plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=False, node_size=20, font_size=8, node_color=clusters, cmap='viridis')
plt.title('Credit Card Segments Visualization (NetworkX)')
plt.show()

# Ustering
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(3,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5, activation='softmax')  # 5 clusters
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(clustering_data_scaled, clusters, epochs=50)


embeddings = model.layers[1].get_weights()[0]


plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters, cmap='viridis', s=100)
plt.title('Credit Card Segments Visualization (Neural Network Embeddings)')
plt.show()
