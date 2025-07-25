from neuprint import Client, fetch_neurons, fetch_adjacencies, fetch_roi_hierarchy, NeuronCriteria as NC
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv('token')

client = Client('https://neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=token)
roi_hierarchy = fetch_roi_hierarchy(client, mark_primary=True, format='text')


roi_hierarchy = fetch_roi_hierarchy(client, mark_primary=True, format='dict')
valid_rois = list(roi_hierarchy.keys())


visual_rois = [roi for roi in ["LO(R)", "LO(L)", "ME(R)", "ME(L)", "LOP(R)", "LOP(L)"] if roi in valid_rois]


neuron_criteria = NC(rois=visual_rois, status='Traced')
neuron_df, roi_counts_df = fetch_neurons(neuron_criteria)


max_neurons = 5000  
if len(neuron_df) > max_neurons:
    neuron_df = neuron_df.sample(n=max_neurons, random_state=42)    
neuron_ids = neuron_df['bodyId'].tolist()


batch_size = 100  
all_connections = []
# for i in range(0, len(neuron_ids), batch_size):
#     batch = neuron_ids[i:i + batch_size]
#     print(f"Fetching batch {i // batch_size + 1}/{len(neuron_ids) // batch_size + 1}...")
    
#     try:
#         _, conn_df = fetch_adjacencies(batch, None)  
#         all_connections.append(conn_df)
#     except Exception as e:
#         print(f"Error fetching batch {i // batch_size + 1}: {e}")


conn_df = pd.concat(all_connections, ignore_index=True)
conn_df = conn_df[conn_df['weight'] > 10]
conn_df.to_csv("filtered_cells.csv", index=False)


neuron_features = neuron_df[['bodyId', 'type', 'pre', 'post', 'size']]
neuron_features.to_csv("filtered_neurons.csv")
neuron_df.to_csv('filtered_neurons_all_features.csv')
