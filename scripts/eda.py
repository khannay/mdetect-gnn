
import pandas as pd
from pathlib import Path
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx 

from torch_geometric.data import Data as PyGData

def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # remove leading and trailing whitespace from column names
    df.columns = df.columns.str.strip()
    # replace spaces with underscores
    df.columns = df.columns.str.replace(' ', '_')
    # replace parentheses with underscores
    df.columns = df.columns.str.replace('(', '_')
    df.columns = df.columns.str.replace(')', '_')
    # replace slashes with underscores
    df.columns = df.columns.str.replace('/', '_')
    # replace dashes with underscores
    df.columns = df.columns.str.replace('-', '_')
    # replace dots with underscores
    df.columns = df.columns.str.replace('.', '_')
    # replace colons with underscores
    df.columns = df.columns.str.replace(':', '_')
    
    return df

def get_all_datafiles() -> List[Path]:
    # recursively find all .csv files in the data directory
    data_dir = Path('./data/csv')
    all_files = list(data_dir.glob('**/*.csv'))
    print("Found {} files".format(len(all_files)))
    
    # get the name of the malware family from the directory name
    all_classifications = [f.parent.name for f in all_files]
    
    # stack the data and labels into a single dataframe
    all_data = pd.concat([pd.read_csv(f) for f in all_files], axis=0)
    all_data = fix_column_names(all_data)
    all_data = all_data.convert_dtypes()
    for column in all_data.columns:
        if all_data[column].dtype == 'object':
            # convert to a float or Nan if not possible
            all_data[column] = pd.to_numeric(all_data[column], errors='coerce')
    print("Combined data shape: {}".format(all_data.shape))
    print(all_data.head())
    # print each column and its data type
    zip_obj = zip(all_data.columns, all_data.dtypes)
    for column, dtype in zip_obj:
        print("Column: {}, dtype: {}".format(column, dtype))
        
    all_data.to_parquet('./data/combined.parquet', index=False)
    return all_classifications
    

df = pd.read_csv("data/csv/SMSmalware/Beanbot/08_18_2017-ps-beanbot-sophos-05af60ce8a8c3cd31682982122423ef3.pcap_ISCX.csv")
print(df.shape)
df = fix_column_names(df)
df = df.convert_dtypes()

# aggregate the data by source IP and destination IP
df['flow'] = df['Source_IP'] + '-' + df['Destination_IP']

# show all string or object columns
for column in df.columns:
    if df[column].dtype == 'object' or df[column].dtype == 'string':
        print(column)

# aggregate the data using the mean of each column for the flow column 
# drop any columns that are not numeric, except for the flow column
df = df.drop(columns=['Source_IP', 'Destination_IP', 'Timestamp', 'Source_Port', 'Destination_Port', 'Protocol', 'Flow_ID', 'Label'])
# force all columns to be numeric except flow 
for column in df.columns:
    if column != 'flow':
        df[column] = pd.to_numeric(df[column], errors='coerce')

df = df.groupby('flow').mean()
df = df.reset_index()

df['Source_IP'] = [x.split('-')[0] for x in df['flow']]
df['Destination_IP'] = [x.split('-')[1] for x in df['flow']]
df = df.drop(columns=['flow'])

print(df.shape)

# create a networkx graph from the dataframe
G = nx.from_pandas_edgelist(df, source='Source_IP', target='Destination_IP', edge_attr=True, create_using=nx.DiGraph)

# add edge weights to the graph
print(G.edges.data('Flow_Duration', default=0))

# print the number of nodes and edges
print("Number of nodes: {}".format(G.number_of_nodes()))
print("Number of edges: {}".format(G.number_of_edges()))

# convert the graph to a pytorch geometric graph
geo_g = from_networkx(G)
geo_g.edge_attr = []
for u, v, data in G.edges(data=True):
    edge_attr = []
    for key, value in data.items():
        edge_attr.append(value)
    geo_g.edge_attr.append(edge_attr)
print("Pytorch Geometric graph")
# print(geo_g.num_nodes)
# print(geo_g.num_edges)
# print(geo_g.edge_attr)

# plot the graph
nx.draw(G, with_labels=False, node_size=20, node_color='r', pos=nx.spring_layout(G))
plt.show()

    
                 
                 