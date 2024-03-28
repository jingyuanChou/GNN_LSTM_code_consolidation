import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap

mte_matrices = np.load('MTE_matrices_67.npy')

temp_list = list()

for index in range(0, 67):
    temp_data = mte_matrices[index]
    temp_matrix = np.sum(temp_data, axis=0)
    temp_list.append(temp_matrix)

temp_list = np.array(temp_list)

state_positions = {'AL': (-86.8073, 32.8067), 'AK': (-125, 48), 'AZ': (-111.0937, 34.0489), 'AR': (-92.3731, 34.9697),
                   'CA': (-119.6816, 36.1162), 'CO': (-105.3111, 39.0598), 'CT': (-72.7554, 41.5978),
                   'DE': (-75.5071, 39.3185), 'FL': (-81.6868, 27.7663), 'GA': (-83.6431, 33.0406), 'HI': (-125, 32),
                   'ID': (-114.4788, 44.2405), 'IL': (-88.9861, 40.3495), 'IN': (-86.2583, 39.8494),
                   'IA': (-93.2105, 42.0115), 'KS': (-96.7265, 38.5266), 'KY': (-84.6701, 37.6681),
                   'LA': (-91.8678, 31.1695), 'ME': (-69.3819, 44.6939), 'MD': (-76.8021, 39.0639),
                   'MA': (-71.5301, 42.2302), 'MI': (-84.5361, 43.3266), 'MN': (-93.9002, 45.6945),
                   'MS': (-89.6787, 32.7416), 'MO': (-92.2884, 38.4561), 'MT': (-110.4544, 46.9219),
                   'NE': (-98.2681, 41.1254), 'NV': (-117.0554, 38.3135), 'NH': (-71.5639, 43.4525),
                   'NJ': (-74.5210, 40.2989), 'NM': (-106.2485, 34.8405), 'NY': (-74.9481, 42.1657),
                   'NC': (-79.8064, 35.6301), 'ND': (-99.7840, 47.5289), 'OH': (-82.7649, 40.3888),
                   'OK': (-96.9289, 35.5653), 'OR': (-122.0709, 44.5720), 'PA': (-77.2098, 40.5908),
                   'RI': (-71.5118, 41.6809), 'SC': (-80.9450, 33.8569), 'SD': (-99.4388, 44.2998),
                   'TN': (-86.6923, 35.7478), 'TX': (-97.5635, 31.0545), 'UT': (-111.8624, 40.1500),
                   'VT': (-72.7107, 44.0459), 'VA': (-78.1699, 37.7693), 'WA': (-121.4905, 47.4009),
                   'WV': (-80.9545, 38.4912), 'WI': (-89.6165, 44.2685), 'WY': (-107.3025, 42.7559),
                   'DC': (-77.0369, 38.9072)}


# Reposition for AK and HI

def matrix_to_graph(matrix, threshold):
    G = nx.DiGraph()
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val != 0 and i != j and val > threshold:
                G.add_edge(i, j, weight=round(val, 2))
    return G


if __name__ == '__main__':
    # Load series of index-value pairs
    loaded_series = pd.read_json('location.json', orient='index', typ='series')
    mapping = loaded_series.to_dict()

    # Take FL as an example, let's examine if it's a really leading indicator for those states FL is pointing to
    # Load case data, FL is index 9
    case_states = pd.read_csv('weekly_filt_case_data_July2020_Mar2022.csv')

    # Manually adjusted positions for Alaska and Hawaii

    # Assume temp_list is predefined, replace it with your actual list of matrices

    with PdfPages('network_graphs_on_us_map.pdf') as pdf:
        for idx, matrix in enumerate(temp_list):
            # Analyze the distribution and set the threshold
            cur_time = idx + 24
            np.fill_diagonal(matrix, 0)
            edge_weights = matrix[matrix != 0]
            threshold = np.percentile(edge_weights, 25)  # Adjust the percentile as needed

            G = matrix_to_graph(matrix, threshold)
            filtered_mapping = {k: v for k, v in mapping.items() if v in state_positions}
            G = nx.relabel_nodes(G, filtered_mapping)

            fig = plt.figure(figsize=(20, 30))  # Increased figure size

            # Upper plot (for the additional graph)
            plt.subplot(2, 1, 1)
            # Here, include your code to plot the additional graph.
            # For example, a simple line plot:
            Florida = matrix[9]
            affected_by_FL = [index for index, element in enumerate(Florida) if element != 0]
            affected_by_FL_states = [mapping[key] for key in affected_by_FL]
            case_states = np.array(case_states)
            current_FL = case_states[9, np.array(range(3, 3 + idx + 24))]
            if len(affected_by_FL) != 0:
                affected_by_FL_ls = case_states[np.ix_(affected_by_FL, np.array(range(3, 3 + idx + 24)))]
                final_mapping = dict()
                final_mapping['FL'] = current_FL
                for idx, state_name in enumerate(affected_by_FL_states):
                    final_mapping[state_name] = affected_by_FL_ls[idx]
                for key, values in final_mapping.items():
                    plt.plot(values, label=key)
                plt.title('Case Counts Graph')
                plt.xlabel('Timestamps')  # Customize X axis label
                plt.ylabel('Number of Cases')
                plt.legend()  # Show legend
            else:
                affected_by_FL_ls = []
                plt.plot(current_FL)
                plt.xlabel('Timestamps')  # Customize X axis label
                plt.ylabel('Number of Cases')
                plt.title('Florida Counts Graph')

            # Lower plot (for the current graph, with more space)
            plt.subplot(2, 1, 2)
            m = Basemap(llcrnrlon=-125, llcrnrlat=24, urcrnrlon=-66, urcrnrlat=50,
                        projection='lcc', lat_1=32, lat_2=45, lon_0=-95)
            m.drawcoastlines()
            m.drawcountries()
            m.drawstates()

            pos = {state: m(lon, lat) for state, (lon, lat) in state_positions.items()}
            nx.draw_networkx_nodes(G, pos, node_color='red', node_size=100, alpha=0.6)
            nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=20, edge_color='blue')
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

            plt.title(f"Graph on US at {idx + 24} Timestamps", size=15)
            plt.axis('off')

            pdf.savefig(fig)
            plt.close(fig)
