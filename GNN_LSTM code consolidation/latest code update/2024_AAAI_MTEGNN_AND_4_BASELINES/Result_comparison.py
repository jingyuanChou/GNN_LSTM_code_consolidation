import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch


def draw():
    with open('index_2_state',"rb") as idx_2_state:
        index_2_state = pickle.load(idx_2_state)
    with open("Multistep results/geo_static_performance_result_subgraph_size_20_and_alpha_1_expnum_10", "rb") as fp:   # Unpickling
        geo_result = pickle.load(fp)

    geo_result_mae_mean = np.mean(geo_result['mae'], axis=0)
    geo_result_mae_std = np.std(geo_result['mae'], axis=0)
    geo_result_mape_mean = np.mean(geo_result['mape'], axis=0)
    geo_result_mape_std = np.std(geo_result['mape'], axis=0)
    print("geo based MAE loss is " + str(np.mean(geo_result_mae_mean)) + " , and MAPE is " + str(np.mean(geo_result_mape_mean)))

    with open("Multistep results/MTGNN_adaptive_performance_result_subgraph_size_20_and_1_expnum_10", "rb") as PDD:   # Unpickling
        Pure_data_driven = pickle.load(PDD)
    Pure_data_driven_mae_mean = np.mean(Pure_data_driven['mae'], axis=0)
    Pure_data_driven_mae_std = np.std(Pure_data_driven['mae'], axis=0)
    Pure_data_driven_mape_mean = np.mean(Pure_data_driven['mape'], axis=0)
    Pure_data_driven_mape_std = np.std(Pure_data_driven['mape'], axis=0)
    print('pure data based MAE loss is ' + str(np.mean(Pure_data_driven_mae_mean))+ " , and MAPE is " + str(np.mean(Pure_data_driven_mape_mean)))

    with open("Multistep results/MTE_adaptive_performance_result_subgraph_size_20_and_1_expnum_10", "rb") as PDD:  # Unpickling
        MTE_data_driven = pickle.load(PDD)
    MTE_data_driven_mae_mean = np.mean(MTE_data_driven['mae'], axis=0)
    MTE_data_driven_mae_std = np.std(MTE_data_driven['mae'], axis=0)
    MTE_data_driven_mape_mean = np.mean(MTE_data_driven['mape'], axis=0)
    MTE_data_driven_mape_std = np.std(MTE_data_driven['mape'], axis=0)
    print('MTE based MAE loss is '+str(np.mean(MTE_data_driven_mae_mean))+ " , and MAPE is " + str(np.mean(MTE_data_driven_mape_mean)))

    with open("Multistep results/MTE_static_performance_result_subgraph_size_20_and_1_expnum_10", "rb") as PDD:  # Unpickling
        MTE_static_data_driven = pickle.load(PDD)
    MTE_static_data_driven_mae_mean = np.mean(MTE_static_data_driven['mae'], axis=0)
    MTE_static_data_driven_mae_std = np.std(MTE_static_data_driven['mae'], axis=0)
    MTE_static_data_driven_mape_mean = np.mean(MTE_static_data_driven['mape'], axis=0)
    MTE_static_data_driven_mape_std = np.std(MTE_static_data_driven['mape'], axis=0)
    print('MTE static MAE loss is '+str(np.mean(MTE_static_data_driven_mae_mean))+ " , and MAPE is " + str(np.mean(MTE_static_data_driven_mape_mean)))

    x_range = range(49)
    x_element = list()
    for i in x_range:
        x_element.append(index_2_state[i])
    plt.plot(x_element, MTE_static_data_driven_mae_mean, color = 'red', marker='o', markersize = 3.0, label = 'MTE_STATIC')
    plt.plot(x_element, geo_result_mae_mean, color = 'black', marker='>', markersize = 3.0, label = 'GEO_STATIC')
    plt.plot(x_element, MTE_data_driven_mae_mean, color = 'red', marker='<', markersize = 3.0, label = 'MTE_ADAPTIVE')
    plt.plot(x_element, Pure_data_driven_mae_mean,color = 'orange', marker='<', markersize = 3.0, label = 'MTGNN' )
    plt.xlabel('States')
    plt.ylabel('Mean Absolute Error')
    plt.title('Performance comparison across region')
    plt.legend()
    plt.show()
    return

def drawAR():
    with open('index_2_state',"rb") as idx_2_state:
        index_2_state = pickle.load(idx_2_state)
    with open("AR results/geo_static_performance_result_subgraph_size_20_and_alpha_1_expnum_10_AR", "rb") as fp:   # Unpickling
        geo_result = pickle.load(fp)
    geo_result_mae_mean = np.mean(geo_result['mae'], axis=0)
    geo_result_mae_std = np.std(geo_result['mae'], axis=0)
    geo_result_mape_mean = np.mean(geo_result['mape'], axis=0)
    geo_result_mape_std = np.std(geo_result['mape'], axis=0)
    print("geo based MAE loss is " + str(np.mean(geo_result_mae_mean)) + " , and MAPE is " + str(np.mean(geo_result_mape_mean)))

    with open("AR results/MTE_adaptive_performance_result_subgraph_size_20_and_1_expnum_10_AR", "rb") as f:   # Unpickling
        TE_enhanced = pickle.load(f)
    TE_enhanced_mae_mean = np.mean(TE_enhanced['mae'], axis=0)
    TE_enhanced_mae_std = np.std(TE_enhanced['mae'], axis=0)
    TE_enhanced_mape_mean = np.mean(TE_enhanced['mape'], axis=0)
    TE_enhanced_mape_std = np.std(TE_enhanced['mape'], axis=0)
    print('MTE based MAE loss is '+str(np.mean(TE_enhanced_mae_mean))+ " , and MAPE is " + str(np.mean(TE_enhanced_mape_mean)))

    with open("AR results/MTGNN_adaptive_performance_result_subgraph_size_20_and_1_expnum_10_AR", "rb") as PDD:   # Unpickling
        Pure_data_driven = pickle.load(PDD)
    Pure_data_driven_mae_mean = np.mean(Pure_data_driven['mae'], axis=0)
    Pure_data_driven_mae_std = np.std(Pure_data_driven['mae'], axis=0)
    Pure_data_driven_mape_mean = np.mean(Pure_data_driven['mape'], axis=0)
    Pure_data_driven_mape_std = np.std(Pure_data_driven['mape'], axis=0)
    print('pure data based MAE loss is ' + str(np.mean(Pure_data_driven_mae_mean))+ " , and MAPE is " + str(np.mean(Pure_data_driven_mape_mean)))

    with open("AR results/MTE_static_performance_result_subgraph_size_20_and_1_expnum_10_AR", "rb") as PDD:   # Unpickling
        MTE_data_driven = pickle.load(PDD)
    MTE_static_data_driven_mae_mean = np.mean(MTE_data_driven['mae'], axis=0)
    MTE_static_data_driven_mae_std = np.std(MTE_data_driven['mae'], axis=0)
    MTE_static_data_driven_mape_mean = np.mean(MTE_data_driven['mape'], axis=0)
    MTE_static_data_driven_mape_std = np.std(MTE_data_driven['mape'], axis=0)
    print('MTE static MAE loss is '+str(np.mean(MTE_static_data_driven_mae_mean))+ " , and MAPE is " + str(np.mean(MTE_static_data_driven_mape_mean)))

    x_range = range(49)
    x_element = list()
    for i in x_range:
        x_element.append(index_2_state[i])

    plt.plot(x_element, MTE_static_data_driven_mae_mean, color = 'red', marker='o', markersize = 3.0, label = 'MTE_STATIC')
    plt.plot(x_element, geo_result_mae_mean, color = 'black', marker='>', markersize = 3.0, label = 'GEO_STATIC')
    plt.plot(x_element, TE_enhanced_mae_mean, color = 'red', marker='<', markersize = 3.0, label = 'MTE_ADAPTIVE')
    plt.plot(x_element, Pure_data_driven_mae_mean,color = 'orange', marker='<', markersize = 3.0, label = 'MTGNN' )

    plt.xlabel('States')
    plt.ylabel('Mean Absolute Error')
    plt.title('Performance comparison across region')
    plt.legend()
    plt.show()
    return



if __name__ == '__main__':
    autoregressive = False
    if autoregressive:
        drawAR()
    else:
        draw()
