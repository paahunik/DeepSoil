import os
import socket
import json
import quadhash_helper as qdh
import matplotlib.pyplot as plt
import numpy as np
import gdal
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import silhouette_score, pairwise_distances

# 5499 quadhashes
def load_dic():
    loaded_dict = {}
    with open("./filter_quads_and_neighbors.json", "r") as file:
        loaded_dict = json.load(file)
    return loaded_dict

def perform_average_of_quadhash(matrix, remove_neg = False):
    if remove_neg:
        mask = np.logical_or(np.isnan(matrix), matrix < 0)
        masked_data = np.ma.masked_array(matrix, mask)
        mean_without_negatives_and_nan = np.mean(masked_data)
        return  mean_without_negatives_and_nan
    else:
        mask = np.logical_or(np.isnan(matrix), matrix == -9999)
        masked_data = np.ma.masked_array(matrix, mask)
        mean_without_negatives_and_nan = np.mean(masked_data)
        return mean_without_negatives_and_nan

def get_max_occuring_val_in_array(arr):
    values, counts = np.unique(arr, return_counts=True)
    max_count_index = np.argmax(counts)
    most_frequent = values[max_count_index]
    return [most_frequent]

def load_land_cover(quad):
    input_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/nlcd/merged_14/" + quad
    if os.path.exists(input_path + '/nlcd.tif'):
        # print("none")
        arr=gdal.Open(input_path + '/nlcd.tif').ReadAsArray()
    else:
        # print("Returning non for nlcd")
        return None
    return scale_val(get_max_occuring_val_in_array(arr), 0, 95)

def scale_val(x, x_min, x_max):
    scaled_x = (x[0] - x_min) / (x_max - x_min)
    return [scaled_x]

def load_climate_condition(quad):
    input_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/koppen_climate/merged_14/" + quad
    if os.path.exists(input_path + '/1km.tif'):
        arr = gdal.Open(input_path + '/1km.tif').ReadAsArray()
    else:
        return None
    return scale_val(get_max_occuring_val_in_array(arr), 0, 30)

def load_soil_properties_gNATSGO(quad):
    properties = []
    input_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/gNATSGO/merged_14/" + quad
    if os.path.exists(input_path + '/30m.tif'):
            arr = gdal.Open(input_path + '/30m.tif').ReadAsArray()
            for ind in range(arr.shape[0]):
                if ind not in [0,3]:
                    continue

                ou = perform_average_of_quadhash(arr[ind], True)
                if np.isnan(ou):
                    ou = -9999
                    properties.append(ou)
                else:
                    if ou < 0.0:
                        ou = -9999
                        properties.append(ou)
                    else:
                        if ind == 0:
                            properties.append(scale_val([ou], 0, 250)[0])
                        if ind == 3:
                            properties.append(scale_val([ou], 0, 50)[0])

            if len(properties) != 2:
                return None

            if np.all(np.array(properties) == -9999):
                return None

    else:
        return None
    return properties

def load_soil_properties_polaris(quad):
    properties = []
    input_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/polaris/merged_14/" + quad
    if os.path.exists(input_path + '/0_5_merged.tif'):
        arr = gdal.Open(input_path + '/0_5_merged.tif').ReadAsArray()
        for ind in range(arr.shape[0]):
            if ind > 8:
                continue

            if ind in [4, 5, 7]: # Ignore saturated swc, residual swc, soil ph
                continue

            # if ind in [0,1,2,3,8]: # Ignore silt, sand, clay, bd, organic matter
            #     continue

            if ind in [0, 1, 2]:
                ou = perform_average_of_quadhash(arr[ind], True)
                ou = scale_val([ou], 0, 100)[0]
            elif ind in [3, 4, 5]:
                ou = perform_average_of_quadhash(arr[ind], True)
                ou = scale_val([ou], 0, 2)[0]
            elif ind in [6, 8]:
                ou = perform_average_of_quadhash(arr[ind], False)
                ou = scale_val([ou], -2, 2)[0]
            else:
                continue

            if np.isnan(ou):
                ou = -9999

                properties.append(ou)
            else:
                properties.append(ou)

        if np.all(np.array(properties) == -9999):
            return None

    else:
        return None
    return properties

def get_season(date_str):
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    if (month == 3 and day >= 20) or (month > 3 and month < 6) or (month == 6 and day <= 20):
        return 1 #Spring
    elif (month == 6 and day >= 21) or (month > 6 and month < 9) or (month == 9 and day <= 21):
        return 2 # "Summer"
    elif (month == 9 and day >= 22) or (month > 9 and month < 12) or (month == 12 and day <= 20):
        return 3 #"Autumn/Fall"
    else:
        return 4 #"Winter"

def make_season_sm():
    input_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/smap_9km/split_14/"
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/smap_9km/season/"
    season_sm_quad = {}
    for quad in os.listdir(input_path):
        os.makedirs(out_path + quad, exist_ok=True)
        seasons = {1:[], 2:[], 3:[], 4:[]}
        for f in os.listdir(input_path + quad):
            season_n = get_season(f.split(".")[0])

            arr = gdal.Open(input_path + quad + "/" + f).ReadAsArray()
            sm = perform_average_of_quadhash(arr, True)
            old_sm = seasons.get(season_n)
            old_sm.append(sm)
            seasons[season_n] = old_sm

        av_sm_1 = np.nanmean(seasons.get(1))
        av_sm_2 = np.nanmean(seasons.get(2))
        av_sm_3 = np.nanmean(seasons.get(3))
        if np.isnan(av_sm_1) or np.isnan(av_sm_2) or np.isnan(av_sm_3):
            # print(quad, [av_sm_1, av_sm_2, av_sm_3])
            continue
            # av_sm_4 = np.nanmean(seasons.get(4))

        season_sm_quad[quad] = [av_sm_1, av_sm_2, av_sm_3]

    return season_sm_quad

def load_smap_sm(quad, season_sm_quad):
    if quad[:12] in season_sm_quad.keys():
        return season_sm_quad[quad[:12]]
    else:
        return None

def read_data_from_file(filename):
    print(filename)
    file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/static_properties/" + filename  + ".txt"

    data = []
    with open(file_path, 'r') as file:
        for line in file:
            features = line.strip().split(',')
            data.append([float(feature) for feature in features])
            # if len(features) != 14:
            #     print(len(features), features)
    print("Performing clustering on array with shape: ", np.array(data).shape)
    return np.array(data)

def davies_bouldin_index(X, labels):
    num_clusters = len(np.unique(labels))
    cluster_centers = [np.mean(X[labels == i], axis=0) for i in range(num_clusters)]

    pairwise_distances_clusters = pairwise_distances(cluster_centers)
    cluster_distances = np.max(pairwise_distances_clusters, axis=1)

    average_cluster_distances = np.zeros(num_clusters)
    for i in range(num_clusters):
        average_cluster_distances[i] = np.mean(
            [np.linalg.norm(cluster_centers[i] - cluster_centers[j])
             for j in range(num_clusters) if j != i])

    db_index = np.mean(cluster_distances / average_cluster_distances)
    return db_index




def scale_dem(data_array):
        x_max = 6500

        data_array[data_array == -999999] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)
        masked_data_array = masked_data_array / x_max
        # masked_data_array = np.nan_to_num(masked_data_array, nan=-1)

        return np.nanmean(masked_data_array)

def load_dem_elevation(quad):
    input_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/dem/merged_14/"
    if os.path.exists(input_path + quad + "/final_elevation_30m.tif"):
        image_dem = gdal.Open(input_path + quad + "/final_elevation_30m.tif").ReadAsArray()
        return scale_dem(image_dem)

def create_samples_for_features(myfil):
    season_sm_quad = make_season_sm()
    my_dic = load_dic()
    file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/static_properties/" + myfil + ".txt"
    fin_arr = []
    count = 0
    print("Total spatial extents: ", len(my_dic.keys()))
    for quad in my_dic.keys():
            # out_climate = load_climate_condition(quad)
            # if out_climate is None:
            #     continue

            # out_landcover = load_land_cover(quad)
            # if out_landcover is None:
            #     continue

            out_gNATSGO = load_soil_properties_gNATSGO(quad)
            if out_gNATSGO is None:
                continue

            out_dem = load_dem_elevation(quad)
            if out_dem is None:
                continue

            out_polaris = load_soil_properties_polaris(quad)
            if out_polaris is None:
                continue

            out_smap = load_smap_sm(quad, season_sm_quad)
            if out_smap is None:
                continue

            count += 1

            final_row = out_gNATSGO.copy()
            final_row.extend(out_polaris)
            final_row.extend(out_smap)
            final_row.extend([out_dem])
            fin_arr.append(final_row)

    # fin_arr_n = np.array(fin_arr)
    # print(fin_arr_n.shape)
    # for f in range(fin_arr_n.shape[-1]):
    #     print(f, np.min(fin_arr_n[:, f]), np.max(fin_arr_n[:, f]))
    print("Found data in : ", count)
    print("Total samples for clusters: ", len(fin_arr), len(fin_arr[0]))
    with open(file_path, "w") as file:
            for row_list in fin_arr:
                row_string = ",".join(str(item) for item in row_list)
                file.write(row_string + "\n")

def plot_clusters(data, season):
    plt.figure(figsize=(10, 6))
    print("Found unique classes:" , season,  len(np.unique(data[:, 2])))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=data[:, 2],  marker='x',palette='viridis', s=10)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("K-means Clustering Visualization")
    plt.legend(title="Cluster")
    plt.show()
    plt.savefig("./my_clusters_season_" + str(season) + ".png")
    plt.close()

def perform_clustering(samples_per_cluster=66, filename = None):
    data = read_data_from_file(filename)
    num_clusters = data.shape[0] // samples_per_cluster

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, init='k-means++')

    num_rows = data.shape[0]
    shuffled_indices = np.random.permutation(num_rows)
    shuffled_data = data[shuffled_indices]

    kmeans.fit(shuffled_data)
    cluster_assignments = kmeans.predict(shuffled_data)

    db_index = davies_bouldin_index(data, cluster_assignments)
    silhouette_avg = silhouette_score(data, cluster_assignments)

    # print("With n_of_clusters: ", no_clusters, round(db_index, 4), round(silhouette_avg, 4))
    # cluster_centers = kmeans.cluster_centers_

    print(len(cluster_assignments))
    clusters = {}
    for i, cluster_label in enumerate(cluster_assignments):
        # print(i, cluster_label)
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(i)

    for cluster_label, sample_indices in clusters.items():
        num_samples_in_cluster = len(sample_indices)
        print(f"Cluster {cluster_label}: {num_samples_in_cluster} samples")


    return kmeans, db_index, silhouette_avg, cluster_assignments

def create_samples_for_features_testing(samples_per_cluster=50, filename=None):
    season_sm_quad = make_season_sm()
    kmeans, db_index, silhouette_avg, clusters = perform_clustering(samples_per_cluster, filename)

    my_dic = load_dic()
    my_clusters = []
    for k in my_dic.keys():
        out = my_dic[k]
        out.extend([k])
        for quad in out:

            out_gNATSGO = load_soil_properties_gNATSGO(quad)
            if out_gNATSGO is None:
                continue

            out_dem = load_dem_elevation(quad)
            if out_dem is None:
                continue

            out_polaris = load_soil_properties_polaris(quad)
            if out_polaris is None:
                continue

            out_smap = load_smap_sm(quad, season_sm_quad)
            if out_smap is None:
                continue

            final_row = out_gNATSGO.copy()
            final_row.extend(out_polaris)
            final_row.extend(out_smap)
            final_row.extend([out_dem])
            cluster_label = kmeans.predict(np.array([final_row]))

            my_clusters.append([quad, cluster_label[0]])
            # print("Cluster Label for the:", quad, " season 1: ",  cluster_label[0])

    # plot_clusters(np.array(my_clusters), 1)
    file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/static_properties/mycluster_fin.txt"

    with open(file_path, 'w') as output_file:
        for i in range(len(my_clusters)):
            output_file.write(str(my_clusters[i][0]) + "," + str(my_clusters[i][1]) + "\n")

def perform_benchmark(filename = 'classes_only_nlcd_gnats'):
    dbs = []
    sis = []
    start = 10
    end = 30
    cluster_sizes = []

    k_values = range(start, end)

    for i in range(start, end):
        _, db, si, clusters = create_samples_for_features_testing(no_clusters=i, filename=filename)
        dbs.append(db)
        sis.append(si)
        cluster_sizes.append(np.bincount(clusters))

    plt.plot(np.arange(start, end, 1), dbs,marker='o', color='red')
    plt.ylim(0,5)

    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Davies-Bouldin Index')

    plt.title('Davies-Bouldin Index vs. Number of Clusters (K)')
    plt.grid(True)
    plt.savefig("./" + filename + "_davies_clusters.png")
    plt.close()

    plt.plot(np.arange(start, end, 1), sis, marker='o', color='green')
    plt.ylim(-1,1)

    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score vs. Number of Clusters (K)')
    plt.grid(True)
    plt.savefig("./" + filename + "_silhouette_clusters.png")
    plt.close()

    for i in range(len(k_values)):
        plt.plot(range(k_values[i]), cluster_sizes[i], label=f'K={k_values[i]}')

    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples Assigned to Each Cluster')
    plt.legend()
    plt.grid(True)
    plt.savefig("./" + filename + "_samples_per_cluster.png")
    plt.close()

if __name__ == '__main__':
    # filename = 'classes_only_koppen_gnats'
    # filename = 'classes_only_koppen_nlcd_gnats'
    # filename = 'classes_only_koppen_nlcd'
    filename = 'classes_only_gnats_dem_polaris_smap'

    # create_samples_for_features(filename)
    create_samples_for_features_testing(filename = filename)
    # perform_benchmark(filename)
    # print("db: ", db_index, silhouette_avg)
    # exit(0)


# Davies-Bouldin index: 17.372642431623543
# Silhouette score: -0.7377480004257347
