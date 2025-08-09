import numpy as np
import matplotlib.pyplot as plt
from som_learn.som import SOM
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

def main():
    """
    An example of how to use the SOM class.
    """
    # 1. Generate sample data
    # Let's create a dataset with 3 clear clusters (e.g., colors)
    data, _ = make_blobs(n_samples=300, centers=3, n_features=3, random_state=42, cluster_std=0.5)

    # Normalize the data to the [0, 1] range for color mapping
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = np.clip(data, 0, 1)

    # 2. Initialize the SOM
    # The input_dim should match the number of features in the data
    som = SOM(dim1=20, dim2=20, input_dim=3, sigma=1.5, eta=0.5, sigma_kernel=0.5, decay_strategy='exponential')

    # 3. Train the SOM and get the cluster labels
    print("Training the SOM...")
    labels = som.fit_predict(data, n_it=100, verbose=True)
    print("Training complete.")

    # 4. Visualize the results

    # Plot the SOM's weights (the "color map")
    plt.figure(figsize=(8, 8))
    plt.title('SOM Color Map')
    plt.imshow(som.params)
    plt.show()

    # Plot the U-Matrix
    print("Plotting U-Matrix...")
    som.plot_u_matrix()

    # Plot the Distance Map
    print("Plotting Distance Map...")
    som.plot_distance_map()

    # 5. Visualize the data points mapped to the SOM

    # Convert 1D labels back to 2D coordinates for plotting
    mapped_i = labels // som.dim2
    mapped_j = labels % som.dim2

    plt.figure(figsize=(10, 10))
    plt.title('Data points mapped to SOM (colored by cluster ID)')
    # Plot the U-matrix as a background
    plt.imshow(som.u_matrix(), cmap='gray', interpolation='none', origin='lower')

    # Plot the mapped data points, colored by their cluster ID
    plt.scatter(mapped_j, mapped_i, c=labels, cmap='viridis', s=20, marker='o')
    plt.colorbar(label='Cluster ID')
    plt.show()


if __name__ == '__main__':
    main()
