# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
import os
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# --- Input Expansion Utilities ---

def gaussian(x, mu=0, sig=0.1):
    """Gaussian function with specified mean and standard deviation."""
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def expand_input(x, n_components=5):
    """Converts a scalar in [0,1] to a normalized Gaussian-encoded vector."""
    positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sigma = 0.1
    expanded = np.array([gaussian(x, mu=pos, sig=sigma) for pos in positions])
    return expanded / np.sum(expanded)

def expand_2d_input(X):
    """Converts a 2D input array to a 10D representation using Gaussian encoding."""
    N = X.shape[0]
    X_expanded = np.zeros((N, 10), dtype=np.float32)
    for i in range(N):
        X_expanded[i, :5] = expand_input(X[i, 0])
        X_expanded[i, 5:] = expand_input(X[i, 1])
    return X_expanded

# --- Visualization of Gaussian Receptive Fields ---

def demonstrate_input_expansion():
    """Generates a plot demonstrating the Gaussian input expansion."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    x = np.linspace(0, 1, 1001)
    positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sigma = 0.1
    styles = ['--', '-', '-.', ':', '-']
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i, pos in enumerate(positions):
        y = gaussian(x, mu=pos, sig=sigma)
        ax1.plot(x, y, styles[i], color=colors[i], label=f'μ={pos}')
    
    example_x = 0.3
    ax1.axvline(x=example_x, color='black', linestyle='--', alpha=0.5)
    ax1.text(example_x + 0.02, 0.9, f'x={example_x}', rotation=0)
    
    responses = np.array([gaussian(example_x, mu=pos, sig=sigma) for pos in positions])
    normalized_responses = responses / np.sum(responses)
    
    ax2.bar(positions, normalized_responses, color=colors, width=0.1, alpha=0.7)
    
    ax1.set_title("Gaussian Receptive Fields", fontsize=24)
    ax1.set_xlabel("Input value", fontsize=20)
    ax1.set_ylabel("Response", fontsize=20)
    ax1.grid(True)
    ax1.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    ax2.set_title(f"Normalized Responses for x={example_x}", fontsize=24)
    ax2.set_xlabel("Receptive Field Centers", fontsize=20)
    ax2.set_ylabel("Normalized Response", fontsize=20)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    for i, (pos, resp) in enumerate(zip(positions, normalized_responses)):
        ax2.text(pos, resp + 0.02, f'{resp:.3f}', ha='center', va='bottom', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    try:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'input_expansion_demo_relu.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Demonstration saved to: {save_path}")
    except Exception as e:
        print(f"Error saving demonstration: {e}")
    
    plt.show()
    plt.close()

# --- Dataset Creation ---

def create_datasets():
    """Creates three synthetic 2D datasets for classification."""
    X1, y1 = make_moons(noise=0.3, random_state=0)
    X2, y2 = make_circles(noise=0.2, factor=0.5, random_state=1)
    X3, y3 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                 random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X3 += 2 * rng.uniform(size=X3.shape)

    return [(X1, y1, "make_moons"), (X2, y2, "make_circles"), (X3, y3, "linearly_separable")]

# --- Neural Network Helpers ---

def softmax(z):
    """Computes softmax probabilities from scores."""
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_ = np.exp(z_shift)
    return exp_ / np.sum(exp_, axis=1, keepdims=True)


# --- Custom Adam Optimizer ---

class AdamParam:
    """Helper class to store Adam optimizer state."""
    def __init__(self, shape):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)

def adam_update(param, grad, adam_state, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=1):
    """Performs a single Adam optimization step."""
    adam_state.m = beta1 * adam_state.m + (1 - beta1) * grad
    adam_state.v = beta2 * adam_state.v + (1 - beta2) * (grad**2)
    m_hat = adam_state.m / (1 - beta1**t)
    v_hat = adam_state.v / (1 - beta2**t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)


# --- Model Definitions ---

class NoHiddenLayerBiprop:
    """
    Full biprop without hidden layer: Directly from expanded input to output.
    """
    def __init__(self, lr=0.001, n_epochs=50001):
        self.lr = lr
        self.n_epochs = n_epochs
        self.input_dim = 10  # Expanded input

    def fit(self, X, y):
        rng = np.random.RandomState(42)
        N, D = X.shape
        X_expanded = expand_2d_input(X)
        self.W = rng.normal(scale=0.1, size=(self.input_dim, 2)).astype(np.float32)
        self.b = np.zeros((2,), dtype=np.float32)
        self.adam_W = AdamParam(self.W.shape)
        self.adam_b = AdamParam(self.b.shape)
        self.adam_X = AdamParam(X.shape)
        for epoch in range(1, self.n_epochs+1):
            z = X_expanded @ self.W + self.b
            prob = softmax(z)
            grad_z = np.copy(prob)
            grad_z[np.arange(N), y] -= 1.0
            grad_W = (X_expanded.T @ grad_z) / N
            grad_b = np.mean(grad_z, axis=0)
            adam_update(self.W, grad_W, self.adam_W, lr=self.lr, t=epoch)
            adam_update(self.b, grad_b, self.adam_b, lr=self.lr, t=epoch)
            grad_expanded = (grad_z @ self.W.T) / N
            grad_X = np.zeros_like(X)
            grad_X[:, 0] = np.mean(grad_expanded[:, :5], axis=1)
            grad_X[:, 1] = np.mean(grad_expanded[:, 5:], axis=1)
            adam_update(X, grad_X, self.adam_X, lr=(self.lr*1e-4), t=epoch)
            X_expanded = expand_2d_input(X)
        return self

    def predict_proba(self, X):
        X_expanded = expand_2d_input(X)
        z = X_expanded @ self.W + self.b
        return softmax(z)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

class NoHiddenLayerBLBP:
    """
    Half biprop without hidden layer.
    """
    def __init__(self, lr=0.001, n_epochs=50001):
        self.lr = lr
        self.n_epochs = n_epochs
        self.input_dim = 10

    def fit(self, X, y):
        rng = np.random.RandomState(123)
        N, D = X.shape
        X_expanded = expand_2d_input(X)
        self.W = rng.normal(scale=0.1, size=(self.input_dim, 2)).astype(np.float32)
        self.b = np.zeros((2,), dtype=np.float32)
        self.adam_W = AdamParam(self.W.shape)
        self.adam_b = AdamParam(self.b.shape)
        self.adam_X = AdamParam(X.shape)
        half = self.n_epochs // 2
        for epoch in range(1, self.n_epochs+1):
            z = X_expanded @ self.W + self.b
            prob = softmax(z)
            grad_z = np.copy(prob)
            grad_z[np.arange(N), y] -= 1.0
            grad_W = (X_expanded.T @ grad_z) / N
            grad_b = np.mean(grad_z, axis=0)
            adam_update(self.W, grad_W, self.adam_W, lr=self.lr, t=epoch)
            adam_update(self.b, grad_b, self.adam_b, lr=self.lr, t=epoch)
            if epoch <= half:
                grad_expanded = (grad_z @ self.W.T) / N
                grad_X = np.zeros_like(X)
                grad_X[:, 0] = np.mean(grad_expanded[:, :5], axis=1)
                grad_X[:, 1] = np.mean(grad_expanded[:, 5:], axis=1)
                adam_update(X, grad_X, self.adam_X, lr=(self.lr*1e-4), t=epoch)
                X_expanded = expand_2d_input(X)
        return self

    def predict_proba(self, X):
        X_expanded = expand_2d_input(X)
        z = X_expanded @ self.W + self.b
        return softmax(z)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# --- Visualization Utilities ---

BASE_COLORS_2 = [
    (1.0, 0.0, 0.0),  # class=0 => red
    (0.0, 0.0, 1.0)   # class=1 => blue
]

def get_interval_color_2(class_idx, p):
    """
    class_idx in {0,1} => base colors: red, blue
    p => predicted probability of the 'winning' class
    We define intervals for coloring:
        p < 0.2 => white
        0.2<=p<0.4 => factor=0.3
        0.4<=p<0.6 => factor=0.5
        0.6<=p<0.8 => factor=0.7
        p>=0.8     => factor=0.9
    """
    if p < 0.2:
        return (1.0, 1.0, 1.0)  # white
    if p < 0.4:
        factor = 0.3
    elif p < 0.6:
        factor = 0.5
    elif p < 0.8:
        factor = 0.7
    else:
        factor = 0.9
    base_r, base_g, base_b = BASE_COLORS_2[class_idx]
    r = (1.0 - factor) + factor * base_r
    g = (1.0 - factor) + factor * base_g
    b = (1.0 - factor) + factor * base_b
    return (r, g, b)

def plot_layered_background(ax, clf, xx, yy):
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_proba = clf.predict_proba(grid_points)  # shape [M,2]

    M = grid_points.shape[0]
    Z_colors = np.zeros((M,3), dtype=np.float32)
    for i in range(M):
        row_probs = Z_proba[i]
        c_ = np.argmax(row_probs)
        p_win = row_probs[c_]
        color_rgb = get_interval_color_2(c_, p_win)
        Z_colors[i] = color_rgb

    nrows, ncols = xx.shape
    Z_colors_reshaped = Z_colors.reshape(nrows, ncols, 3)
    ax.imshow(Z_colors_reshaped, origin='lower',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              vmin=0, vmax=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.2, color='gray', linestyle='-', linewidth=0.5)


# --- Main Execution ---

def main():
    demonstrate_input_expansion()
    ds_list = create_datasets()

    clf_list = [
        ("SVM RBF", SVC(kernel="rbf", probability=True, random_state=42)),
        ("GP", GaussianProcessClassifier(RBF(1.0), random_state=42)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(10,10), max_iter=50001, alpha=0.0001, random_state=42)),
        ("BL", NoHiddenLayerBiprop(lr=0.001, n_epochs=5001)),
        ("BL->BP", NoHiddenLayerBLBP(lr=0.001, n_epochs=5001)),
    ]

    ncols = 1 + len(clf_list)
    fig = plt.figure(figsize=(6*ncols, 18))  # Changed from (6*ncols, 15)

    # Create subplot grid with adjusted spacing
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, ncols, figure=fig)
    gs.update(wspace=0.1, hspace=0.15)  # Match sigmoid spacing

    axes = gs.subplots()
    axes = axes.ravel()
    idx_subplot = 0

    for row_idx, (X, y, ds_name) in enumerate(ds_list):
        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        # scale to [0,1]
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # define grid for background with fixed limits for ticks
        x_min, x_max = -0.15, 1.15
        y_min, y_max = -0.15, 1.15
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        for col_idx in range(ncols):
            ax = axes[idx_subplot]
            idx_subplot += 1

            # Set title for the first row of plots
            if row_idx == 0:
                title = "Input" if col_idx == 0 else clf_list[col_idx - 1][0]
                ax.set_title(title, fontsize=35)

            if col_idx == 0:
                # Set dataset name as Y-label for the first column
                ax.set_ylabel(ds_name, fontsize=33, labelpad=22)
                # Plot input data
                ax.scatter(X_train_scaled[:,0], X_train_scaled[:,1],
                           c=y_train, cmap=ListedColormap(["red","blue"]),
                           edgecolors='k')
                ax.scatter(X_test_scaled[:,0], X_test_scaled[:,1],
                           c=y_test, cmap=ListedColormap(["red","blue"]),
                           edgecolors='k', alpha=0.6, marker='s')
                
                # Set custom ticks and labels for the first column
                ticks = [-0.1, 0.5, 1.1]
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels([f'{t:.1f}' for t in ticks], fontsize=24)
                ax.set_yticklabels([f'{t:.1f}' for t in ticks], fontsize=24)
                # Hide the first y-tick label to create the shared corner
                ax.get_yticklabels()[0].set_visible(False)
                
                # Position the '-0.1' x-label exactly in the corner
                label = ax.get_xticklabels()[0]
                label.set_ha('right')
                label.set_va('top')

            else:
                # Train and plot classifier results
                clf_name, clf_obj = clf_list[col_idx - 1]
                X_train_copy = X_train_scaled.copy()
                clf_obj.fit(X_train_copy, y_train)
                score_val = clf_obj.score(X_test_scaled, y_test)

                # Plot background and data points
                plot_layered_background(ax, clf_obj, xx, yy)
                ax.scatter(X_train_scaled[:,0], X_train_scaled[:,1],
                           c=y_train, cmap=ListedColormap(["red","blue"]), edgecolors='k')
                ax.scatter(X_test_scaled[:,0], X_test_scaled[:,1],
                           c=y_test, cmap=ListedColormap(["red","blue"]),
                           edgecolors='k', alpha=0.6, marker='s')

                # Remove ticks for all classifier plots
                ax.set_xticks([])
                ax.set_yticks([])

                # Show test accuracy in corner
                ax.text(
                    x_max, y_min,
                    f"{score_val*100:.1f}%",
                    size=26,
                    ha="right",
                    va="bottom",
                    bbox=dict(facecolor="white", alpha=0.6, boxstyle="round", pad=0.3)
                )

            # Common settings for all subplots
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2, color='gray', linestyle='-', linewidth=0.5)

    # First legend for class colors
    legend_elements = [
        Patch(facecolor='red', label='Class 0'),
        Patch(facecolor='blue', label='Class 1')
    ]
    fig.legend(handles=legend_elements, 
              loc='lower center', 
              ncol=2, 
              bbox_to_anchor=(0.5, -0.12),
              fontsize=35,
              markerscale=2,
              handlelength=3,
              handletextpad=0.5)

    # Probability legend
    prob_legend_elements = [
        Patch(facecolor=(1.0, 1.0, 1.0), label='p < 0.2', edgecolor='black'),
        Patch(facecolor=(1.0, 0.7, 0.7), label='0.2 ≤ p < 0.4'),
        Patch(facecolor=(1.0, 0.5, 0.5), label='0.4 ≤ p < 0.6'),
        Patch(facecolor=(1.0, 0.3, 0.3), label='0.6 ≤ p < 0.8'),
        Patch(facecolor=(1.0, 0.1, 0.1), label='p ≥ 0.8')
    ]

    fig.legend(handles=prob_legend_elements, 
              loc='lower center', 
              ncol=5, 
              bbox_to_anchor=(0.5, -0.05),
              fontsize=30,
              title='      Probability intervals for Red color\n        (Same intervals apply to Blue)',
              title_fontsize=36,
              markerscale=2,
              handlelength=3,
              handletextpad=0.5)

    # Save results with error handling
    try:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'visualization_synthetic.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Results successfully saved to: {save_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

    plt.show()

if __name__ == "__main__":
    main()