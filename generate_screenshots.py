"""Script to generate required screenshots for interactive playground experiments.

Abdul Azeem
Roll number 24i2013
Section BAI5A
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mlcode as ml

os.makedirs('screenshots', exist_ok=True)
rng = np.random.default_rng(2013)


def save_perceptron_screenshot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Generate linearly separable clusters
    c1 = rng.normal(loc=[-2.0, -1.5], scale=0.7, size=(15, 2))
    c2 = rng.normal(loc=[2.0, 1.5], scale=0.7, size=(15, 2))
    X = np.vstack([c1, c2])
    y = np.array([-1] * 15 + [1] * 15)

    w, b, mistakes_hist, total_epochs = ml.perceptron_train(X, y, max_epochs=200, rng=rng)

    # Plot boundary
    xs = np.array([-5.0, 5.0])
    ys = (-w[0] * xs - b) / w[1]
    ax.plot(xs, ys, color='black', linewidth=2.0)

    # Plot points
    ax.scatter(c1[:, 0], c1[:, 1], color='red', s=60, label='Negative One')
    ax.scatter(c2[:, 0], c2[:, 1], color='blue', s=60, label='Positive One')

    ax.set_title(f"Perceptron | Epoch: {total_epochs} | Mistakes: {mistakes_hist[-1]}", fontsize=11)
    ax.set_xlabel('Coordinate X1')
    ax.set_ylabel('Coordinate X2')
    plt.tight_layout()
    plt.savefig('screenshots/perceptron.png', dpi=150)
    plt.close()
    print('Saved screenshots/perceptron.png')


def save_logistic_screenshot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

    c1 = rng.normal(loc=[-1.8, -1.8], scale=0.8, size=(18, 2))
    c2 = rng.normal(loc=[1.8, 1.8], scale=0.8, size=(18, 2))
    X = np.vstack([c1, c2])
    y_raw = np.array([-1] * 18 + [1] * 18)
    y_bin = np.where(y_raw == 1, 1.0, 0.0)

    w, b, loss_hist = ml.gd(X, y_bin, lam=0.01, max_iter=200)

    xs = np.array([-5.0, 5.0])
    ys = (-w[0] * xs - b) / w[1]
    ax.plot(xs, ys, color='black', linewidth=2.0)

    ax.scatter(c1[:, 0], c1[:, 1], color='red', s=60)
    ax.scatter(c2[:, 0], c2[:, 1], color='blue', s=60)

    final_loss = loss_hist[-1]
    ax.set_title(f"Logistic Regression | Iteration: {len(loss_hist)} | Loss: {final_loss:.4f}", fontsize=11)
    ax.set_xlabel('Coordinate X1')
    ax.set_ylabel('Coordinate X2')
    plt.tight_layout()
    plt.savefig('screenshots/logistic.png', dpi=150)
    plt.close()
    print('Saved screenshots/logistic.png')


def save_svm_screenshot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

    c1 = rng.normal(loc=[-2.2, -1.2], scale=0.7, size=(14, 2))
    c2 = rng.normal(loc=[1.8, 1.8], scale=0.7, size=(14, 2))
    X = np.vstack([c1, c2])
    y = np.array([-1] * 14 + [1] * 14)

    C_val = 4.0
    alpha, b_val, _, _ = ml.smo_train(X, y, C=C_val, kernel_func=ml.linear_kernel, max_passes=10, max_iter=2000)
    w = np.sum((alpha * y)[:, np.newaxis] * X, axis=0)

    xs = np.array([-5.0, 5.0])
    ys = (-w[0] * xs - b_val) / w[1]
    ys_margin_neg = (-w[0] * xs - (b_val - 1.0)) / w[1]
    ys_margin_pos = (-w[0] * xs - (b_val + 1.0)) / w[1]

    ax.plot(xs, ys, color='black', linewidth=2.0)
    ax.plot(xs, ys_margin_neg, color='gray', linestyle='--', linewidth=1.5)
    ax.plot(xs, ys_margin_pos, color='gray', linestyle='--', linewidth=1.5)

    ax.scatter(c1[:, 0], c1[:, 1], color='red', s=60)
    ax.scatter(c2[:, 0], c2[:, 1], color='blue', s=60)

    sv_idx = np.where(alpha > 1e-4)[0]
    for idx in sv_idx:
        pt = X[idx]
        ax.scatter(pt[0], pt[1], s=180, facecolors='none', edgecolors='green', linewidths=2.2)

    w_norm = float(np.linalg.norm(w))
    margin_width = 2.0 / w_norm if w_norm > 1e-6 else 0.0

    ax.set_title(f"SVM | Support Vectors: {len(sv_idx)} | Margin Width: {margin_width:.3f}", fontsize=11)
    ax.set_xlabel('Coordinate X1')
    ax.set_ylabel('Coordinate X2')
    plt.tight_layout()
    plt.savefig('screenshots/svm.png', dpi=150)
    plt.close()
    print('Saved screenshots/svm.png')


def save_kmeans_screenshot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

    c1 = rng.normal(loc=[-2.5, -2.0], scale=0.6, size=(15, 2))
    c2 = rng.normal(loc=[0.0, 2.5], scale=0.6, size=(15, 2))
    c3 = rng.normal(loc=[2.5, -2.0], scale=0.6, size=(15, 2))
    X = np.vstack([c1, c2, c3])

    K = 3
    centroids, labels, inertia, hist = ml.kmeans(X, K=K, init='kmeans++', max_iter=50, n_init=5, rng=rng)

    colors = ['blue', 'red', 'green']
    for k in range(K):
        pts_k = X[labels == k]
        ax.scatter(pts_k[:, 0], pts_k[:, 1], color=colors[k], s=60)

    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', s=160, linewidths=3.0, zorder=5)

    ax.set_title(f"KMeans | K: {K} | Iteration: {len(hist)} | Inertia: {inertia:.2f}", fontsize=11)
    ax.set_xlabel('Coordinate X1')
    ax.set_ylabel('Coordinate X2')
    plt.tight_layout()
    plt.savefig('screenshots/kmeans.png', dpi=150)
    plt.close()
    print('Saved screenshots/kmeans.png')


def save_exp1_screenshot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Exp 1: Two well separated groups + 1 wrong point deep inside other group
    c_blue = rng.normal(loc=[2.2, 2.0], scale=0.6, size=(16, 2))
    c_red = rng.normal(loc=[-2.2, -2.0], scale=0.6, size=(16, 2))

    # Add single red point deep inside blue cluster
    outlier = np.array([[2.0, 2.2]])
    X = np.vstack([c_blue, c_red, outlier])
    y = np.array([1] * 16 + [-1] * 16 + [-1])

    # Fit SVM
    alpha, b_val, _, _ = ml.smo_train(X, y, C=4.0, kernel_func=ml.linear_kernel, max_passes=10, max_iter=2000)
    w_svm = np.sum((alpha * y)[:, np.newaxis] * X, axis=0)

    xs = np.array([-5.0, 5.0])
    ys = (-w_svm[0] * xs - b_val) / w_svm[1]
    ax.plot(xs, ys, color='black', linewidth=2.0, label='SVM Boundary')

    ax.scatter(c_blue[:, 0], c_blue[:, 1], color='blue', s=60)
    ax.scatter(c_red[:, 0], c_red[:, 1], color='red', s=60)
    ax.scatter(outlier[:, 0], outlier[:, 1], color='red', s=90, edgecolors='black', linewidths=1.5)

    sv_idx = np.where(alpha > 1e-4)[0]
    for idx in sv_idx:
        pt = X[idx]
        ax.scatter(pt[0], pt[1], s=180, facecolors='none', edgecolors='green', linewidths=2.0)

    w_norm = float(np.linalg.norm(w_svm))
    margin_width = 2.0 / w_norm if w_norm > 1e-6 else 0.0

    ax.set_title(f"Experiment 1 Outlier Point | SVM Margin: {margin_width:.3f} | SVs: {len(sv_idx)}", fontsize=11)
    ax.set_xlabel('Coordinate X1')
    ax.set_ylabel('Coordinate X2')
    plt.tight_layout()
    plt.savefig('screenshots/exp1.png', dpi=150)
    plt.close()
    print('Saved screenshots/exp1.png')


def save_exp2_screenshot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Exp 2: Four groups in XOR pattern (two blue diagonally, two red diagonally)
    g1_blue = rng.normal(loc=[-2.0, 2.0], scale=0.5, size=(12, 2))
    g2_blue = rng.normal(loc=[2.0, -2.0], scale=0.5, size=(12, 2))
    g3_red = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(12, 2))
    g4_red = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(12, 2))

    X = np.vstack([g1_blue, g2_blue, g3_red, g4_red])
    y = np.array([1] * 24 + [-1] * 24)

    # Train linear perceptron on XOR (will oscillate / fail)
    w_p, b_p, mistakes_hist, ep = ml.perceptron_train(X, y, max_epochs=50, rng=rng)
    if np.linalg.norm(w_p) > 1e-6:
        xs = np.array([-5.0, 5.0])
        ys = (-w_p[0] * xs - b_p) / w_p[1]
        ax.plot(xs, ys, color='black', linewidth=2.0, label='Perceptron Linear')

    ax.scatter(g1_blue[:, 0], g1_blue[:, 1], color='blue', s=60)
    ax.scatter(g2_blue[:, 0], g2_blue[:, 1], color='blue', s=60)
    ax.scatter(g3_red[:, 0], g3_red[:, 1], color='red', s=60)
    ax.scatter(g4_red[:, 0], g4_red[:, 1], color='red', s=60)

    ax.set_title(f"Experiment 2 XOR Pattern | Linear Separator Failure | Mistakes: {mistakes_hist[-1]}", fontsize=11)
    ax.set_xlabel('Coordinate X1')
    ax.set_ylabel('Coordinate X2')
    plt.tight_layout()
    plt.savefig('screenshots/exp2.png', dpi=150)
    plt.close()
    print('Saved screenshots/exp2.png')


def save_exp3_screenshot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Exp 3: Three long thin horizontal stripes close together
    x_coords = np.linspace(-4.0, 4.0, 30)
    s1 = np.column_stack([x_coords, np.full(30, 2.0) + rng.normal(0, 0.1, 30)])
    s2 = np.column_stack([x_coords, np.full(30, 0.0) + rng.normal(0, 0.1, 30)])
    s3 = np.column_stack([x_coords, np.full(30, -2.0) + rng.normal(0, 0.1, 30)])

    X = np.vstack([s1, s2, s3])
    K = 3
    centroids, labels, inertia, hist = ml.kmeans(X, K=K, init='random', max_iter=100, n_init=1, rng=rng)

    colors = ['blue', 'red', 'green', 'purple']
    for k in range(K):
        pts_k = X[labels == k]
        ax.scatter(pts_k[:, 0], pts_k[:, 1], color=colors[k], s=50)

    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', s=160, linewidths=3.0, zorder=5)

    ax.set_title(f"Experiment 3 Horizontal Stripes | KMeans K: {K} | Inertia: {inertia:.2f}", fontsize=11)
    ax.set_xlabel('Coordinate X1')
    ax.set_ylabel('Coordinate X2')
    plt.tight_layout()
    plt.savefig('screenshots/exp3.png', dpi=150)
    plt.close()
    print('Saved screenshots/exp3.png')


if __name__ == '__main__':
    save_perceptron_screenshot()
    save_logistic_screenshot()
    save_svm_screenshot()
    save_kmeans_screenshot()
    save_exp1_screenshot()
    save_exp2_screenshot()
    save_exp3_screenshot()
    print('All 7 screenshots created successfully.')
