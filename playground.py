"""Interactive educational playground for linear models and clustering.

Abdul Azeem
Roll number 24i2013
Section BAI5A
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import mlcode as ml


class Playground:
    """Interactive visual environment for machine learning models."""

    def __init__(self):
        self.points = []
        self.labels = []
        self.current_model = 'p'
        self.K = 3

        self.w_perc = np.zeros(2, dtype=float)
        self.b_perc = 0.0
        self.epoch_perc = 0
        self.mistakes_perc = 0

        self.w_lr = np.zeros(2, dtype=float)
        self.b_lr = 0.0
        self.iter_lr = 0
        self.loss_lr = 0.0

        self.alpha_svm = np.array([], dtype=float)
        self.b_svm = 0.0
        self.w_svm = np.zeros(2, dtype=float)
        self.num_sv = 0
        self.margin_width = 0.0

        self.centroids = None
        self.cluster_labels = None
        self.iter_kmeans = 0
        self.inertia_kmeans = 0.0

        self.drag_idx = None
        self.is_dragging = False

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title('Machine Learning Playground')
        self.connect_events()
        self.reset_model()
        self.redraw()

    def connect_events(self):
        """Connect matplotlib mouse and keyboard canvas interactions."""
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def reset_model(self):
        """Reset parameter values for all active learning models."""
        self.w_perc = np.zeros(2, dtype=float)
        self.b_perc = 0.0
        self.epoch_perc = 0
        self.mistakes_perc = 0

        self.w_lr = np.zeros(2, dtype=float)
        self.b_lr = 0.0
        self.iter_lr = 0
        self.loss_lr = 0.0

        self.alpha_svm = np.array([], dtype=float)
        self.b_svm = 0.0
        self.w_svm = np.zeros(2, dtype=float)
        self.num_sv = 0
        self.margin_width = 0.0

        self.centroids = None
        self.cluster_labels = None
        self.iter_kmeans = 0
        self.inertia_kmeans = 0.0

    def step_perceptron(self):
        """Perform one complete pass over points with perceptron rule."""
        if len(self.points) == 0:
            return
        X = np.array(self.points, dtype=float)
        y = np.array(self.labels, dtype=float)
        self.epoch_perc += 1
        mistakes = 0

        for i in range(len(X)):
            val = y[i] * (np.dot(X[i], self.w_perc) + self.b_perc)
            if val <= 0.0:
                self.w_perc += y[i] * X[i]
                self.b_perc += y[i]
                mistakes += 1

        self.mistakes_perc = mistakes

    def step_logistic(self):
        """Execute twenty iterations of batch gradient descent on log loss."""
        if len(self.points) == 0:
            return
        X = np.array(self.points, dtype=float)
        y_binary = np.where(np.array(self.labels) == 1, 1.0, 0.0)

        step_size = 0.05
        lam = 0.01

        for _ in range(20):
            self.iter_lr += 1
            z = np.dot(X, self.w_lr) + self.b_lr
            p = ml.sigmoid(z)
            diff = p - y_binary
            n = float(len(y_binary))

            grad_w = np.dot(X.T, diff) / n + lam * self.w_lr
            grad_b = float(np.sum(diff) / n)

            self.w_lr -= step_size * grad_w
            self.b_lr -= step_size * grad_b

        z = np.dot(X, self.w_lr) + self.b_lr
        self.loss_lr = float(np.mean(ml.softplus(z) - y_binary * z) + 0.5 * lam * np.sum(self.w_lr**2))

    def step_svm(self):
        """Train dual support vector machine via sequential minimal optimization."""
        if len(self.points) < 2:
            return
        X = np.array(self.points, dtype=float)
        y = np.array(self.labels, dtype=float)

        if len(np.unique(y)) < 2:
            return

        C_val = 4.0
        alpha, b_val, _, _ = ml.smo_train(
            X, y, C=C_val, kernel_func=ml.linear_kernel, max_passes=10, max_iter=2000
        )

        self.alpha_svm = alpha
        self.b_svm = b_val
        self.w_svm = np.sum((alpha * y)[:, np.newaxis] * X, axis=0)

        w_norm = float(np.linalg.norm(self.w_svm))
        self.margin_width = (2.0 / w_norm) if w_norm > 1e-6 else 0.0
        self.num_sv = int(np.sum(alpha > 1e-4))

    def step_kmeans(self):
        """Execute single Lloyd clustering update iteration."""
        if len(self.points) < self.K:
            return
        X = np.array(self.points, dtype=float)
        n_samples = len(X)

        if self.centroids is None or len(self.centroids) != self.K:
            rng = np.random.default_rng(2013 + self.iter_kmeans)
            idx = rng.choice(n_samples, size=self.K, replace=False)
            self.centroids = X[idx].copy()
            self.iter_kmeans = 0

        self.iter_kmeans += 1

        dists_sq = np.zeros((n_samples, self.K), dtype=float)
        for k in range(self.K):
            dists_sq[:, k] = np.sum((X - self.centroids[k]) ** 2, axis=1)

        self.cluster_labels = np.argmin(dists_sq, axis=1)
        self.inertia_kmeans = float(np.sum(np.min(dists_sq, axis=1)))

        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.K):
            cluster_pts = X[self.cluster_labels == k]
            if len(cluster_pts) > 0:
                new_centroids[k] = np.mean(cluster_pts, axis=0)
            else:
                new_centroids[k] = self.centroids[k]

        self.centroids = new_centroids

    def step_model(self):
        """Dispatch step command to active selected algorithm."""
        if self.current_model == 'p':
            self.step_perceptron()
        elif self.current_model == 'l':
            self.step_logistic()
        elif self.current_model == 's':
            self.step_svm()
        elif self.current_model == 'k':
            self.step_kmeans()
        self.redraw()

    def on_press(self, event):
        """Handle mouse click and drag selection events."""
        if event.inaxes != self.ax:
            return

        click_pos = np.array([event.xdata, event.ydata], dtype=float)

        if len(self.points) > 0:
            dists = [np.linalg.norm(click_pos - np.array(p)) for p in self.points]
            min_dist = min(dists)
            if min_dist < 0.3:
                self.drag_idx = int(np.argmin(dists))
                self.is_dragging = True
                return

        if event.button == 1:
            self.points.append([float(event.xdata), float(event.ydata)])
            self.labels.append(1)
        elif event.button == 3:
            self.points.append([float(event.xdata), float(event.ydata)])
            self.labels.append(-1)

        self.redraw()

    def on_motion(self, event):
        """Update point coordinates while dragging across canvas."""
        if not self.is_dragging or self.drag_idx is None:
            return
        if event.inaxes != self.ax:
            return

        self.points[self.drag_idx] = [float(event.xdata), float(event.ydata)]
        self.redraw()

    def on_release(self, event):
        """Finalize dragging and trigger immediate model updates."""
        if self.is_dragging:
            self.is_dragging = False
            self.drag_idx = None

            if self.current_model == 's':
                self.step_svm()
            elif self.current_model == 'l':
                self.step_logistic()

            self.redraw()

    def on_key(self, event):
        """Process keyboard shortcut inputs for selection and actions."""
        key = event.key
        if key == ' ':
            self.step_model()
        elif key in ['p', 'l', 's', 'k']:
            self.current_model = key
            self.redraw()
        elif key in ['2', '3', '4', '5', '6']:
            self.K = int(key)
            self.centroids = None
            self.cluster_labels = None
            self.iter_kmeans = 0
            self.redraw()
        elif key == 'r':
            self.reset_model()
            self.redraw()
        elif key == 'u':
            if len(self.points) > 0:
                self.points.pop()
                self.labels.pop()
                self.reset_model()
                self.redraw()
        elif key == 'c':
            self.points = []
            self.labels = []
            self.reset_model()
            self.redraw()

    def draw_boundary_line(self, w, b, style='solid', color='black', label=None):
        """Draw linear hyperplane across visible coordinate frame."""
        if abs(w[1]) > 1e-6:
            xs = np.array([-5.0, 5.0])
            ys = (-w[0] * xs - b) / w[1]
            self.ax.plot(xs, ys, linestyle=style, color=color, linewidth=1.8, label=label)
        elif abs(w[0]) > 1e-6:
            x_val = -b / w[0]
            self.ax.axvline(x=x_val, linestyle=style, color=color, linewidth=1.8, label=label)

    def redraw(self):
        """Refresh canvas graphics and live header statistics."""
        self.ax.clear()
        self.ax.set_xlim(-5.0, 5.0)
        self.ax.set_ylim(-5.0, 5.0)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle=':', alpha=0.6)

        title_text = ''
        if self.current_model == 'p':
            title_text = f"Perceptron | Epoch: {self.epoch_perc} | Mistakes: {self.mistakes_perc}"
            if np.linalg.norm(self.w_perc) > 1e-6:
                self.draw_boundary_line(self.w_perc, self.b_perc, style='solid', color='black')
        elif self.current_model == 'l':
            title_text = f"Logistic Regression | Iteration: {self.iter_lr} | Loss: {self.loss_lr:.4f}"
            if np.linalg.norm(self.w_lr) > 1e-6:
                self.draw_boundary_line(self.w_lr, self.b_lr, style='solid', color='black')
        elif self.current_model == 's':
            title_text = f"SVM | Support Vectors: {self.num_sv} | Margin Width: {self.margin_width:.3f}"
            if np.linalg.norm(self.w_svm) > 1e-6:
                self.draw_boundary_line(self.w_svm, self.b_svm, style='solid', color='black')
                self.draw_boundary_line(self.w_svm, self.b_svm - 1.0, style='dashed', color='gray')
                self.draw_boundary_line(self.w_svm, self.b_svm + 1.0, style='dashed', color='gray')

                if len(self.alpha_svm) == len(self.points):
                    sv_idx = np.where(self.alpha_svm > 1e-4)[0]
                    for idx in sv_idx:
                        pt = self.points[idx]
                        self.ax.scatter(pt[0], pt[1], s=180, facecolors='none', edgecolors='green', linewidths=2.0)
        elif self.current_model == 'k':
            title_text = f"KMeans | K: {self.K} | Iteration: {self.iter_kmeans} | Inertia: {self.inertia_kmeans:.2f}"
            if self.centroids is not None:
                self.ax.scatter(
                    self.centroids[:, 0],
                    self.centroids[:, 1],
                    marker='x',
                    s=140,
                    color='black',
                    linewidths=2.5,
                    zorder=5
                )

        self.ax.set_title(title_text, fontsize=10)
        self.ax.set_xlabel('Coordinate X1')
        self.ax.set_ylabel('Coordinate X2')

        if len(self.points) > 0:
            pts = np.array(self.points)
            if self.current_model == 'k' and self.cluster_labels is not None and len(self.cluster_labels) == len(pts):
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
                for k_i in range(self.K):
                    k_pts = pts[self.cluster_labels == k_i]
                    if len(k_pts) > 0:
                        c_color = colors[k_i % len(colors)]
                        self.ax.scatter(k_pts[:, 0], k_pts[:, 1], color=c_color, s=50, zorder=3)
            else:
                for pt, lbl in zip(self.points, self.labels):
                    pt_color = 'blue' if lbl == 1 else 'red'
                    self.ax.scatter(pt[0], pt[1], color=pt_color, s=50, zorder=3)

        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    app = Playground()
    plt.show()
