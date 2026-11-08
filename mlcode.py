"""Module providing scratch implementations of machine learning models.

Abdul Azeem
Roll number 24i2013
Section BAI5A
"""

import math
import time
from collections import deque
import numpy as np


def perceptron_train(X, y, max_epochs=1000, rng=None):
    """Train standard primal perceptron using stochastic updates."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0
    mistakes_per_epoch = []
    total_epochs = 0

    for epoch in range(max_epochs):
        total_epochs += 1
        mistakes = 0
        if rng is not None:
            order = rng.permutation(n_samples)
        else:
            order = np.arange(n_samples)

        for i in order:
            margin_val = y[i] * (np.dot(X[i], w) + b)
            if margin_val <= 0.0:
                w += y[i] * X[i]
                b += float(y[i])
                mistakes += 1

        mistakes_per_epoch.append(mistakes)
        if mistakes == 0:
            break

    return w, b, mistakes_per_epoch, total_epochs


def perceptron_predict(X, w, b):
    """Predict binary class labels in positive one and negative one."""
    scores = np.dot(X, w) + b
    preds = np.where(scores >= 0.0, 1, -1)
    return preds


def novikoff_bound(X, w_star, margin):
    """Compute Novikoff upper bound on total perceptron mistakes."""
    norms = np.sqrt(np.sum(X**2, axis=1) + 1.0)
    R = float(np.max(norms))
    bound = float((R / margin) ** 2)
    return R, margin, bound


def averaged_perceptron(X, y, epochs=50, rng=None):
    """Train averaged perceptron using efficient order d accumulator trick."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0
    u_w = np.zeros(n_features, dtype=float)
    u_b = 0.0
    t = 0
    history = []

    for epoch in range(epochs):
        mistakes = 0
        if rng is not None:
            order = rng.permutation(n_samples)
        else:
            order = np.arange(n_samples)

        for i in order:
            t += 1
            margin_val = y[i] * (np.dot(X[i], w) + b)
            if margin_val <= 0.0:
                w += y[i] * X[i]
                b += float(y[i])
                u_w += float(t) * (y[i] * X[i])
                u_b += float(t) * float(y[i])
                mistakes += 1

        w_avg = w - (u_w / float(t))
        b_avg = b - (u_b / float(t))
        history.append({
            'epoch': epoch + 1,
            'mistakes': mistakes,
            'w_plain': w.copy(),
            'b_plain': b,
            'w_avg': w_avg.copy(),
            'b_avg': b_avg
        })

    w_final_avg = w - (u_w / float(t))
    b_final_avg = b - (u_b / float(t))
    return w, b, w_final_avg, b_final_avg, history


def kernel_perceptron(X, y, kernel_func, max_epochs=100):
    """Train dual kernel perceptron using precomputed Gram matrix."""
    n_samples = X.shape[0]
    K = kernel_func(X, X)
    alpha = np.zeros(n_samples, dtype=float)
    b = 0.0
    mistakes_per_epoch = []

    for epoch in range(max_epochs):
        mistakes = 0
        for i in range(n_samples):
            pred_score = np.sum(alpha * y * K[:, i]) + b
            if y[i] * pred_score <= 0.0:
                alpha[i] += 1.0
                b += float(y[i])
                mistakes += 1
        mistakes_per_epoch.append(mistakes)
        if mistakes == 0:
            break

    return alpha, b, mistakes_per_epoch


def multiclass_perceptron(X, y, num_classes=10, max_epochs=50, rng=None):
    """Train multiclass perceptron with one weight vector per class."""
    n_samples, n_features = X.shape
    W = np.zeros((num_classes, n_features), dtype=float)
    b = np.zeros(num_classes, dtype=float)
    history = []

    for epoch in range(max_epochs):
        mistakes = 0
        if rng is not None:
            order = rng.permutation(n_samples)
        else:
            order = np.arange(n_samples)

        for i in order:
            scores = np.dot(W, X[i]) + b
            pred_class = int(np.argmax(scores))
            true_class = int(y[i])
            if pred_class != true_class:
                W[true_class] += X[i]
                W[pred_class] -= X[i]
                b[true_class] += 1.0
                b[pred_class] -= 1.0
                mistakes += 1

        history.append({'epoch': epoch + 1, 'mistakes': mistakes})
        if mistakes == 0:
            break

    return W, b, history


def accuracy(y_true, y_pred):
    """Calculate classification accuracy proportion."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.mean(y_t == y_p))


def confusion_matrix(y_true, y_pred, num_classes=None):
    """Construct square confusion matrix for predictions."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    if num_classes is None:
        c_max = max(int(np.max(y_t)), int(np.max(y_p))) + 1
    else:
        c_max = int(num_classes)

    mat = np.zeros((c_max, c_max), dtype=int)
    for t_val, p_val in zip(y_t, y_p):
        if 0 <= t_val < c_max and 0 <= p_val < c_max:
            mat[t_val, p_val] += 1
    return mat


def precision_recall_f1(y_true, y_pred, pos_label=1):
    """Calculate precision recall and f1 score for specified target label."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    tp = int(np.sum((y_t == pos_label) & (y_p == pos_label)))
    fp = int(np.sum((y_t != pos_label) & (y_p == pos_label)))
    fn = int(np.sum((y_t == pos_label) & (y_p != pos_label)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_val = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1_val


def macro_f1(y_true, y_pred):
    """Calculate unweighted macro average f1 across present classes."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_t, y_p]))
    f1_scores = []
    for c in classes:
        _, _, f1_c = precision_recall_f1(y_t, y_p, pos_label=c)
        f1_scores.append(f1_c)
    return float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0


def sigmoid(z):
    """Compute numerically stable sigmoid activation on input array z."""
    z_arr = np.asarray(z, dtype=float)
    pos_mask = z_arr >= 0.0
    out = np.zeros_like(z_arr)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z_arr[pos_mask]))
    exp_neg = np.exp(z_arr[~pos_mask])
    out[~pos_mask] = exp_neg / (1.0 + exp_neg)
    return out


def softplus(z):
    """Compute numerically stable log of one plus exponential of z."""
    z_arr = np.asarray(z, dtype=float)
    return np.maximum(0.0, z_arr) + np.log1p(np.exp(-np.abs(z_arr)))


def lr_objective(w, b, X, y, lam):
    """Compute binary logistic loss objective with L2 regularisation."""
    z = np.dot(X, w) + b
    log_terms = softplus(z) - y * z
    n = float(len(y))
    loss = float(np.sum(log_terms) / n + 0.5 * lam * np.sum(w**2))
    return loss


def lr_gradient(w, b, X, y, lam):
    """Compute analytic gradient of logistic regression objective."""
    z = np.dot(X, w) + b
    p = sigmoid(z)
    diff = p - y
    n = float(len(y))
    grad_w = np.dot(X.T, diff) / n + lam * w
    grad_b = float(np.sum(diff) / n)
    return grad_w, grad_b


def lr_hessian(w, b, X, y, lam):
    """Compute full Hessian matrix for weights and bias."""
    z = np.dot(X, w) + b
    p = sigmoid(z)
    d = p * (1.0 - p)
    n = float(len(y))
    d_features = len(w)

    X_weighted = X.T * d
    H_ww = np.dot(X_weighted, X) / n + lam * np.eye(d_features)
    H_wb = np.dot(X.T, d) / n
    H_bb = float(np.sum(d) / n)

    H = np.zeros((d_features + 1, d_features + 1), dtype=float)
    H[:d_features, :d_features] = H_ww
    H[:d_features, d_features] = H_wb
    H[d_features, :d_features] = H_wb
    H[d_features, d_features] = H_bb
    return H


def gradient_check(w, b, X, y, lam, eps=1e-6):
    """Compare analytic gradient against two sided finite differences."""
    grad_w, grad_b = lr_gradient(w, b, X, y, lam)
    grad_analytic = np.append(grad_w, grad_b)
    grad_numeric = np.zeros_like(grad_analytic)

    theta = np.append(w, b)
    d_dim = len(theta)

    for i in range(d_dim):
        theta_pos = theta.copy()
        theta_neg = theta.copy()
        theta_pos[i] += eps
        theta_neg[i] -= eps

        loss_pos = lr_objective(theta_pos[:-1], theta_pos[-1], X, y, lam)
        loss_neg = lr_objective(theta_neg[:-1], theta_neg[-1], X, y, lam)
        grad_numeric[i] = (loss_pos - loss_neg) / (2.0 * eps)

    diff_norm = np.linalg.norm(grad_analytic - grad_numeric)
    sum_norm = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric)
    rel_error = float(diff_norm / sum_norm) if sum_norm > 0 else 0.0
    return rel_error


def gd(X, y, lam, max_iter=2000, tol=1e-12):
    """Execute batch gradient descent with step size one over Lipschitz constant."""
    n_samples, n_features = X.shape
    XTX = np.dot(X.T, X)
    eig_vals = np.linalg.eigvalsh(XTX / (4.0 * float(n_samples)))
    L = float(eig_vals[-1]) + lam
    step_size = 1.0 / L

    w = np.zeros(n_features, dtype=float)
    b = 0.0
    loss_history = []

    for it in range(max_iter):
        loss = lr_objective(w, b, X, y, lam)
        loss_history.append(loss)
        grad_w, grad_b = lr_gradient(w, b, X, y, lam)
        grad_norm = math.sqrt(float(np.sum(grad_w**2) + grad_b**2))
        if grad_norm < tol:
            break
        w -= step_size * grad_w
        b -= step_size * grad_b

    return w, b, loss_history


def sgd(X, y, lam, eta=0.02, max_epochs=100, batch_size=32, rng=None):
    """Execute mini batch stochastic gradient descent with decaying step."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0
    loss_history = []
    step_count = 0

    for epoch in range(max_epochs):
        loss = lr_objective(w, b, X, y, lam)
        loss_history.append(loss)

        if rng is not None:
            order = rng.permutation(n_samples)
        else:
            order = np.arange(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            step_count += 1
            batch_idx = order[start_idx : start_idx + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            decay_eta = eta / math.sqrt(float(step_count))
            grad_w, grad_b = lr_gradient(w, b, X_batch, y_batch, lam)
            w -= decay_eta * grad_w
            b -= decay_eta * grad_b

    return w, b, loss_history


def momentum_gd(X, y, lam, eta=None, beta=0.9, max_iter=2000, tol=1e-12):
    """Execute gradient descent with classical Polyak momentum."""
    n_samples, n_features = X.shape
    if eta is None:
        XTX = np.dot(X.T, X)
        eig_vals = np.linalg.eigvalsh(XTX / (4.0 * float(n_samples)))
        L = float(eig_vals[-1]) + lam
        eta = 1.0 / L

    w = np.zeros(n_features, dtype=float)
    b = 0.0
    v_w = np.zeros(n_features, dtype=float)
    v_b = 0.0
    loss_history = []

    for it in range(max_iter):
        loss = lr_objective(w, b, X, y, lam)
        loss_history.append(loss)
        grad_w, grad_b = lr_gradient(w, b, X, y, lam)
        grad_norm = math.sqrt(float(np.sum(grad_w**2) + grad_b**2))
        if grad_norm < tol:
            break

        v_w = beta * v_w + eta * grad_w
        v_b = beta * v_b + eta * grad_b
        w -= v_w
        b -= v_b

    return w, b, loss_history


def newton(X, y, lam, max_iter=100, tol=1e-12):
    """Execute Newton method with Armijo backtracking line search."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0
    loss_history = []

    for it in range(max_iter):
        loss = lr_objective(w, b, X, y, lam)
        loss_history.append(loss)

        grad_w, grad_b = lr_gradient(w, b, X, y, lam)
        g = np.append(grad_w, grad_b)
        if np.linalg.norm(g) < tol:
            break

        H = lr_hessian(w, b, X, y, lam)
        try:
            delta = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            H_reg = H + 1e-6 * np.eye(len(g))
            delta = np.linalg.solve(H_reg, -g)

        step = 1.0
        delta_w = delta[:-1]
        delta_b = float(delta[-1])
        c1 = 1e-4
        directional = float(np.dot(g, delta))

        while step > 1e-8:
            w_new = w + step * delta_w
            b_new = b + step * delta_b
            loss_new = lr_objective(w_new, b_new, X, y, lam)
            if loss_new <= loss + c1 * step * directional:
                break
            step *= 0.5

        w += step * delta_w
        b += step * delta_b

    return w, b, loss_history


def soft_threshold(v, alpha):
    """Apply coordinate soft thresholding operator for L1 proximal step."""
    v_arr = np.asarray(v, dtype=float)
    return np.sign(v_arr) * np.maximum(0.0, np.abs(v_arr) - alpha)


def ista(X, y, lam, L=None, max_iter=2000, tol=1e-10):
    """Fit L1 penalised logistic regression via proximal gradient ISTA."""
    n_samples, n_features = X.shape
    if L is None:
        XTX = np.dot(X.T, X)
        eig_vals = np.linalg.eigvalsh(XTX / (4.0 * float(n_samples)))
        L = float(eig_vals[-1]) + 1e-4

    step_size = 1.0 / L
    threshold = lam * step_size

    w = np.zeros(n_features, dtype=float)
    b = 0.0
    loss_history = []

    for it in range(max_iter):
        z = np.dot(X, w) + b
        p = sigmoid(z)
        diff = p - y
        n = float(len(y))

        grad_w_loss = np.dot(X.T, diff) / n
        grad_b_loss = float(np.sum(diff) / n)

        w_new = soft_threshold(w - step_size * grad_w_loss, threshold)
        b_new = b - step_size * grad_b_loss

        w_change = np.max(np.abs(w_new - w))
        w = w_new
        b = b_new

        loss = float(np.mean(softplus(z) - y * z) + lam * np.sum(np.abs(w)))
        loss_history.append(loss)

        if w_change < tol:
            break

    return w, b, loss_history


def kfold_stratified(y, k=5, rng=None):
    """Generate k stratified folds preserving class proportions."""
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    class_folds = {c: [] for c in classes}

    for c in classes:
        idx_c = np.where(y_arr == c)[0]
        if rng is not None:
            idx_c = rng.permutation(idx_c)
        folds_c = np.array_split(idx_c, k)
        class_folds[c] = folds_c

    for fold_i in range(k):
        val_idx_list = [class_folds[c][fold_i] for c in classes]
        val_idx = np.concatenate(val_idx_list)
        if rng is not None:
            val_idx = rng.permutation(val_idx)

        train_idx_list = []
        for c in classes:
            for other_i in range(k):
                if other_i != fold_i:
                    train_idx_list.append(class_folds[c][other_i])
        train_idx = np.concatenate(train_idx_list)
        if rng is not None:
            train_idx = rng.permutation(train_idx)

        yield train_idx, val_idx


def roc_curve(y_true, y_scores):
    """Compute false positive rate and true positive rate across thresholds."""
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_scores, dtype=float)

    desc_order = np.argsort(y_s)[::-1]
    y_t_sorted = y_t[desc_order]
    y_s_sorted = y_s[desc_order]

    distinct_indices = np.where(np.diff(y_s_sorted))[0]
    threshold_indices = np.r_[distinct_indices, len(y_s_sorted) - 1]

    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    tps = np.cumsum(y_t_sorted == 1)[threshold_indices]
    fps = np.cumsum(y_t_sorted == 0)[threshold_indices]

    tpr = tps / float(n_pos)
    fpr = fps / float(n_neg)

    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[y_s_sorted[0] + 1.0, y_s_sorted[threshold_indices]]
    return fpr, tpr, thresholds


def auc_ranksum(y_true, y_scores):
    """Compute area under ROC curve via Mann Whitney U rank sum statistic."""
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_scores, dtype=float)

    pos_idx = np.where(y_t == 1)[0]
    neg_idx = np.where(y_t == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = np.empty_like(y_s)
    sort_order = np.argsort(y_s)
    ranks[sort_order] = np.arange(1, len(y_s) + 1)

    rank_sum_pos = float(np.sum(ranks[pos_idx]))
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = float(u_stat / (n_pos * n_neg))
    return auc


def auc_trapezoid(fpr, tpr):
    """Integrate area under receiver operating characteristic via trapezoids."""
    auc = 0.0
    for i in range(1, len(fpr)):
        dx = fpr[i] - fpr[i - 1]
        avg_y = (tpr[i] + tpr[i - 1]) / 2.0
        auc += dx * avg_y
    return float(auc)


def cross_val_lambda(X, y, lambdas, k=5, penalty='l2', rng=None):
    """Select regularisation parameter lambda using stratified cross validation."""
    mean_aucs = []

    for lam in lambdas:
        fold_aucs = []
        for train_idx, val_idx in kfold_stratified(y, k=k, rng=rng):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            if penalty == 'l1':
                w_f, b_f, _ = ista(X_tr, y_tr, lam=lam, max_iter=1000)
            else:
                w_f, b_f, _ = gd(X_tr, y_tr, lam=lam, max_iter=1000)

            scores_vl = sigmoid(np.dot(X_vl, w_f) + b_f)
            val_auc = auc_ranksum(y_vl, scores_vl)
            fold_aucs.append(val_auc)

        mean_aucs.append(float(np.mean(fold_aucs)))

    best_idx = int(np.argmax(mean_aucs))
    best_lambda = float(lambdas[best_idx])
    return best_lambda, mean_aucs


def expected_cost(y_true, y_pred, r):
    """Calculate mean clinical misclassification cost given penalty ratio r."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    false_negatives = np.sum((y_t == 1) & (y_p == 0))
    false_positives = np.sum((y_t == 0) & (y_p == 1))

    total_cost = float(r * false_negatives + 1.0 * false_positives)
    return float(total_cost / len(y_t))


def softmax(Z):
    """Compute numerically stable row softmax across class dimensions."""
    Z_arr = np.asarray(Z, dtype=float)
    shift_z = Z_arr - np.max(Z_arr, axis=-1, keepdims=True)
    exp_z = np.exp(shift_z)
    probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    return probs


def softmax_regression(X, y, num_classes=10, lam=0.01, eta=0.1, max_iter=500, rng=None):
    """Train multinomial logistic regression with weight decay using batch updates."""
    n_samples, n_features = X.shape
    W = np.zeros((num_classes, n_features), dtype=float)
    b = np.zeros(num_classes, dtype=float)

    Y_onehot = np.zeros((n_samples, num_classes), dtype=float)
    for i in range(n_samples):
        Y_onehot[i, int(y[i])] = 1.0

    loss_history = []

    for it in range(max_iter):
        logits = np.dot(X, W.T) + b
        probs = softmax(logits)

        loss = float(-np.mean(np.sum(Y_onehot * np.log(probs + 1e-15), axis=1)) + 0.5 * lam * np.sum(W**2))
        loss_history.append(loss)

        error = probs - Y_onehot
        grad_W = np.dot(error.T, X) / float(n_samples) + lam * W
        grad_b = np.sum(error, axis=0) / float(n_samples)

        W -= eta * grad_W
        b -= eta * grad_b

    return W, b, loss_history


def hinge_objective(w, b, X, y, C):
    """Compute soft margin support vector machine primal hinge objective."""
    margin_violations = np.maximum(0.0, 1.0 - y * (np.dot(X, w) + b))
    obj = float(0.5 * np.sum(w**2) + C * np.sum(margin_violations))
    return obj


def pegasos(X, y, C, max_iter=10000, rng=None):
    """Train linear soft margin support vector machine via Pegasos subgradient descent."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    lam_svm = 1.0 / (float(C) * float(n_samples))
    obj_history = []

    if rng is None:
        rng = np.random.default_rng(2013)

    for t in range(1, max_iter + 1):
        idx = rng.integers(0, n_samples)
        eta_t = 1.0 / (lam_svm * float(t))

        margin_val = y[idx] * (np.dot(X[idx], w) + b)
        if margin_val < 1.0:
            w = (1.0 - eta_t * lam_svm) * w + eta_t * (y[idx] * X[idx])
            b = b + eta_t * float(y[idx])
        else:
            w = (1.0 - eta_t * lam_svm) * w

        if t % 100 == 0 or t == max_iter:
            obj = hinge_objective(w, b, X, y, C)
            obj_history.append(obj)

    return w, b, obj_history


def linear_kernel(X1, X2):
    """Compute dot product linear Gram matrix."""
    return np.dot(X1, X2.T)


def poly_kernel(X1, X2, degree=3, coef0=1.0):
    """Compute polynomial kernel Gram matrix with offset."""
    return (np.dot(X1, X2.T) + coef0) ** float(degree)


def rbf_kernel(X1, X2, gamma):
    """Compute radial basis function kernel Gram matrix."""
    X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
    X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
    dists_sq = X1_sq + X2_sq - 2.0 * np.dot(X1, X2.T)
    dists_sq = np.maximum(0.0, dists_sq)
    return np.exp(-float(gamma) * dists_sq)


def svm_dual_objective(alpha, K, y):
    """Evaluate dual quadratic objective of support vector machine."""
    ay = alpha * y
    quad_term = float(np.sum(ay[:, np.newaxis] * ay[np.newaxis, :] * K))
    return float(np.sum(alpha) - 0.5 * quad_term)


def smo_train(X, y, C, kernel_func, tol=1e-3, max_passes=20, max_iter=10000):
    """Train dual support vector machine via sequential minimal optimization."""
    n_samples, n_features = X.shape
    K = kernel_func(X, X)
    alpha = np.zeros(n_samples, dtype=float)
    b = 0.0

    errors = -y.astype(float).copy()
    passes = 0
    entire_set = True
    iterations = 0

    dual_history = []
    gap_history = []

    def get_error(idx):
        return errors[idx]

    def update_error(idx):
        score = np.sum(alpha * y * K[:, idx]) + b
        errors[idx] = score - float(y[idx])

    while passes < max_passes and iterations < max_iter:
        num_changed_alphas = 0

        if entire_set:
            candidate_indices = range(n_samples)
        else:
            candidate_indices = np.where((alpha > 1e-6) & (alpha < C - 1e-6))[0]

        for i1 in candidate_indices:
            iterations += 1
            E1 = get_error(i1)
            y1 = float(y[i1])

            r1 = E1 * y1
            if (r1 < -tol and alpha[i1] < C) or (r1 > tol and alpha[i1] > 0.0):
                non_bound = np.where((alpha > 1e-6) & (alpha < C - 1e-6))[0]
                i2 = -1

                if len(non_bound) > 1:
                    max_delta_e = 0.0
                    for j in non_bound:
                        if j == i1:
                            continue
                        delta_e = abs(E1 - get_error(j))
                        if delta_e > max_delta_e:
                            max_delta_e = delta_e
                            i2 = j

                if i2 == -1:
                    candidates = np.random.permutation(n_samples)
                    for j in candidates:
                        if j != i1:
                            i2 = j
                            break

                E2 = get_error(i2)
                y2 = float(y[i2])
                alpha1_old = alpha[i1]
                alpha2_old = alpha[i2]

                if y1 != y2:
                    L = max(0.0, alpha2_old - alpha1_old)
                    H = min(float(C), float(C) + alpha2_old - alpha1_old)
                else:
                    L = max(0.0, alpha1_old + alpha2_old - float(C))
                    H = min(float(C), alpha1_old + alpha2_old)

                if abs(L - H) < 1e-8:
                    continue

                eta = 2.0 * K[i1, i2] - K[i1, i1] - K[i2, i2]
                if eta < 0.0:
                    alpha2_new = alpha2_old - (y2 * (E1 - E2)) / eta
                    if alpha2_new > H:
                        alpha2_new = H
                    elif alpha2_new < L:
                        alpha2_new = L
                else:
                    continue

                if abs(alpha2_new - alpha2_old) < 1e-5:
                    continue

                alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)

                b1 = b - E1 - y1 * (alpha1_new - alpha1_old) * K[i1, i1] - y2 * (alpha2_new - alpha2_old) * K[i1, i2]
                b2 = b - E2 - y1 * (alpha1_new - alpha1_old) * K[i1, i2] - y2 * (alpha2_new - alpha2_old) * K[i2, i2]

                if 0.0 < alpha1_new < C:
                    b_new = b1
                elif 0.0 < alpha2_new < C:
                    b_new = b2
                else:
                    b_new = (b1 + b2) / 2.0

                alpha[i1] = alpha1_new
                alpha[i2] = alpha2_new
                b = b_new

                scores = np.dot(K, alpha * y) + b
                errors = scores - y.astype(float)
                num_changed_alphas += 1

        dual_val = svm_dual_objective(alpha, K, y)
        dual_history.append(dual_val)

        if entire_set:
            entire_set = False
        elif num_changed_alphas == 0:
            entire_set = True
            passes += 1

    return alpha, b, dual_history, gap_history


def kkt_violations(alpha, y, X, b, kernel_func, C, tol=1e-3):
    """Count number of samples that violate Karush Kuhn Tucker conditions."""
    K = kernel_func(X, X)
    scores = np.dot(K, alpha * y) + b
    violations = 0

    for i in range(len(y)):
        margin_val = float(y[i]) * scores[i]
        a_val = alpha[i]

        if a_val < 1e-5:
            if margin_val < 1.0 - tol:
                violations += 1
        elif a_val > C - 1e-5:
            if margin_val > 1.0 + tol:
                violations += 1
        else:
            if abs(margin_val - 1.0) > tol:
                violations += 1

    return violations


def duality_gap(alpha, w, b, X, y, C, kernel_func):
    """Compute difference between primal objective and dual objective."""
    primal_val = hinge_objective(w, b, X, y, C)
    K = kernel_func(X, X)
    dual_val = svm_dual_objective(alpha, K, y)
    gap = primal_val - dual_val
    return float(gap)


def grid_search_cv(X, y, C_values, gamma_values, k=5, rng=None):
    """Conduct cross validation grid search over penalty C and kernel width gamma."""
    best_score = -1.0
    best_params = (C_values[0], gamma_values[0])
    score_matrix = np.zeros((len(C_values), len(gamma_values)), dtype=float)

    for i, c_val in enumerate(C_values):
        for j, g_val in enumerate(gamma_values):
            fold_accs = []
            for train_idx, val_idx in kfold_stratified(y, k=k, rng=rng):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_vl, y_vl = X[val_idx], y[val_idx]

                def k_fn(x1, x2):
                    return rbf_kernel(x1, x2, gamma=g_val)

                alpha_m, b_m, _, _ = smo_train(X_tr, y_tr, C=c_val, kernel_func=k_fn, max_passes=5, max_iter=2000)
                K_vl = k_fn(X_vl, X_tr)
                scores_vl = np.dot(K_vl, alpha_m * y_tr) + b_m
                preds_vl = np.where(scores_vl >= 0.0, 1, -1)
                fold_accs.append(accuracy(y_vl, preds_vl))

            mean_acc = float(np.mean(fold_accs))
            score_matrix[i, j] = mean_acc
            if mean_acc > best_score:
                best_score = mean_acc
                best_params = (c_val, g_val)

    return best_params, score_matrix


def one_vs_rest(X_train, y_train, X_test, num_classes, C, kernel_func):
    """Train one vs rest multiclass support vector machine ensemble."""
    models = []
    n_train = len(y_train)

    for c in range(num_classes):
        y_binary = np.where(y_train == c, 1, -1)
        alpha, b, _, _ = smo_train(X_train, y_binary, C=C, kernel_func=kernel_func, max_passes=5, max_iter=2000)
        models.append((alpha, b, y_binary))

    n_test = len(X_test)
    scores_all = np.zeros((n_test, num_classes), dtype=float)

    for c in range(num_classes):
        alpha, b, y_bin = models[c]
        K_test_train = kernel_func(X_test, X_train)
        scores_all[:, c] = np.dot(K_test_train, alpha * y_bin) + b

    preds = np.argmax(scores_all, axis=1)
    return preds, models


def one_vs_one(X_train, y_train, X_test, num_classes, C, kernel_func):
    """Train one vs one multiclass support vector machine ensemble with voting."""
    models = {}
    pair_list = []

    for c1 in range(num_classes):
        for c2 in range(c1 + 1, num_classes):
            idx = np.where((y_train == c1) | (y_train == c2))[0]
            X_pair = X_train[idx]
            y_pair = np.where(y_train[idx] == c1, 1, -1)

            alpha, b, _, _ = smo_train(X_pair, y_pair, C=C, kernel_func=kernel_func, max_passes=5, max_iter=1500)
            models[(c1, c2)] = (alpha, b, X_pair, y_pair)
            pair_list.append((c1, c2))

    n_test = len(X_test)
    votes = np.zeros((n_test, num_classes), dtype=int)
    decision_sum = np.zeros((n_test, num_classes), dtype=float)

    for (c1, c2), (alpha, b, X_pair, y_pair) in models.items():
        K_test_pair = kernel_func(X_test, X_pair)
        scores = np.dot(K_test_pair, alpha * y_pair) + b

        c1_wins = scores >= 0.0
        votes[c1_wins, c1] += 1
        votes[~c1_wins, c2] += 1

        decision_sum[:, c1] += scores
        decision_sum[:, c2] -= scores

    preds = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        v_row = votes[i]
        max_vote = int(np.max(v_row))
        top_candidates = np.where(v_row == max_vote)[0]
        if len(top_candidates) == 1:
            preds[i] = top_candidates[0]
        else:
            tie_scores = decision_sum[i, top_candidates]
            best_idx = int(np.argmax(tie_scores))
            preds[i] = top_candidates[best_idx]

    return preds, models


def kmeans_pp_init(X, K, rng):
    """Initialize cluster centroids using probabilistically distributed distances."""
    n_samples, n_features = X.shape
    centroids = np.zeros((K, n_features), dtype=float)

    first_idx = rng.integers(0, n_samples)
    centroids[0] = X[first_idx]

    for k in range(1, K):
        dists_sq = np.min([np.sum((X - centroids[j]) ** 2, axis=1) for j in range(k)], axis=0)
        total_sq = float(np.sum(dists_sq))
        if total_sq == 0.0:
            probs = np.ones(n_samples) / float(n_samples)
        else:
            probs = dists_sq / total_sq
        next_idx = rng.choice(n_samples, p=probs)
        centroids[k] = X[next_idx]

    return centroids


def kmeans(X, K, init='kmeans++', max_iter=300, n_init=10, rng=None):
    """Cluster data points via Lloyd iteration with empty cluster relocation."""
    n_samples, n_features = X.shape
    if rng is None:
        rng = np.random.default_rng(2013)

    best_inertia = float('inf')
    best_centroids = None
    best_labels = None
    best_inertia_history = []

    for restart in range(n_init):
        if init == 'kmeans++':
            centroids = kmeans_pp_init(X, K, rng)
        else:
            init_idx = rng.choice(n_samples, size=K, replace=False)
            centroids = X[init_idx].copy()

        inertia_history = []
        labels = np.zeros(n_samples, dtype=int)

        for it in range(max_iter):
            dists_sq = np.zeros((n_samples, K), dtype=float)
            for k in range(K):
                dists_sq[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)

            labels = np.argmin(dists_sq, axis=1)
            inertia = float(np.sum(np.min(dists_sq, axis=1)))
            inertia_history.append(inertia)

            new_centroids = np.zeros_like(centroids)
            for k in range(K):
                cluster_pts = X[labels == k]
                if len(cluster_pts) > 0:
                    new_centroids[k] = np.mean(cluster_pts, axis=0)
                else:
                    furthest_idx = int(np.argmax(np.min(dists_sq, axis=1)))
                    new_centroids[k] = X[furthest_idx]

            shift = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids
            if shift < 1e-6:
                break

        final_inertia = inertia_history[-1]
        if final_inertia < best_inertia:
            best_inertia = final_inertia
            best_centroids = centroids
            best_labels = labels
            best_inertia_history = inertia_history

    return best_centroids, best_labels, best_inertia, best_inertia_history


def logsumexp(a, axis=None, keepdims=False):
    """Compute log of sum of exponentials stably across designated axis."""
    a_arr = np.asarray(a, dtype=float)
    a_max = np.max(a_arr, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a_arr - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def gmm_em(X, K, max_iter=200, tol=1e-6, reg=1e-6, rng=None):
    """Fit full covariance Gaussian mixture model via expectation maximisation."""
    n_samples, n_features = X.shape
    if rng is None:
        rng = np.random.default_rng(2013)

    init_idx = rng.choice(n_samples, size=K, replace=False)
    means = X[init_idx].copy()
    covs = np.array([np.eye(n_features) for _ in range(K)], dtype=float)
    weights = np.ones(K, dtype=float) / float(K)

    ll_history = []

    for it in range(max_iter):
        log_resp = np.zeros((n_samples, K), dtype=float)

        for k in range(K):
            cov_k = covs[k] + reg * np.eye(n_features)
            L = np.linalg.cholesky(cov_k)
            log_det = 2.0 * float(np.sum(np.log(np.diag(L))))

            diff = X - means[k]
            v = np.zeros_like(diff.T)
            for row_i in range(n_features):
                v[row_i] = (diff.T[row_i] - np.dot(L[row_i, :row_i], v[:row_i])) / L[row_i, row_i]

            quad_form = np.sum(v**2, axis=0)
            log_prob = -0.5 * (float(n_features) * math.log(2.0 * math.pi) + log_det + quad_form)
            log_resp[:, k] = math.log(weights[k] + 1e-15) + log_prob

        total_log_likelihood = logsumexp(log_resp, axis=1)
        current_ll = float(np.sum(total_log_likelihood))
        ll_history.append(current_ll)

        log_gamma = log_resp - total_log_likelihood[:, np.newaxis]
        gamma = np.exp(log_gamma)

        N_k = np.sum(gamma, axis=0)
        weights = N_k / float(n_samples)

        for k in range(K):
            means[k] = np.sum(gamma[:, k : k + 1] * X, axis=0) / N_k[k]
            diff = X - means[k]
            covs[k] = np.dot((gamma[:, k : k + 1] * diff).T, diff) / N_k[k] + reg * np.eye(n_features)

        if it > 0 and abs(current_ll - ll_history[-2]) < tol:
            break

    cluster_assignments = np.argmax(gamma, axis=1)
    return weights, means, covs, gamma, cluster_assignments, ll_history


def bic(log_likelihood, n_samples, n_features, K):
    """Compute Bayesian Information Criterion for mixture model fit."""
    cov_params = K * (n_features * (n_features + 1)) // 2
    mean_params = K * n_features
    weight_params = K - 1
    p = cov_params + mean_params + weight_params
    val = -2.0 * log_likelihood + float(p) * math.log(float(n_samples))
    return float(val)


def agglomerative(X, linkage='ward'):
    """Execute agglomerative hierarchical clustering with Lance Williams distances."""
    n_samples, n_features = X.shape

    X_sq = np.sum(X**2, axis=1)[:, np.newaxis]
    D_sq = X_sq + X_sq.T - 2.0 * np.dot(X, X.T)
    D_sq = np.maximum(0.0, D_sq)

    if linkage == 'ward':
        dist_matrix = D_sq.copy()
    else:
        dist_matrix = np.sqrt(D_sq)

    np.fill_diagonal(dist_matrix, float('inf'))

    cluster_sizes = {i: 1 for i in range(n_samples)}
    active_clusters = list(range(n_samples))
    linkage_rows = []

    for step in range(n_samples - 1):
        min_val = float('inf')
        best_i = -1
        best_j = -1

        for a_idx in range(len(active_clusters)):
            i_node = active_clusters[a_idx]
            for b_idx in range(a_idx + 1, len(active_clusters)):
                j_node = active_clusters[b_idx]
                d_val = dist_matrix[i_node, j_node]
                if d_val < min_val:
                    min_val = d_val
                    best_i = i_node
                    best_j = j_node

        new_cluster_id = n_samples + step
        size_i = cluster_sizes[best_i]
        size_j = cluster_sizes[best_j]
        new_size = size_i + size_j
        cluster_sizes[new_cluster_id] = new_size

        display_height = math.sqrt(min_val) if linkage == 'ward' else min_val
        linkage_rows.append([float(best_i), float(best_j), float(display_height), float(new_size)])

        n_active = max(new_cluster_id + 1, dist_matrix.shape[0])
        if n_active > dist_matrix.shape[0]:
            expanded = np.full((n_active + 50, n_active + 50), float('inf'), dtype=float)
            expanded[: dist_matrix.shape[0], : dist_matrix.shape[1]] = dist_matrix
            dist_matrix = expanded

        active_clusters.remove(best_i)
        active_clusters.remove(best_j)

        for k_node in active_clusters:
            size_k = cluster_sizes[k_node]
            d_ik = dist_matrix[best_i, k_node]
            d_jk = dist_matrix[best_j, k_node]

            if linkage == 'single':
                d_new = min(d_ik, d_jk)
            elif linkage == 'complete':
                d_new = max(d_ik, d_jk)
            elif linkage == 'average':
                d_new = (size_i * d_ik + size_j * d_jk) / float(new_size)
            elif linkage == 'ward':
                d_ij = min_val
                denom = float(size_i + size_j + size_k)
                d_new = ((size_i + size_k) * d_ik + (size_j + size_k) * d_jk - size_k * d_ij) / denom
            else:
                d_new = min(d_ik, d_jk)

            dist_matrix[new_cluster_id, k_node] = d_new
            dist_matrix[k_node, new_cluster_id] = d_new

        active_clusters.append(new_cluster_id)

    return np.array(linkage_rows, dtype=float)


def cut_tree(linkage_matrix, K, n_samples):
    """Cut hierarchical dendrogram into designated K discrete clusters."""
    parent = {}
    for i in range(2 * n_samples):
        parent[i] = i

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    n_merges = n_samples - K
    for step in range(n_merges):
        c1 = int(linkage_matrix[step, 0])
        c2 = int(linkage_matrix[step, 1])
        new_id = n_samples + step
        parent[find(c1)] = new_id
        parent[find(c2)] = new_id

    roots = [find(i) for i in range(n_samples)]
    unique_roots = list(np.unique(roots))
    root_to_label = {r: idx for idx, r in enumerate(unique_roots)}
    labels = np.array([root_to_label[r] for r in roots], dtype=int)
    return labels


def draw_dendrogram(linkage_matrix, labels=None, title=None, ax=None):
    """Render hierarchical tree lines using matplotlib coordinate drawing."""
    import matplotlib.pyplot as plt

    n_merges = len(linkage_matrix)
    n_samples = n_merges + 1

    def get_leaves(node):
        if node < n_samples:
            return [node]
        row_idx = node - n_samples
        left_child = int(linkage_matrix[row_idx, 0])
        right_child = int(linkage_matrix[row_idx, 1])
        return get_leaves(left_child) + get_leaves(right_child)

    root_id = 2 * n_samples - 2
    leaf_order = get_leaves(root_id)
    leaf_positions = {leaf: float(pos) for pos, leaf in enumerate(leaf_order)}

    node_x = leaf_positions.copy()
    node_y = {i: 0.0 for i in range(n_samples)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    for step, row in enumerate(linkage_matrix):
        c1 = int(row[0])
        c2 = int(row[1])
        height = float(row[2])
        new_node = n_samples + step

        x1, y1 = node_x[c1], node_y[c1]
        x2, y2 = node_x[c2], node_y[c2]
        x_mid = (x1 + x2) / 2.0

        ax.plot([x1, x1], [y1, height], color='black', linewidth=1.2)
        ax.plot([x2, x2], [y2, height], color='black', linewidth=1.2)
        ax.plot([x1, x2], [height, height], color='black', linewidth=1.2)

        node_x[new_node] = x_mid
        node_y[new_node] = height

    ax.set_xticks(range(len(leaf_order)))
    if labels is not None:
        ax.set_xticklabels([labels[i] for i in leaf_order], rotation=90, fontsize=8)
    else:
        ax.set_xticklabels(leaf_order, rotation=90, fontsize=8)

    ax.set_ylabel('Height')
    if title is not None:
        ax.set_title(title)
    return ax


def dbscan(X, eps, min_samples):
    """Execute density based spatial clustering with noise separation."""
    n_samples = X.shape[0]
    X_sq = np.sum(X**2, axis=1)[:, np.newaxis]
    dists = np.sqrt(np.maximum(0.0, X_sq + X_sq.T - 2.0 * np.dot(X, X.T)))

    labels = np.full(n_samples, -1, dtype=int)
    visited = np.zeros(n_samples, dtype=bool)
    current_cluster = 0

    neighbors_list = [np.where(dists[i] <= eps)[0] for i in range(n_samples)]

    for i in range(n_samples):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = neighbors_list[i]
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            labels[i] = current_cluster
            seed_set = deque([pt for pt in neighbors if pt != i])

            while len(seed_set) > 0:
                q = seed_set.popleft()
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = neighbors_list[q]
                    if len(q_neighbors) >= min_samples:
                        for neighbor_pt in q_neighbors:
                            if neighbor_pt not in seed_set:
                                seed_set.append(neighbor_pt)

                if labels[q] == -1:
                    labels[q] = current_cluster

            current_cluster += 1

    return labels


def k_distance(X, k):
    """Compute sorted distance to kth nearest neighbor for eps evaluation."""
    n_samples = X.shape[0]
    X_sq = np.sum(X**2, axis=1)[:, np.newaxis]
    dists = np.sqrt(np.maximum(0.0, X_sq + X_sq.T - 2.0 * np.dot(X, X.T)))

    k_dists = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        sorted_dists = np.sort(dists[i])
        k_dists[i] = sorted_dists[k]

    return np.sort(k_dists)


def spectral_clustering(X, K, gamma=1.0, n_neighbors=None, rng=None):
    """Partition manifold geometry through normalized Laplacian spectrum."""
    n_samples = X.shape[0]
    X_sq = np.sum(X**2, axis=1)[:, np.newaxis]
    dists_sq = np.maximum(0.0, X_sq + X_sq.T - 2.0 * np.dot(X, X.T))
    W = np.exp(-float(gamma) * dists_sq)
    np.fill_diagonal(W, 0.0)

    if n_neighbors is not None:
        mask = np.zeros_like(W, dtype=bool)
        for i in range(n_samples):
            top_k = np.argsort(dists_sq[i])[1 : n_neighbors + 1]
            mask[i, top_k] = True
        mask = mask | mask.T
        W = np.where(mask, W, 0.0)

    d = np.sum(W, axis=1)
    d_inv_sqrt = np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    L_sym = np.eye(n_samples) - np.dot(np.dot(D_inv_sqrt, W), D_inv_sqrt)
    eig_vals, eig_vecs = np.linalg.eigh(L_sym)

    U = eig_vecs[:, :K]
    row_norms = np.sqrt(np.sum(U**2, axis=1, keepdims=True))
    T = np.where(row_norms > 1e-12, U / row_norms, 0.0)

    _, labels, _, _ = kmeans(T, K=K, init='kmeans++', n_init=10, rng=rng)
    return labels, eig_vals[: min(15, len(eig_vals))]


def ari(labels_true, labels_pred):
    """Calculate Adjusted Rand Index comparing discrete clustering partitions."""
    y_t = np.asarray(labels_true)
    y_p = np.asarray(labels_pred)
    n = len(y_t)
    if n <= 1:
        return 1.0

    classes_t = np.unique(y_t)
    classes_p = np.unique(y_p)

    t_map = {c: i for i, c in enumerate(classes_t)}
    p_map = {c: i for i, c in enumerate(classes_p)}

    table = np.zeros((len(classes_t), len(classes_p)), dtype=float)
    for t_val, p_val in zip(y_t, y_p):
        table[t_map[t_val], p_map[p_val]] += 1.0

    def comb2(arr):
        return np.sum(arr * (arr - 1.0) / 2.0)

    sum_comb_table = comb2(table)
    a = np.sum(table, axis=1)
    b = np.sum(table, axis=0)

    sum_comb_a = comb2(a)
    sum_comb_b = comb2(b)
    total_comb = float(n * (n - 1) / 2.0)

    expected_index = (sum_comb_a * sum_comb_b) / total_comb
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected_index
    if abs(denom) < 1e-12:
        return 0.0

    return float((sum_comb_table - expected_index) / denom)


def nmi(labels_true, labels_pred):
    """Calculate Normalized Mutual Information comparing cluster assignments."""
    y_t = np.asarray(labels_true)
    y_p = np.asarray(labels_pred)
    n = float(len(y_t))
    if n <= 1.0:
        return 1.0

    classes_t = np.unique(y_t)
    classes_p = np.unique(y_p)

    t_map = {c: i for i, c in enumerate(classes_t)}
    p_map = {c: i for i, c in enumerate(classes_p)}

    table = np.zeros((len(classes_t), len(classes_p)), dtype=float)
    for t_val, p_val in zip(y_t, y_p):
        table[t_map[t_val], p_map[p_val]] += 1.0

    p_t = np.sum(table, axis=1) / n
    p_p = np.sum(table, axis=0) / n
    p_joint = table / n

    mi = 0.0
    for i in range(len(classes_t)):
        for j in range(len(classes_p)):
            if p_joint[i, j] > 1e-15:
                mi += p_joint[i, j] * math.log(p_joint[i, j] / (p_t[i] * p_p[j]))

    h_t = -float(np.sum([pt * math.log(pt) for pt in p_t if pt > 1e-15]))
    h_p = -float(np.sum([pp * math.log(pp) for pp in p_p if pp > 1e-15]))

    denom = math.sqrt(h_t * h_p)
    if denom < 1e-12:
        return 1.0 if h_t < 1e-12 and h_p < 1e-12 else 0.0

    return float(mi / denom)


def silhouette(X, labels):
    """Calculate silhouette score measuring intra cluster compactness versus separation."""
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    valid_labels = [l for l in unique_labels if l != -1]

    if len(valid_labels) <= 1:
        return 0.0

    X_sq = np.sum(X**2, axis=1)[:, np.newaxis]
    dists = np.sqrt(np.maximum(0.0, X_sq + X_sq.T - 2.0 * np.dot(X, X.T)))

    sil_values = []

    for i in range(n_samples):
        l_i = labels[i]
        if l_i == -1:
            continue

        same_cluster = np.where(labels == l_i)[0]
        if len(same_cluster) <= 1:
            sil_values.append(0.0)
            continue

        a_i = float(np.sum(dists[i, same_cluster]) / float(len(same_cluster) - 1))

        b_i = float('inf')
        for other_l in valid_labels:
            if other_l == l_i:
                continue
            other_pts = np.where(labels == other_l)[0]
            if len(other_pts) > 0:
                mean_dist_other = float(np.mean(dists[i, other_pts]))
                if mean_dist_other < b_i:
                    b_i = mean_dist_other

        max_ab = max(a_i, b_i)
        if max_ab > 0.0:
            sil_values.append((b_i - a_i) / max_ab)
        else:
            sil_values.append(0.0)

    return float(np.mean(sil_values)) if len(sil_values) > 0 else 0.0
