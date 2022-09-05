import numpy as np

def train(x_train, y_train, x_val, y_val, epochs, alpha, norm_f, denorm_f, extra_f, winit_f, hypo_f, diff_f, loss_f, delta_f, update_f, batch_f):
    
    loss_train, loss_val, dw_list = np.zeros(epochs), np.zeros(epochs), np.zeros(epochs)

    x_min, x_max = x_train.min(axis=0), x_train.max(axis=0)
    y_min, y_max = y_train.min(axis=0), y_train.max(axis=0)

    x_tn, y_tn = norm_f(x_train, x_min, x_max), norm_f(y_train, y_min, y_max)
    x_vn, y_vn = norm_f(x_val, x_min, x_max), norm_f(y_val, y_min, y_max)

    x_tn = extra_f(x_tn)
    x_vn = extra_f(x_vn)

    w = winit_f(x_tn.shape[1])

    for i in range(epochs):
        x_tb, y_tb = batch_f(x_tn, y_tn)
        x_vb, y_vb = batch_f(x_vn, y_vn)

        diff_tb = diff_f(y_tb, hypo_f(x_tb, w))
        diff_vb = diff_f(y_vb, hypo_f(x_vb, w))

        loss_train[i] = loss_f(x_tb, y_tb, w, diff_tb)
        loss_val[i] = loss_f(x_vb, y_vb, w, diff_vb)

        dw = delta_f(x_tb, y_tb, w, diff_tb)
        dw_list[i] = dw[0]
        w = update_f(w, dw, alpha)
        

    def predict(x):
        x_n = norm_f(x, x_min, x_max)
        x_n = extra_f(x_n)
        y_n = hypo_f(x_n, w)
        y = denorm_f(y_n, y_min, y_max)
        return y

    return w, predict, loss_train, loss_val, dw_list

