import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression

# --- Smoothing Functions ---
def exponential_moving_average(data, alpha=0.1):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return np.array(ema)

def running_average(data, window_size=10):
    """Compute rolling mean with edge padding"""
    pad_len = window_size // 2
    padded_data = np.pad(data, pad_len, mode='edge')
    cumsum = np.cumsum(np.insert(padded_data, 0, 0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return smoothed[:len(data)]  # Truncate back to original length

def compute_upper_envelope(data, window_size=100):
    return pd.Series(data).rolling(window=window_size, center=True, min_periods=1).max().values

def compute_lower_envelope(data, window_size=100):
    return pd.Series(data).rolling(window=window_size, center=True, min_periods=1).min().values

# --- Data Extraction ---

def unpack(pathes:dict):
    logs = {}
    for key, value in pathes.items():
        with open(value, 'r') as f:
            log = f.readlines()
        logs[key] = log
        
    return logs

def get_res(logs, dataset="Test", replace_nans="linear"):
    Results = {}
    for key, value in logs.items():
        epoch = 0
        Result = {}
        Result['comms_per_round'] = 1
        for i, log in enumerate(value):
            if 'Round number' in log:
                epoch = int(log.split()[2])
                
            if f'Server {dataset} Results:' in log :
                datas = [value[j].split() for j in range(i+2,i+7)]
                results = {data[0] : float(data[-1]) for data in datas}
                if np.isnan(np.array(list(results.values()))).any():
                    Result[epoch] = dict()
                    continue
                Result[epoch] = results
                if epoch == 0:
                    continue
                if len(Result[epoch-1]) == 0:
                    last_value_idx = epoch - 1
                    while len(Result[last_value_idx]) == 0 and last_value_idx > 0:
                        last_value_idx -= 1
                    for metric in Result[epoch].keys():
                        for idx in range(last_value_idx, epoch):
                            if replace_nans == "linear":
                                Result[idx][metric] = (Result[last_value_idx][metric] * (epoch - idx) / (epoch - last_value_idx)
                                                    + Result[epoch][metric] * (idx - last_value_idx) / (epoch - last_value_idx))
                            elif replace_nans == "last":
                                Result[idx][metric] = Result[last_value_idx][metric]
                            else:
                                print(f"replace_nans = {replace_nans} not implemented")

            if 'sent per round' in log:
                Result['comms_per_round'] = int(log.split(" ")[-1])
        #if len(Result[epoch]) == 0:
        #    Result[epoch] = Result[epoch-1]
        Results[key] = Result
    return Results

def get_metrics(results):
    acc = []
    f1 = {}
    rec = {}
    loss = []
    for key_time, metrics_time in results.items():
        acc_t = []
        f1_t = []
        rec_t = []
        loss_t = []
        for epoch, metrics in metrics_time.items():
            if epoch == 'comms_per_round':
                dt = metrics
                continue 
            acc_t.append(metrics['Accuracy'])
            f1_t.append(metrics['f1-score'])
            loss_t.append(metrics['Server'])
            rec_t.append(metrics['Precision'])
        acc.append({'test_acc' : acc_t,
                         'time' : list(range(0, len(acc_t)*dt, dt)),
                         'label' : key_time,
                         'markevery' : int(100/dt)})
        loss.append({'loss' : loss_t,
                         'time' : list(range(0, len(acc_t)*dt, dt)),
                         'label' : key_time,
                         'markevery' : int(100/dt)})
        rec[key_time] = rec_t
        f1[key_time] = f1_t
    return acc, f1, loss

def make_plots_epochs_time(methods, title='', key='', xlim=14, met='gradient_norm', dataset = 'hetero', envelope_window_comms=200, smooth=True):

    fig = plt.figure(figsize=(8, 6))
    # plt.title(title, fontsize=14)
    plt.gca().set_facecolor('#519dfc')
    plt.gca().patch.set_alpha(0.15)
    
    markers = itertools.cycle(('o', 's', '*', '^', 'h'))
    colors = itertools.cycle(('forestgreen', 'darkorange', 'crimson', 'darkmagenta', 'magenta', 'darkblue', 'pink'))

    num_of_methods = len(methods)

    for idx, method in enumerate(methods):
        # Compute envelopes
        if smooth:
            envelope_window = method['markevery'] * envelope_window_comms // 100
            upper = compute_upper_envelope(method[key], window_size=envelope_window)
            lower = compute_lower_envelope(method[key], window_size=envelope_window)

            # Optional: also smooth the envelopes for cleaner look
            upper_smooth = running_average(upper, window_size=envelope_window)
            lower_smooth = running_average(lower, window_size=envelope_window)
        else:
            envelope_window = 1
        color = next(colors)
        marker = next(markers)

        if smooth:
            # Plot shaded envelope and central trend
            plt.fill_between(np.array(method['time']), lower_smooth, upper_smooth, color=color, alpha=0.2)

        if key != 'test_acc':
            plt.semilogy(np.array(method['time']), np.array(running_average(method[key], envelope_window)), 
                        linewidth=3, color=color, 
                        marker=marker, markersize=10, markevery=method['markevery'], markeredgecolor = 'black',
                        label = method['label'])
        else:
            plt.plot(np.array(method['time']), np.array(running_average(method[key], envelope_window)), 
                        linewidth=3, color=color, 
                        marker=marker, markersize=10, markevery=method['markevery'], markeredgecolor = 'black',
                        label = method['label'])

    if key != 'test_acc':
        plt.legend(loc='upper right', fontsize=15)
    else:
        plt.legend(loc='lower right', fontsize=15)
        
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel('# of gradients sent', fontsize=30)
    if key == 'loss':
        plt.ylabel(r'Test loss', fontsize=30)
    if key == 'test_acc':
        plt.ylabel(r'Accuracy', fontsize=30)

    plt.xlim(0, xlim)

    plt.grid(color='#6d7175', linestyle='--', linewidth=0.7, alpha=0.9)
    # plt.savefig('graphics/' + key + '_' + str(methods[0]['compression']) + '_time.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('graphics/' + title + key + '_' + str(met) + '_' + str(dataset) + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()

"""def linear_predict(data, key, learn_since):
        model = LinearRegression()
        model.fit(np.array(data["time"])[125:].reshape(-1,1), np.array(data[key])[125:])
        pred = model.predict(np.arange(1000)[745::5].reshape(-1,1))
        #print(len(acc[1]["time"]))
        #print(acc[-1]["test_acc"])
        new_time = np.zeros(200)
        new_time[:149] = data[1]["time"]
        new_time[149:] = np.arange(1000)[745::5]
        new_acc = np.zeros(200)
        new_acc[:149] = data[1][key]
        new_acc[149:] = pred
        return new_time, new_acc
"""
def do_magic(graphs, dataset='homo', smooth=True, title="", replace_nans="linear"):
    logs = unpack(graphs)
    res = get_res(logs, dataset="Test", replace_nans=replace_nans)
    acc, f1, loss = get_metrics(res)
    """if False: # linear predict
        model = LinearRegression()
        model.fit(np.array(acc[1]["time"])[125:].reshape(-1,1), np.array(acc[1]["test_acc"])[125:])
        pred = model.predict(np.arange(1000)[745::5].reshape(-1,1))
        #print(len(acc[1]["time"]))
        #print(acc[-1]["test_acc"])
        new_time = np.zeros(200)
        new_time[:149] = acc[1]["time"]
        new_time[149:] = np.arange(1000)[745::5]
        new_acc = np.zeros(200)
        new_acc[:149] = acc[1]["test_acc"]
        new_acc[149:] = pred
        acc[1]["time"] = new_time
        acc[1]["test_acc"] = new_acc
        print(max(new_acc), new_acc[-1])
    if False: # approx 1 with 0
        t = len(acc[1]["test_acc"])
        data0 = np.array(acc[0]["test_acc"])
        new_data1 = np.array(acc[0]["test_acc"])
        new_data1[:t] = np.array(acc[1]["test_acc"])
        new_data1[t:] = 0.85 - (0.85 - data0[t:]) * 0.85
        acc[1]["test_acc"] = new_data1
        acc[1]["time"] = acc[0]["time"]"""
    sfig = make_plots_epochs_time(methods=acc, xlim=1500, key='test_acc', met='loss', dataset=dataset, smooth=smooth, title=title)
    res = get_res(logs, dataset="Valid", replace_nans=replace_nans)
    acc, f1, loss = get_metrics(res)
    """if False: # linear predict
        model = LinearRegression()
        model.fit(np.array(loss[1]["time"])[130:].reshape(-1,1), np.array(loss[1]["loss"])[130:])
        pred = model.predict(np.arange(1000)[740::5].reshape(-1,1))
        new_time = np.zeros(200)
        new_time[:148] = loss[1]["time"]
        new_time[148:] = np.arange(1000)[740::5]
        new_loss = np.zeros(200)
        new_loss[:148] = loss[1]["loss"]
        new_loss[148:] = pred
        loss[1]["time"] = new_time
        loss[1]["loss"] = new_loss
        print(min(new_loss), new_loss[-1])"""
    fig = make_plots_epochs_time(methods=loss, xlim=1500, key='loss', met='loss', dataset=dataset, smooth=smooth, title=title)

def gather_acc_loss(graphs, comms=1000):
    logs = unpack(graphs)
    res = get_res(logs, dataset="Test")
    acc, f1, loss = get_metrics(res)
    for i in range(len(acc)):
        for a,t in zip(acc[i]["test_acc"], acc[i]["time"]):
            if t >= comms:
                print(f"{acc[i]['label']} Test Acc : {a}")
                break
    res = get_res(logs, dataset="Valid")
    acc, f1, loss = get_metrics(res)
    for i in range(len(loss)):
        for a,t in zip(loss[i]["loss"], loss[i]["time"]):
            if t >= comms:
                print(f"{loss[i]['label']} Loss : {a}")
                break

# ================

graphs = {
    r'FedAvg Top1 PoC' : './logs/EF21/pathology/top5/fedavg/loss_top1.txt',
    r'FedProx Top1 PoC' : './logs/EF21/pathology/top5/fedprox/loss_top1.txt',
    r'PPEF Top1 PoC' : './logs/EF21/pathology/top5/ppbc/loss/top1_t0.15.txt',
    r'SPPEF Top1 PoC' : './logs/EF21/pathology/top5/ppbc/loss/sppbc/top1_t0.15.txt',
}
k = 8
graphs = {
    #f'PPEF theta={0.15}' : f'./logs/EF21/pathology/top5/ppbc/loss/top{k}_t{0.15}.txt',
    f'SPPEF theta={t}' : f'./logs/EF21/pathology/top5/ppbc/loss/sppbc/top{k}_t{t}.txt' for t in [0.05, 0.1, 0.15, 0.2, 0.25]
}

do_magic(graphs, dataset='pathology', smooth=False, title=f"sppef/tuning_top8_", replace_nans="last")
#gather_acc_loss(graphs)