import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

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

def get_res(logs, dataset="Test"):
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
                            Result[idx][metric] = (Result[last_value_idx][metric] * (epoch - idx) / (epoch - last_value_idx)
                                                   + Result[epoch][metric] * (idx - last_value_idx) / (epoch - last_value_idx))

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
        loss.append({'test_loss' : loss_t,
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
    if key == 'test_loss':
        plt.ylabel(r'Test loss', fontsize=30)
    if key == 'test_acc':
        plt.ylabel(r'Accuracy', fontsize=30)

    plt.xlim(0, xlim)

    plt.grid(color='#6d7175', linestyle='--', linewidth=0.7, alpha=0.9)
    # plt.savefig('graphics/' + key + '_' + str(methods[0]['compression']) + '_time.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('graphics/' + key + '_' + str(met) + '_' + str(dataset) + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def do_magic(graphs, dataset='homo', smooth=True):
    logs = unpack(graphs)
    res = get_res(logs, dataset="Test")
    acc, f1, test_loss = get_metrics(res)
    fig = make_plots_epochs_time(methods=acc, xlim=1500, key='test_acc', met='loss', dataset=dataset, smooth=smooth)
    res = get_res(logs, dataset="Valid")
    acc, f1, test_loss = get_metrics(res)
    fig = make_plots_epochs_time(methods=test_loss, xlim=1500, key='test_loss', met='loss', dataset=dataset, smooth=smooth)

# ================

graphs = {
    r'FedAvg' : './logs/fedavg_ef_300.txt',
    r'Scaffold' : './logs/scaffold_300.txt',
    #r'PPBCEF $\theta$=0.1' : './outputs/2025-12-14/20-21-07/output.txt',
    r'PPBCEF $\theta$=0.15' : './logs/ppbc_ef_100_t0.15.txt',
    #r'PPBCEF $\theta$=0.2' : './logs/ppbc_ef_76.txt',
    r'PPBCEF $\theta$=0.15' : './logs/ppbc_ef_100_t0.15.txt',
    }

"""r'FedAvg 3it' : './logs/homo/fedavg/200_3it.txt',
    r'PPBCEF 3it $\theta$=0.1' : './logs/homo/ppbc/poc/50_t0.1_3it.txt',
    r'PPBCEF 3it $\theta$=0.15' : './logs/homo/ppbc/poc/50_t0.15_3it.txt',
    r'PPBCEF 3it $\theta$=0.2' : './logs/homo/ppbc/poc/50_t0.2_3it.txt',
    """

graphs = {
    #r'FedAvg' : './logs/fedavg_ef_300.txt',
    'Scaffold' : './logs/EF21/pathology/top1/scaffold/loss_top1_lr5e-3.txt',
}

do_magic(graphs, dataset='pathology', smooth=True)