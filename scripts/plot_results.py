import tqdm
import seaborn as sns
sns.set_style('white')
from parasol.util import json
from path import Path
import click
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

gfile = tf.gfile

def flatten_dict(params):
    flattened = {}
    for k, v in params.items():
        if isinstance(v, dict):
            for k_, v_ in flatten_dict(v).items():
                flattened["%s.%s" % (k, k_)] = v_
        else:
            flattened[k] = v
    return flattened

def find_files(path, name):
    if gfile.IsDirectory(path):
        files = gfile.ListDirectory(path)
        for p in files:
            if p == name:
                yield path / p
                return
        for p in files:
            yield from find_files(path / p, name)

def load_experiments(path, experiment_type='solar'):
    for experiment in find_files(path, 'params.json'):
        with tf.gfile.GFile(experiment, 'r') as fp:
            try:
                params = json.load(fp)
            except:
                print("Found malformed experiment:", experiment)
                continue
            if 'experiment_type' not in params:
                print("Found malformed experiment:", experiment)
                continue
            if params['experiment_type'] != experiment_type:
                continue
            if gfile.Exists(experiment.parent / "results.csv"):
                yield experiment, flatten_dict(params)

@click.command()
@click.argument('path')
@click.option('--x_axis', default='episode_number')
@click.option('--y_axis', default='total_cost')
def plot_results(path, x_axis='episode_number', y_axis='total_cost',
                 aggregateby='seed', groupby=None):
    path = Path(path)
    data = pd.DataFrame()
    param_columns = set()
    for experiment, params in load_experiments(path):
        print("Loading experiment:", experiment)
        results_path = experiment.parent / 'results.csv'
        with gfile.GFile(results_path, 'r') as fp:
            results = pd.read_csv(fp)
        if results.dtypes['total_cost'] == object:
            print("Bad results:", experiment)
            continue
        for k, v in params.items():
            param_columns.add(k)
            results[k] = v
        data = pd.concat([data, results], sort=False)
    groupable_columns = (set([c for c in data.columns if len(pd.unique(data[c]))
                             > 1]) & param_columns - {'experiment_name',
                                                      aggregateby} | {x_axis})
    groups = data.groupby(list(groupable_columns))
    fig, ax = plt.subplots()
    plot_groups = list(groupable_columns - {x_axis})
    colors = sns.color_palette()
    if len(plot_groups) > 0:
        results = groups[y_axis].describe()[['mean', 'std']].reset_index().set_index('episode_number').groupby(
                                                plot_groups
                                            )
        for i, (group, data) in enumerate(results):
            data = data[['mean', 'std']].rolling(10).mean().dropna(how='all').fillna(0)
            ax.plot(data.index, data['mean'], label=str(group), color=colors[i],
                    alpha=0.8)
            ax.fill_between(data.index, data['mean'] - data['std'], data['mean'] +
                            data['std'], color=colors[i], alpha=0.3)
    else:
        results = groups[y_axis].describe()[['mean', 'std']].reset_index().set_index('episode_number')
        data = results[['mean', 'std']].rolling(10).mean().dropna(how='all').fillna(0)
        ax.plot(data.index, data['mean'], color=colors[0],
                alpha=0.8)
        ax.fill_between(data.index, data['mean'] - data['std'], data['mean'] +
                        data['std'], color=colors[0], alpha=0.3)
    ax.legend(loc='best', title=','.join(plot_groups))
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    plt.show()


if __name__ == "__main__":
    plot_results()
