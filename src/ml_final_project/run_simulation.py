import json
import os
import argparse
import pandas as pd
from .utils.simulators.dataset_simulator import DatasetSimulator
from .utils.data.dataset2xy import dataset2Xy
from .utils.data.load import load_dataset
from .utils.runners.runners_factory import get_runner
from .consts import scan_spaces, network_conf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset-csv-path', '-d', dest='csv_path', action='store', required=True, help='path to a csv file containing the dataset to simulate on')
parser.add_argument('--output-csv-path', '-o', dest='res_path', action='store', required=True, help='path to a csv file to write the results to')
parser.add_argument('--network-conf', '-n', dest='network_conf', action='store', required=True, help='network-conf')

args = parser.parse_args()

if __name__ == '__main__':
    '''The script to run a full simulation.'''

    for k, v in json.loads(args.network_conf).items():
        network_conf[k] = v

    dataset = load_dataset(args.csv_path)
    X, y, output_dim = dataset2Xy(dataset)
    ds = []

    for optimizer in ['momentum', 'lookahead', 'improved_lookahead']:
        simulator = DatasetSimulator(X, y, output_dim, get_runner(optimizer), scan_spaces[optimizer])
        ds.append(simulator.evaluate().assign(algname=optimizer).rename(columns={'algname': 'Algorithm Name'}))

    res_df = pd.concat(ds)
    res_df['Dataset Name'] = os.path.split(args.csv_path)[1]
    res_df.to_csv(args.res_path)
