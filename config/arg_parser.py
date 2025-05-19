import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DEVICE', type=str, default='cpu')
    parser.add_argument('--data_save_path', type=str, default='Example/MC_triad/results/simulation_results.npz')
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--params_name', type=str, default='equipart')
    parser.add_argument('--figure_save_path', type=str, default='Example/MC_triad/results')
    return parser
