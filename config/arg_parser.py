import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', type=str, default='MC_triad')
    parser.add_argument('--DEVICE', type=str, default='cuda:0')
    parser.add_argument('--data_save_path', type=str, default=None)
    parser.add_argument('--log_save_path', type=str, default=None)
    parser.add_argument('--figure_save_path', type=str, default=None)
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--params_name', type=str, default='equipart')

    parser.add_argument('--CONTROLLER_LR', type=float, default=1e-1)
    parser.add_argument('--CONTROLLER_INPUT_SIZE', type=int, default=20)
    parser.add_argument('--CONTROLLER_TOP_SAMPLES_FRACTION', type=float, default=0.25)
    parser.add_argument('--CONTROLLER_QUANTILE_METHOD', type=str, default='linear')
    parser.add_argument('--EXPLORATION_ITERS', type=int, default=30)
    parser.add_argument('--NUM_TREES', type=int, default=100)
    parser.add_argument('--INTEGRATOR_METHOD', type=str, default='integration-based')
    # FEX optimizer settings
    parser.add_argument('--FEX_STAGE_OPEN_BOOL',type=bool,default = True)
    parser.add_argument('--FEX_LR', type=float, default=8e-3)
    parser.add_argument('--TRAIN_EPOCHS_FIRST', type=int, default=20)
    parser.add_argument('--TRAIN_EPOCHS_SECOND', type=int, default=2000)
    parser.add_argument('--TRAIN_GROUND_TRUTH',type=bool,default = False)
    parser.add_argument('--MULTI_FEX_OPEN',type=float,default = False)
    #FEX-DM settings
    parser.add_argument('--SECOND_STAGE_OPEN_BOOL',type=bool,default = True)
    parser.add_argument('--DIFF_SCALE',type=float,default = 100)
    parser.add_argument('--ODESLOVER_TIME_STEPS',type=int,default = 2000)
    parser.add_argument('--SHORT_SIZE',type=int,default = 2048)
    parser.add_argument('--NN_SOLVER_LR',type=float,default = 0.01)
    parser.add_argument('--NN_SOLVER_EPOCHS',type=int,default = 2000)
    return parser

