import argparse

def get_params():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/medhnh/workhorse3/medi_dataset/mini_data",
        help="audio directory",
    )
    
    return parser.parse_args()
