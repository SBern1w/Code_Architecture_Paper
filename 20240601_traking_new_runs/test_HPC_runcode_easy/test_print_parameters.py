import argparse
from mpi4py import MPI

# MPI inizialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()      # Get the rank of the current process within the communicator
size_comm = comm.Get_size()      # Get the total number of processes in the communicator

def main(n_inputs, i_loss, imbalance, folder_path):
    print(f"n_inputs: {n_inputs}")
    print(f"i_loss: {i_loss}")
    print(f"imbalance: {imbalance}")
    print(f"folder_path: {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the hyperparameters.')
    parser.add_argument('--n_inputs', required=True, help='n_inputs')
    parser.add_argument('--i_loss', required=True, help='i_loss')
    parser.add_argument('--imbalance', required=True, help='imbalance')
    parser.add_argument('--folder_path', required=True, help='folder_path')
    args = parser.parse_args()

    if rank == 0:
        main(args.n_inputs, args.i_loss, args.imbalance, args.folder_path)