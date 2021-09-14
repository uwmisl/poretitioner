#from .poretitioner import main


from . import poretitioner as porty
# from dask.distributed import Client, LocalCluster
# import dask.multiprocessing as dask_mp


def main():
    porty.main()

# if __name__ == "poretitioner.__main__":
#     cluster = LocalCluster()
#     # Trying single-process thread-based processing 
#     client = Client(cluster, processes=False)
#     dask_mp.multiprocessing.freeze_support()
#     main()
