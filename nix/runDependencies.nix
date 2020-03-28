###########################################################################################
#                       
# runDependencies.nix 
#                      
###########################################################################################
#
# This expression hosts the project's runtime dependencies (basically, anything you 
# need to explicitly import in python)
#
# It takes in a pythonPackage, which is intended to be provided by `python.withPackages`.
#
###########################################################################################


pythonPackages: [ 
        # Numerical computation library
        pythonPackages.numpy
        # Data manipulation and analysis
        pythonPackages.pandas
        # Hierarchical Data Format utilities 
        pythonPackages.h5py
        # Parallel computing library
        pythonPackages.dask
        # Charts and plotting library
        pythonPackages.matplotlib
        # Data visualization 
        pythonPackages.seaborn
        # Interactive computing
        pythonPackages.notebook
        # For interactive builds
        pythonPackages.jupyter
]