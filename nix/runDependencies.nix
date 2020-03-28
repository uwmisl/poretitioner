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

{ python } : 
with python.pkgs; [ 
    # Numerical computation library
    numpy
    # Data manipulation and analysis
    pandas
    # Hierarchical Data Format utilities 
    h5py
    # Parallel computing library
    dask
    # Charts and plotting library
    matplotlib
    # Data visualization 
    seaborn
    # Interactive computing
    notebook
    # For interactive builds
    jupyter
]