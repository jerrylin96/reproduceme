*environment created on 3/14*

commands to run:
conda create -n pritch python=3.11.8
conda config --env --add channels conda-forge
conda install numpy
conda install pandas
conda install -c conda-forge matplotlib
conda install -c conda-forge xarray dask netCDF4 bottleneck
pip install ipython
pip install jupyterlab
python3 -m pip install tensorflow[and-cuda]
pip install --upgrade keras
pip install tensorflow-addons
pip install qhoptim
pip install keras-tuner --upgrade
