echo "Installing matplotlib..."
conda install matplotlib
echo "Installing torch-geometric..."
pip install --no-cache-dir torch-scatter
pip install --no-cache-dir torch-sparse
pip install --no-cache-dir torch-cluster
pip install torch-geometric

#looks like we won't need these dependencies for now; might add them later
#echo "Installing dependencies for NetVLAD..."
#conda install faiss-gpu cudatoolkit=9.0 -c pytorch
#pip install tensorboardX

echo "Fetching sumodules..."
git submodule init
git submodule update

#create init file for NetVLAD
echo "Creating an __init__.py file for NetVLAD"
touch pytorch-NetVlad/__init__.py
