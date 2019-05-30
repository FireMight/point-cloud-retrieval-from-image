conda install matplotlib
echo "Installing torch-geometric..."
pip install --no-cache-dir torch-scatter
pip install --no-cache-dir torch-sparse
pip install --no-cache-dir torch-cluster
pip install torch-geometric
echo "Installing dependencies for NetVLAD..."
conda install faiss-gpu cudatoolkit=9.0 -c pytorch
pip install tensorboardX

echo "Fetching sumodules..."
git submodule init
git submodule update
