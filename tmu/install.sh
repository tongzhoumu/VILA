mamba create -n vila python=3.10 -y
mamba activate vila

pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install -e .
pip install -e ".[train]"

pip install git+https://github.com/huggingface/transformers@v4.36.2
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/

pip install gdown flask

