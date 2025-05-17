# Docker container for efficient iterative testing
# - build with
#   $ docker buildx build -t spacec_test_gpu -f Docker/spacec_test_gpu.dockerfile .
# - run with:
#   $ docker run --gpus all spacec_test_gpu
#   or if iteratively updating tests:
#   $ docker run --gpus all -v .:/app/spacec spacec_test_gpu
FROM spacec_test_cpu

# GPU stuff
RUN mamba install -n spacec -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0.77 -y
RUN mamba run -n spacec pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN mamba run -n spacec pip install torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 torch-spline-conv==1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# Clean up
RUN mamba clean --all -f -y && \
    rm -rf /root/.cache/pip

# Default command
CMD ["bash", "-c", "source /miniforge/etc/profile.d/conda.sh && conda activate spacec && pytest -m 'gpu and not skip and not slow'"]
