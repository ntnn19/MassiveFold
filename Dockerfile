ARG CUDA_VERSION=11.8.0
ARG COLABFOLD_VERSION=1.5.5
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04

# Install prerequisites
RUN apt-get update && apt-get install -y \
    wget \
    parallel \
    git \
    build-essential \
    libmunge-dev \
    munge \
    libssl-dev \
    wget \
    tar \
    bzip2 \
    python3 \
    fakeroot \
    devscripts \
    equivs \
    cuda-nvcc-$(echo $CUDA_VERSION | cut -d'.' -f1,2 | tr '.' '-') \
    --no-install-recommends --no-install-suggests && \
    rm -rf /var/lib/apt/lists/*

# Install slurm
RUN wget https://download.schedmd.com/slurm/slurm-23.11.1.tar.bz2 && tar xf slurm-23.11.1.tar.bz2 && cd slurm-23.11.1 && ./configure --prefix=/usr/local --with-munge && make -j 4 && make install
# Install Miniconda
RUN apt-get update && apt-get install -y wget parallel cuda-nvcc-$(echo $CUDA_VERSION | cut -d'.' -f1,2 | tr '.' '-') --no-install-recommends --no-install-suggests && rm -rf /var/lib/apt/lists/* && \
    wget -qnc https://github.com/conda-forge/miniforge/releases/download/24.7.1-0/Mambaforge-24.7.1-0-Linux-x86_64.sh && \
    bash Mambaforge-24.7.1-0-Linux-x86_64.sh -bfp /usr/local && \
    conda config --set auto_update_conda false && \
    rm -f Mambaforge-24.7.1-0-Linux-x86_64.sh
WORKDIR /app

# Copy project files into the container
COPY install.sh /usr/local/bin/install.sh
COPY environment.yml /app/environment.yml
COPY mf-alphafold3.yml /app/mf-alphafold3.yml
COPY mf_colabfold.yml /app/mf_colabfold.yml
COPY examples /app/examples
COPY imgs /app/imgs
COPY massivefold /app/massivefold
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE
COPY massivefold_runs /app/massivefold_runs

# Make install.sh executable
RUN chmod +x /usr/local/bin/install.sh
# Set the working directory

# Run install.sh in --only-envs mode
RUN /usr/local/bin/install.sh --only-envs
COPY model_config.py /usr/local/envs/mf-alphafold3/lib/python3.12/site-packages/alphafold3/model/model_config.py 
COPY massivefold/plots /usr/local/envs/massivefold/bin/plots
COPY massivefold/massivefold_plots.py /usr/local/envs/massivefold/bin/
RUN chmod +x /usr/local/envs/massivefold/bin/massivefold_plots.py
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV TF_FORCE_UNIFIED_MEMORY=true
ENV XLA_CLIENT_MEM_FRACTION=3.2

