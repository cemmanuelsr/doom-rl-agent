# OpenSIM-RL Project

A reinforcement learning agent developed on the [OpenSIM-RL environment](http://osim-rl.kidzinski.com/docs/quickstart/).

## Setup

Primeiramente será preciso instalar Anaconda localmente para criarmos um ambiente.

1. Download um instalador:
    - [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers)
    - [Anaconda](https://www.anaconda.com/download/)
2. Rode, no diretório do instalador `bash {installer_filename}`
3. Siga os passos até finalizar e pronto!

Qualquer problema com a instalação, consultar [documentação oficial do Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

Crie e ative um virtual environment para executar o projeto:

1. `conda create -n {virtual_env_name} -c kidzik opensim python=3.6.1`
    - Se não conseguir criar o ambiente dessa forma, remova a versão do python e instale depois.
2. `source activate {virtual_env_name}`
3. Instale os pacotes necessários:
    - `conda install -c conda-forge lapack git`
    - `pip install git+https://github.com/stanfordnmbl/osim-rl.git`
4. Se você não conseguiu escolher a versão do python inicialmente, faça isso agora:
    - `conda install python=3.6`
    - **PRECISA SER PYTHON 3.6.X**

Novamente, qualquer problema, consulte a [documentação oficial do OpenSIM](http://osim-rl.kidzinski.com/docs/quickstart/).

Teste o código `python -c "import opensim"`, se não retornar nenhum problema o setup foi bem sucedido.
