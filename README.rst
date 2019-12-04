.. highlight:: bash

For a more up-to-date repo on the same subject please look at `AD-DL repo <https://github.com/aramis-lab/AD-DL>`_.

Before launching this code or one of your own you should create a conda env

Conda environment
-----------------

You can install miniconda on Linux with the following commands::

  $ curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda-installer.sh
  $ bash /tmp/miniconda-installer.sh
  
Type yes when asking to add the miniconda path to your path and restart your session

You can now create your environment and install all the recquirements with::

  $ conda create -n deep_ADNI python=3.6
  $ git clone https://github.com/14thibea/deep_learning_ADNI.git
  $ pip install -r deep_learning_ADNI/recquirements.txt

You also need to install pytorch. Please see `Pytorch installation <https://pytorch.org/>`_ in order to choose the correct command.

Training a network
------------------

You can train a network by typing::

  $ python main/network.py tsv_path results_path caps_path
