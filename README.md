# lunania-ai

our first ai.

# pyenv setup
git clone https://github.com/yyuu/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
exec $SHELL

# pyenv-virtualenv setup
git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
exec $SHELL

# pyenv-pip-rehash setup
git clone https://github.com/yyuu/pyenv-pip-rehash.git ~/.pyenv/plugins/pyenv-pip-rehash

# install python 3.5.2
pyenv install 3.5.2
pyenv global 3.5.2

# create a viratualenv called py35
pyenv virtualenv 3.5.2 py35

# use the virtualenv in a work directory
mkdir work
cd work
pyenv local py35

# install keras and tensorflow
pip install keras tensorflow

# uninstall the virtualenv
pyenv uninstall py35

# remove pyenv-virtualenv setting
rm .python-version

# set enviroment and have some base view.

https://blog.keras.io/category/tutorials.html

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# excute

ex: python tutorial01.py


