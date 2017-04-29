# lunania-ai

our first ai.

# pyenv setup
```
git clone https://github.com/yyuu/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
exec $SHELL
```

# install python 3.5.2
```
pyenv install 3.5.2
pyenv global 3.5.2
```

# install keras and tensorflow
```
pip install keras tensorflow
pip install pillow
pip install h5py
brew install graphviz
pip install pydot
pip install pyparsing
pip install matplotlib
vi ~/.matplotlib/matplotlibrc

	backend : TkAgg

```

# set enviroment and have some base view.

https://blog.keras.io/category/tutorials.html

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# excute

## you can train with 3 method

### completely train by yourself
```
cd catvsdog/01_job
python scratch.py
```

### use vgg16 base, train top fc only.
```
cd catvsdog/01_job
python bottleneck.py
```

### use vgg16 base, train top fc and the 16th convenience2D.
```
cd catvsdog/01_job
python finetuning.py
```

## use weights to predict cat or dog

you can set mode from 1 to 3(1: scratch, 2: bottleneck, 3: finetuning)
and use image param to specify a image
```
cd catvsdog/01_job
python predict.py --mode 3 --image ../99_data/input/1.png
```

