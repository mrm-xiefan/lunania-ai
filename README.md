# lunania-ai

our first ai.

## install brew
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

## install git
```
yum install git
```

## pyenv setup
```
git clone https://github.com/yyuu/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
```

## install python 3.5.2
```
pyenv install 3.5.2
pyenv global 3.5.2
```

## install keras and tensorflow
```
pip install keras==1.2.2
pip install tensorflow
pip install pillow
pip install h5py
brew install graphviz
pip install pydot
pip install pydot-ng
pip install pyparsing
pip install matplotlib

```

## excute

### at first you can train with 3 method

1. completely train by yourself
```
cd catvsdog/01_job
python scratch.py
```

2. use vgg16 base, train top fc only.
```
cd catvsdog/01_job
python bottleneck.py
```

3. use vgg16 base, train top fc and the 16th convenience2D.
```
cd catvsdog/01_job
python finetuning.py
```

### then use weights to predict cat or dog

you can set mode from 1 to 3(1: scratch, 2: bottleneck, 3: finetuning)

and use image param to specify an image.
*for example:*
```
cd catvsdog/01_job
python predict.py --mode 3 --image ../99_data/input/1.png
```
