# RIAモジュール

### DEV

1. 新規ファイルテンプレート  
```python
#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
LOGGER = logging.getLogger(__name__)
```

2. 環境設定ファイルの修正  
[application.yaml](./app/resources/application.yaml)内の学習データとモデルの格納ディレクトリを設定する。  
同フォルダ内に[application.local.yaml](./app/resources/application.local.yaml)を作成すると
設定を上書きできます。


3. 依存ライブラリのインストール
```
pip install -r requirements.txt
```
失敗する場合に個別にinstallを行ってください。  
またpydensecrfは下記のURLよりinstallを行ってください。  
https://github.com/lucasb-eyer/pydensecrf

4. 可視化ツールのinstall
kerasのモデル可視化機能にてgraphvizを使用します。
```
sudo yum -y install graphviz
```