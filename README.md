# image_converter
## 概要
メルアイコン変換器で用いたソースコードです。  
詳しい解説は<a href="https://qiita.com/zassou65535/items/4bc42fa36203c13fe2d3">こちら</a>。

## 想定環境
python 3.7.1  
`pip install -r requirements.txt`で環境を揃えることができます。 

## プログラム
* `UGATIT_train.py`は学習を実行し、学習の過程と学習済みモデルを出力するプログラムです。  
* `UGATIT_inference.py`は`UGATIT_train.py`で出力した学習済みモデルを読み込み、推論(画像の変換)を実行、生成画像を出力するプログラムです。 

## 使い方
以下では変換元ドメインをA、変換先ドメインをBと表現します。
### 学習の実行
1. `UGATIT_train.py`のあるディレクトリに`./dataset`ディレクトリを作成します
1. `./dataset`ディレクトリ内に`group_A`ディレクトリと`group_B`ディレクトリの2つを作成します。
1. `./dataset/group_A`ディレクトリに、Aに属する画像を`./dataset/group_A/*/*`という形式で好きな数入れます(画像のファイル形式はpng)。
1. `./dataset/group_B`ディレクトリに、Bに属する画像を`./dataset/group_B/*/*`という形式で好きな数入れます(画像のファイル形式はpng)。
1. `UGATIT_train.py`の置いてあるディレクトリで`python UGATIT_train.py`を実行することで、「A⇄B」の変換ができるよう目指して学習を実行します。
	* 学習の過程が`./output`以下に出力されます。
	* 学習済みモデルが`./trained_model/generator_A2B_trained_model_cpu.pth`として出力されます。
### 推論の実行
1. `UGATIT_inference.py`のあるディレクトリに`./conversion`ディレクトリを作成します
1. `./conversion`内に`target`ディレクトリを作成し、Aに属する画像を好きな数入れます。
1. `UGATIT_inference.py`の置いてあるディレクトリで`python UGATIT_inference.py`を実行して`./conversion/target`内の画像をBへ変換します
	* A→Bの変換結果が`./conversion/converted/`以下に出力されます。
	* 注意点として、`./trained_model`内に学習済みモデル`generator_A2B_trained_model_cpu.pth`がなければエラーとなります

学習には環境によっては12時間以上要する場合があります。    
入力された画像は256×256にリサイズされた上で学習に使われます。出力画像も256×256です。 
