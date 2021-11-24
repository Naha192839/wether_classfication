# wether_classfication
# 学科サービスのSingularity + Slurm 環境下での機械学習
- ＜目次＞
    - ゴール
    - ハンズオンの流れ
    - 全体像
    - Tips
        - (a) どうやってコンテナを用意するのか
        - (b) どうやってSlurmで動かすのか
        - その他
---
## 目標
- （自分に）必要なコンテナイメージを作成できる。
- コンテナイメージで実行したプログラムが、実際にGPUで処理されていることを確認できる。
- Slurmへのジョブ投入を通してプログラムを実行できる。

---
## 作成までの流れ

```shell
# step 1: amane（大学の学科サーバー）にログイン。作業用ディレクトリを作成して移動。
ssh amane
mkdir hoge
cd hoge

# step 2: リポジトリのクローンを用意。
git clone https://github.com/Naha192839/wether_classfication
cd wether_classfication

# step 3: コンテナの用意。
singularity pull docker://tensorflow/tensorflow:latest-gpu-py3
singularity build --fakeroot trial.sif clear_rain.def

# step4: GPUの確認
singularity　exec --nv trial.sif python
>>> import tensorflow as tf
2021-03-05 22:15:15.324078: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
2021-03-05 22:15:15.397152: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
>>> print(tf.test.is_gpu_available)
<function is_gpu_available at 0x7f63921e6c80>


# step 5: Slurm経由で実行するためにバッチファイルを作成。
# このままだと、train,pyが実行される。実行したいファイルの変更があれば修性。
vim train.sbatch

# step 6: Slurm経由で実行。
sbatch train.sbatch

# step 7: ジョブ確認。
# squeueで NAME, JOBID を確認。
squeue

# step 8: tail による動作確認。
# 今回用意した train.sbatch では「logs/%x-%j.log」に標準出力を書き出すよう指定している。
# %xはNAME, %jはJOBIDに置き換えられる。
# 下記の xxx は step 9 で確認した JOBID を指定しよう。
# 確認済んだら Ctrl-c でtailコマンドを終了。
tail -f keras-bert-normal-xxx.log
