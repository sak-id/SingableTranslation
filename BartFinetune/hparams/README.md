# 各シェルプログラムの説明

## 使い方
with_XX.shの代わりに使いたいシェルプログラムの名前を入れる
```bash:
$ cd SingableTranslation/BartFinetune

# for finetuning
$ sh train.sh hparams/with_XX.sh

# After finetuning, for inference
$ sh infer.sh hparams/with_XX.sh
```
with_not_finetuned.sh以外はtrain.sh, infer.sh両方に使える

## シェルプログラム
### with_bt.sh, with_parallel.sh
- それぞれ逆翻訳、対訳データでの実行用パラメータ
- 元論文の著者の方が提供している中国語データで動作確認した時の値が入っています

### with_try_bt.sh
- 単純なfinetuning
- データセット：RWCデータセットの日本語歌詞+ 逆翻訳
- Model, tokenizerともにmbart-large-50-one-to-many-mmt

### with_not_finetuned.sh
- finetuningをしていないmbartの性能を計るためのパラメータ
- データセットは同じ（RWC)
- train.shには使えません

### with_newtoken_bt.sh
- tokenizerのみ変更
    - len_XX(1~20), rhy_X(0~6), str_X(0~1)と\<prefix>,\</prefix>, \<brk>トークンを追加している
    - トークンを使わずにfinetuneし、正しく動くか確認する
        - 厳密には一致しないが、ほぼwith_try_btと同じ時のスコアが出たのでOK

### with_pref_len.sh
- lenトークンのみ使ってfinetune
    - len_XXでXX字と指定しencoderに入れる
    - が、length_accuracyはwith_try_btと変わらない