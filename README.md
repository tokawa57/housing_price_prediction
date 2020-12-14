## 住宅価格の予測モデル作成

### 予測フロー
1. scraping.pyファイルでSUUMOから物件情報をexcelとして落とす (ex. suumo_chiyoda.csv)
2. preprocessing_and_learning.pyファイルで落とした情報を前処理&学習(lihgtBGM)
- 最終的な出力は、実際の値と予測値の差を算出(ex. diff_chiyoda.csv)


#### 変更出来るパラメータ
- スクレイピング&予測する地区はscraping.py,preprocessing_and_learning.py共にitemで指定
- item = 'chuo'とすると中央区のデータを取ってきて学習(SUUMOのサイトに準ずる)
- endパラメータで取得するページ数を設定
- end = Noneとすると、全ページ取得（数十分程度）。デフォルトでは3ページと設定。		

#### 実行
- pyファイルをダウンロードしてpython scraping.py等で実行可能
