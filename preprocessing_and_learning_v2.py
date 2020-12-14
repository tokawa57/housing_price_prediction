#!/usr/bin/env python
# coding: utf-8



#必要なライブラリをインポート
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import pandas_profiling as pdp 



item = 'chiyoda'
df = pd.read_csv(f'suumo_{item}.csv', sep='\t', encoding='utf-16')


#不要な列を削除
df.drop(['Unnamed: 0'], axis=1, inplace=True)

#立地を「最寄駅」と「徒歩〜分」に分割
splitted1 = df['立地1'].str.split(' 歩', expand=True)
splitted1.columns = ['立地11', '立地12']
splitted2 = df['立地2'].str.split(' 歩', expand=True)
splitted2.columns = ['立地21', '立地22']
splitted3 = df['立地3'].str.split(' 歩', expand=True)
splitted3.columns = ['立地31', '立地32']

#その他費用を、「敷金」「礼金」「保証金」「敷引,償却」に分割
splitted4 = df['敷/礼/保証/敷引,償却'].str.split('/', expand=True)
splitted4.columns = ['敷金', '礼金']


# In[566]:


#「敷引,償却」をさらに「敷引」「償却」に分割
#splitted5 = df['敷引,償却'].str.split('・', expand=True)
#splitted5.columns = ['敷引', '償却']

#分割したカラムを結合
df = pd.concat([df, splitted1, splitted2, splitted3, splitted4], axis=1)

#分割前のカラムは分析に使用しないので削除しておく
df.drop(['立地1','立地2','立地3','敷/礼/保証/敷引,償却'], axis=1, inplace=True)

#「賃料」がNAの行を削除
df = df.dropna(subset=['賃料'])

#エンコードをcp932に変更しておく（これをしないと、replaceできない）
df['賃料'].str.encode('cp932')
df['敷金'].str.encode('cp932')
df['礼金'].str.encode('cp932')
#df['保証金'].str.encode('cp932')
#df['敷引'].str.encode('cp932')
#df['償却'].str.encode('cp932')
df['管理費'].str.encode('cp932')
df['築年数'].str.encode('cp932')
df['専有面積'].str.encode('cp932')
df['立地12'].str.encode('cp932')
df['立地22'].str.encode('cp932')
df['立地32'].str.encode('cp932')

#数値として扱いたいので、不要な文字列を削除
df['賃料'] = df['賃料'].str.replace(u'万円', u'')
df['敷金'] = df['敷金'].str.replace(u'万円', u'')
df['礼金'] = df['礼金'].str.replace(u'万円', u'')
#df['保証金'] = df['保証金'].str.replace(u'万円', u'')
#df['敷引'] = df['敷引'].str.replace(u'万円', u'')
#df['償却'] = df['償却'].str.replace(u'万円', u'')
df['管理費'] = df['管理費'].str.replace(u'円', u'')
df['築年数'] = df['築年数'].str.replace(u'新築', u'0') #新築は築年数0年とする
df['築年数'] = df['築年数'].str.replace(u'築', u'')
df['築年数'] = df['築年数'].str.replace(u'年', u'')
df['築年数'] = df['築年数'].str.replace(u'99年以上', u'99') #
df['専有面積'] = df['専有面積'].str.replace(u'm', u'')
df['立地12'] = df['立地12'].str.replace(u'分', u'')
df['立地22'] = df['立地22'].str.replace(u'分', u'')
df['立地32'] = df['立地32'].str.replace(u'分', u'')

#「-」を0に変換
df['管理費'] = df['管理費'].replace('-',0)
df['敷金'] = df['敷金'].replace('-',0)
df['礼金'] = df['礼金'].replace('-',0)
#df['保証金'] = df['保証金'].replace('-',0)
#df['敷引'] = df['敷引'].replace('-',0)
#df['敷引'] = df['敷引'].replace('実費',0) #「実費」と文字列が入っている場合がある
#df['償却'] = df['償却'].replace('-',0)

#Noneを0に変換
#df['償却'] = [0 if x is None else x for x in df['償却']]

#文字列から数値に変換
df['賃料'] = pd.to_numeric(df['賃料'])
df['管理費'] = pd.to_numeric(df['管理費'])
df['敷金'] = pd.to_numeric(df['敷金'])
df['礼金'] = pd.to_numeric(df['礼金'])
#df['保証金'] = pd.to_numeric(df['保証金'])
#df['敷引'] = pd.to_numeric(df['敷引'])
#df['償却'] = pd.to_numeric(df['償却'])
df['築年数'] = pd.to_numeric(df['築年数'])
df['専有面積'] = pd.to_numeric(df['専有面積'])
df['立地12'] = pd.to_numeric(df['立地12'])
df['立地22'] = pd.to_numeric(df['立地22'])
df['立地32'] = pd.to_numeric(df['立地32'])

#単位を合わせるために、管理費以外を10000倍。
df['賃料'] = df['賃料'] * 10000
df['敷金'] = df['敷金'] * 10000
df['礼金'] = df['礼金'] * 10000
#df['保証金'] = df['保証金'] * 10000
#df['敷引'] = df['敷引'] * 10000
#df['償却'] = df['償却'] * 10000

#管理費は実質的には賃料と同じく毎月支払うことになるため、「賃料+管理費」を家賃を見る指標とする
df['賃料+管理費'] = df['賃料'] + df['管理費']

#敷金/礼金と保証金は同じく初期費用であり、どちらかが適用されるため、合計を初期費用を見る指標とする
df['敷/礼/保証金'] = df['敷金'] + df['礼金'] #+ df['保証金']

#住所を「東京都」「〜区」「市町村番地」に分割
splitted6 = df['住所'].str.split('区', expand=True)
splitted6.columns = ['区', '市町村']
splitted6['区'] = splitted6['区'] + '区'
splitted6['区'] = splitted6['区'].str.replace('東京都','')

#路線と駅が３つ以上に分かれてしまう行を削除
#drop_idx = [98764] #手動でindexを確認
#df = df.drop(drop_idx, axis=0)

#立地を「路線」「駅」「徒歩〜分」に分割
splitted7 = df['立地11'].str.split('/', expand=True)
splitted7.columns = ['路線1', '駅1']
splitted7['徒歩1'] = df['立地12']
splitted8 = df['立地21'].str.split('/', expand=True)
splitted8.columns = ['路線2', '駅2']
splitted8['徒歩2'] = df['立地22']
splitted9 = df['立地31'].str.split('/', expand=True)
splitted9.columns = ['路線3', '駅3']
splitted9['徒歩3'] = df['立地32']

#結合
df = pd.concat([df, splitted6, splitted7, splitted8, splitted9], axis=1)

#不要なカラムを削除
df.drop(['立地11','立地12','立地21','立地22','立地31','立地32'], axis=1, inplace=True)

#階を数値化。地下はマイナスとして扱う

splitted10 = df['階'].str.split('-', expand=True)
splitted10.columns = ['階1', '階2']
splitted10['階1'].str.encode('cp932')
splitted10['階1'] = splitted10['階1'].str.replace(u'階', u'')
splitted10['階1'] = splitted10['階1'].str.replace(u'B', u'-')
splitted10['階1'] = pd.to_numeric(splitted10['階1'])
df = pd.concat([df, splitted10], axis=1)

#建物高さを数値化。地下は無視。
df['建物高さ'].str.encode('cp932')
df['建物高さ'] = df['建物高さ'].str.replace(u'地下1地上', u'')
df['建物高さ'] = df['建物高さ'].str.replace(u'地下2地上', u'')
df['建物高さ'] = df['建物高さ'].str.replace(u'地下3地上', u'')
df['建物高さ'] = df['建物高さ'].str.replace(u'地下4地上', u'')
df['建物高さ'] = df['建物高さ'].str.replace(u'地下5地上', u'')
df['建物高さ'] = df['建物高さ'].str.replace(u'地下6地上', u'')
df['建物高さ'] = df['建物高さ'].str.replace(u'地下7地上', u'')
df['建物高さ'] = df['建物高さ'].str.replace(u'地下8地上', u'')
df['建物高さ'] = df['建物高さ'].str.replace(u'地下9地上', u'')
df['建物高さ'] = df['建物高さ'].str.replace(u'平屋', u'1')
df['建物高さ'] = df['建物高さ'].str.replace(u'階建', u'')
df['建物高さ'] = pd.to_numeric(df['建物高さ'])

#indexを振り直す（これをしないと、以下の処理でエラーが出る）
df = df.reset_index(drop=True)

#間取りを「部屋数」「DK有無」「K有無」「L有無」「S有無」に分割
df['間取りDK'] = 0
df['間取りK'] = 0
df['間取りL'] = 0
df['間取りS'] = 0
df['間取り'].str.encode('cp932')
df['間取り'] = df['間取り'].str.replace(u'ワンルーム', u'1') #ワンルームを1に変換

for x in range(len(df)):
    if 'DK' in df['間取り'][x]:
        df.loc[x,'間取りDK'] = 1
df['間取り'] = df['間取り'].str.replace(u'DK',u'')

for x in range(len(df)):
    if 'K' in df['間取り'][x]:
        df.loc[x,'間取りK'] = 1        
df['間取り'] = df['間取り'].str.replace(u'K',u'')

for x in range(len(df)):
    if 'L' in df['間取り'][x]:
        df.loc[x,'間取りL'] = 1        
df['間取り'] = df['間取り'].str.replace(u'L',u'')

for x in range(len(df)):
    if 'S' in df['間取り'][x]:
        df.loc[x,'間取りS'] = 1        
df['間取り'] = df['間取り'].str.replace(u'S',u'')

df['間取り'] = pd.to_numeric(df['間取り'])



df[['区','市町村','路線1','路線2','路線3', '駅1', '駅2','駅3']] = df[['区','市町村','路線1','路線2','路線3', '駅1', '駅2','駅3']].fillna("NAN")
oe = preprocessing.OrdinalEncoder()
df[['区','市町村','路線1','路線2','路線3', '駅1','駅2','駅3']] = oe.fit_transform(df[['区','市町村','路線1','路線2','路線3', '駅1', '駅2','駅3']].values) 



#範囲限定
df = df[(df['賃料+管理費'] < 300000)]
df.reset_index(inplace=True)
df




df_for_search = df.copy()

df = df[['マンション名','賃料+管理費','築年数','建物高さ','階1','専有面積','住所','区','市町村','間取り','間取りDK','間取りK','間取りL','間取りS',                '路線1','駅1','徒歩1','路線2','駅2','徒歩2','路線3','駅3','徒歩3']]
df.columns = ['name','real_rent','age', 'height', 'level','area','address','address1','address2', 'room_plan','DK','K','L','S'              ,'route_1','station_1','distance_1','route_2','station_2','distance_2','route_3','station_3','distance_3']


#特徴量
#一つの部屋の面積。いちばん重要なのは、
df["per_area"] = df["area"]/df["room_plan"]
df["height_level"] = df["height"]*df["level"]
df["area_height_level"] = df["area"]*df["height_level"]
#df["distance_staion_1"] = df["station_1"]*df["distance_1"]




#learning



import matplotlib.pyplot as plt
import japanize_matplotlib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score





y = df["real_rent"]
#X = df.drop(['real_rent',"name"], axis=1) 
#X = df[['age', 'height', 'level', 'area', 'address1', 'address2','room_plan', 'DK', 'K', 'L', 'S', 'station_1','distance_1', 'per_area','area_height_level']]

X = df.drop(['real_rent','name','address'], axis=1) 





X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)




lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

lgbm_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves':50
}


# In[577]:


model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval, verbose_eval=-1)


# In[578]:


y_pred = model.predict(X_test, num_iteration=model.best_iteration)
print('r2_score: ', r2_score(y_test, y_pred))
#lgb.plot_importance(model, figsize=(12, 6))
#plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[579]:


pred = list(model.predict(X, num_iteration=model.best_iteration))
pred = pd.Series(pred, name="予測値")
diff = pd.Series(df_for_search['賃料+管理費']-pred,name="予測値との差")

df_search = pd.concat([df_for_search,diff,pred], axis=1)
df_search = df_search.sort_values("予測値との差")
df_search = df_search[["マンション名",'賃料+管理費', '予測値',  '予測値との差', '詳細URL']]
df_search.to_csv(f'diff_{item}.csv', sep = '\t',encoding='utf-16')


