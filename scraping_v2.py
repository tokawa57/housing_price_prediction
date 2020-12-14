#!/usr/bin/env python
# coding: utf-8



#必要なライブラリをインポート
from bs4 import BeautifulSoup
import requests
import urllib3
import re
import pandas as pd
from pandas import Series, DataFrame
import time



#URL（東京都足立区の賃貸住宅情報 検索結果の1ページ目）
#url = 'http://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&ta=13&sc=13121&cb=0.0&ct=9999999&et=9999999&cn=9999999&mb=0&mt=9999999&shkr1=03&shkr2=03&shkr3=03&shkr4=03&fw2=&srch_navi=1'

#地区指定
item = 'chiyoda'
#url = 'https://suumo.jp/chintai/tokyo/sc_adachi/'

#スクレイピングするページ数の指定
#end = Noneとすると全ページ取得
#end = None
end = 3


url = f'https://suumo.jp/chintai/tokyo/sc_{item}/'

#データ取得
result = requests.get(url)
c = result.content

#HTMLを元に、オブジェクトを作る
soup = BeautifulSoup(c)

#物件リストの部分を切り出し
summary = soup.find("div",{'id':'js-bukkenList'})

#ページ数を取得
body = soup.find("body")
pages = body.find_all("div",{'class':'pagination pagination_set-nav'})
pages_text = str(pages)
pages_split = pages_text.split('</a></li>\n</ol>')
pages_split0 = pages_split[0]
pages_split1 = pages_split0[-3:]
pages_split2 = pages_split1.replace('>','')
pages_split3 = int(pages_split2)

#URLを入れるリスト
urls = []

#1ページ目を格納
urls.append(url)

#2ページ目から最後のページまでを格納
for i in range(pages_split3-1):
    pg = str(i+2)
    url_page = url + '?page=' + pg
    urls.append(url_page)

name = [] #マンション名
address = [] #住所
locations0 = [] #立地1つ目（最寄駅/徒歩~分）
locations1 = [] #立地2つ目（最寄駅/徒歩~分）
locations2 = [] #立地3つ目（最寄駅/徒歩~分）
age = [] #築年数
height = [] #建物高さ
floor = [] #階
rent = [] #賃料
admin = [] #管理費
others = [] #敷/礼/保証/敷引,償却
floor_plan = [] #間取り
area = [] #専有面積

detail_url = []



#各ページで以下の動作をループ
for url in urls[:end]:
    #物件リストを切り出し
    result = requests.get(url)
    c = result.content
    soup = BeautifulSoup(c)
    summary = soup.find("div",{'id':'js-bukkenList'})
    #マンション名、住所、立地（最寄駅/徒歩~分）、築年数、建物高さが入っているcassetteitemを全て抜き出し
    cassetteitems = summary.find_all("div",{'class':'cassetteitem'})
    #各cassetteitemsに対し、以下の動作をループ

    for i in range(len(cassetteitems)):
        #各建物から売りに出ている部屋数を取得
        tbodies = cassetteitems[i].find_all('tbody')     
        
        #こっちに持ってくると動作する。なぜ？
        
        age_and_height = cassetteitems[i].find('li', class_='cassetteitem_detail-col3')
        _age = age_and_height('div')[0].text
        _height = age_and_height('div')[1].text
        #_age = str(cassetteitems[i].find("li",{'class':'cassetteitem_detail-col3'}).find_all('div')[0]).replace('<div>','').replace('</div>','')
        #_height = str(cassetteitems[i].find("li",{'class':'cassetteitem_detail-col3'}).find_all('div')[1]).replace('<div>','').replace('</div>','')
        
        _name = cassetteitems[i].find('div', class_='cassetteitem_content-title').text
        _address = cassetteitems[i].find('li', class_='cassetteitem_detail-col1').text
        
        #部屋数だけ、マンション名と住所を繰り返しリストに格納（部屋情報と数を合致させるため）
        for y in range(len(tbodies)):
            name.append(_name)
            address.append(_address)
            age.append(_age)
            height.append(_height)
        #立地を取得
        sublocations = cassetteitems[i].find_all("li",{
            'class':'cassetteitem_detail-col2'})     
        #立地は、1つ目から3つ目までを取得（4つ目以降は無視）
        for x in sublocations:
            cols = x.find_all('div')
            for i in range(len(cols)):
                text = cols[i].find(text=True)
                for y in range(len(tbodies)):
                    if i == 0:
                        locations0.append(text)
                    elif i == 1:
                        locations1.append(text)
                    elif i == 2:
                        locations2.append(text)                    
        
    #階、賃料、管理費、敷/礼/保証/敷引,償却、間取り、専有面積が入っているtableを全て抜き出し
    tables = summary.find_all('table')
    #各建物（table）に対して、売りに出ている部屋（row）を取得
    rows = []
    for i in range(len(tables)):
        rows.append(tables[i].find_all('tr'))

    for row in rows:
        #trは一つの部屋のデータ
        #trごとにdataを作りたい。
        for tr in row:
            cols = tr.find_all('td')
            #print(cols)    
            if len(cols) != 0:
                
                _floor = cols[2].text
                _floor = re.sub('[\r\n\t]', '', _floor)
                floor.append(_floor)

                _rent_cell = cols[3].find('ul').find_all('li')
                _rent = _rent_cell[0].find('span').text
                _admin = _rent_cell[1].find('span').text
                rent.append(_rent)
                admin.append(_admin)

                _deposit_cell = cols[4].find('ul').find_all('li')
                _deposit = _deposit_cell[0].find('span').text
                _reikin = _deposit_cell[1].find('span').text
                _others = _deposit + '/' + _reikin
                others.append(_others)
                
                _floor_cell = cols[5].find('ul').find_all('li')
                _floor_plan = _floor_cell[0].find('span').text
                _area = _floor_cell[1].find('span').text
                floor_plan.append(_floor_plan)
                area.append(_area)

                _detail_url = cols[8].find('a')['href']
                _detail_url = 'https://suumo.jp' + _detail_url
                
                detail_url.append(_detail_url)
                
    #プログラムを10秒間停止する（スクレイピングマナー）
    time.sleep(3)



#各リストをシリーズ化
name = Series(name)
address = Series(address)
locations0 = Series(locations0)
locations1 = Series(locations1)
locations2 = Series(locations2)
age = Series(age)
height = Series(height)
floor = Series(floor)
rent = Series(rent)
admin = Series(admin)
others = Series(others)
floor_plan = Series(floor_plan)
area = Series(area)
detail_url = Series(detail_url)



#各シリーズをデータフレーム化
suumo_df = pd.concat([name, address, locations0, locations1, locations2, age, height, floor, rent, admin, others, floor_plan, area,detail_url], axis=1)
#カラム名
suumo_df.columns=['マンション名','住所','立地1','立地2','立地3','築年数','建物高さ','階','賃料','管理費', '敷/礼/保証/敷引,償却',                  '間取り','専有面積', '詳細URL']

#csvファイルとして保存
suumo_df.to_csv(f'suumo_{item}.csv', sep = '\t',encoding='utf-16')



