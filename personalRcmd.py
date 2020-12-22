# -*- coding: utf-8 -*-
# @Time : 2019/1/5 11:28
# @Author : RedCedar
# @File : run.py
# @Software: PyCharm
# @note:

import pandas as pd
from math import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import  fsolve
import math

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_judgement_matrix(scores):
    '''
    get judgement matrix  according to personal score.
    :param scores: a list, the item is the score range 1 to 10 means the importance of each sub-indicator.
    :return: judgement matrix, item range 1 to 9.

    - more: in judgement matrix:
    1 means two sub-indicators are the same important.

    3 means the first sub-indicator is a little important than another one.

    5 means the first sub-indicator is apparently important than another one.

    7 means the first sub-indicator is strongly significant than another one.

    9 means the first sub-indicator is extremely significant than another one.

    and 2, 4, 6, 8 are in the middle degree.
    '''

    # 评分1——10
    length = len(scores)

    array = np.zeros((length, length))
    for i in range(0, length):
        for j in range(0, length):
            point1 = scores[i]
            point2 = scores[j]
            deta = point1 - point2
            if deta < 0:
                continue
            elif deta == 0 or deta == 1:
                array[i][j] = 1
                array[j][i] = 1
            else:
                array[i][j] = deta
                array[j][i] = 1 / deta

    return array


def get_tezheng(array):
    '''
    get the max eigenvalue and eigenvector
    :param array: judgement matrix
    :return: max eigenvalue and the corresponding eigenvector
    '''
    # 获取最大特征值和对应的特征向量
    te_val, te_vector = np.linalg.eig(array)
    list1 = list(te_val)

    max_val = np.max(list1)
    index = list1.index(max_val)
    max_vector = te_vector[:, index]

    return max_val, max_vector


def RImatrix(n):
    '''
    get RI value according the the order
    :param n: matrix order
    :return: Random consistency index RI of a n order matrix
    '''
    n1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    n2 = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59, 1.60]
    d = dict(zip(n1, n2))
    return d[n]


def consitstence(max_val, RI, n):
    '''
    use the CR indicator to test the consistency of a matrix.
    :param max_val: eigenvalue
    :param RI: Random consistency index
    :param n: matrix order
    :return: true or false, denotes whether it meat the validation of consistency
    '''
    CI = (max_val - n) / (n - 1)
    if RI == 0:

        return True
    else:
        CR = CI / RI
        if CR < 0.10:

            return True
        else:

            return False


def minMax(array):
    result = []
    for x in array:
        x = float(x - np.min(array)) / (np.max(array) - np.min(array))
        result.append(x)
    return np.array(result)


def normalize_vector(max_vector):
    '''
    normalize the vector, the sum of elements is 1.0
    :param max_vector: a eigenvector
    :return: normalized eigenvector
    '''
    vector = []
    for i in max_vector:
        vector.append(i.real)
    vector_after_normalization = []
    sum0 = np.sum(vector)
    for i in range(len(vector)):
        vector_after_normalization.append(vector[i] / sum0)
    vector_after_normalization = np.array(vector_after_normalization)

    return vector_after_normalization


def get_weight(score):
    '''
    get weight vector according to personal score.
    :param score: a list, the item is the score range 1 to 10 means the importance of each sub-indicator.
    :return: a list, the item is the weight range 0.0 to 1.0.
    '''
    n = len(score)
    array = get_judgement_matrix(score)
    max_val, max_vector = get_tezheng(array)
    RI = RImatrix(n)
    if consitstence(max_val, RI, n) == True:
        feature_weight = normalize_vector(max_vector)
        return feature_weight
    else:
        return [1 / n] * n


def getScore(array, point1, point2):
    '''
    a normalization function based on Human psychological satisfaction
    :param array: list, element is indicator's original value
    :param point1: the left expectation point, a list, [x1,y1]
    :param point2: the right expectation point, a list, [x2,y2]
    :return: normalized array
    '''
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]

    def f1(a):
        equation1 = 1 / (1 + math.exp(-a * x1)) - y1
        return equation1

    def f2(a):
        equation1 = 1 / (1 + math.exp(-a * x2)) - y2
        return equation1

    # 存储归一化后的值
    values = []

    for i in array:
        try:
            i=i[0]
        except:
            pass
        if i < x1:
            sol3_fsolve = fsolve(f1, [0])
            a = sol3_fsolve[0]
            value = 1 / (1 + math.exp(a * (i - 2 * x1)))
        elif x1 <= i and i <= x2:
            value = (i - x1) * (y2 - y1) / (x2 - x1) + y1
        else:
            sol3_fsolve = fsolve(f2, [0])
            a = sol3_fsolve[0]
            value = 1 / (1 + math.exp(-a * i))
        values.append(value)

    # plt.scatter(array, values)
    # plt.show()
    return values


def show_score(value, title=''):
    x = np.linspace(1, len(value) + 1, len(value))
    plt.scatter(x, value)
    plt.title(title)
    plt.show()


def result(dataDict):
    def price_score():
        '''

        :return: 返回价格指数
        '''

        # 单价
        df = pd.read_csv('all.csv', index_col=0)
        eachPrice_array = df.loc[:, ['单价']].values
        # print('eachPrice_array')
        # print(eachPrice_array)

        # 单价分数
        eachPrice_value = getScore(eachPrice_array[:, 0], [dataDict['price[each_price_min]'], 0.8],
                                   [dataDict['price[each_price_max]'], 0.3])

        # 物业费
        propertyFee_array = df.loc[:, ['物业费']].values
        # 物业费分数
        propertyFee_value = getScore(propertyFee_array, [dataDict['price[property_price_min]'], 0.8],
                                     [dataDict['price[property_price_max]'], 0.3])

        # 楼盘单价  物业费
        price_score = [dataDict['price[price_each]'], dataDict['price[price_property]']]
        price_weight = get_weight(price_score)

        prices_values = []
        for i in range(0, len(propertyFee_array)):
            a1 = eachPrice_value[i]
            a2 = propertyFee_value[i]
            price_value = price_weight * [a1, a2]
            prices_values.append(sum(price_value))

        # show_score(prices_values,'价格指数')
        return prices_values

    def traffic_score():
        df = pd.read_csv('all.csv', index_col=0)
        array = df.loc[:, ['站点数目', '平均距离', '平均线路数目', '平均打分', '平均轨道数目']].values

        # 站点数目打分
        zhandianshumu_value = getScore(array[:, 0], [dataDict['traffic[zhandianshumu_min]'], 0.2],
                                       [dataDict['traffic[zhandianshumu_max]'], 0.8])

        # 平均距离打分
        pinjunjuli_value = getScore(array[:, 1], [dataDict['traffic[pingjunjuli_min]'], 0.8],
                                    [dataDict['traffic[pingjunjuli_max]'], 0.4])

        # 平均线路数目打分
        pingjunxianlu_value = getScore(array[:, 2], [dataDict['traffic[pingjungongjiaoxianlu_min]'], 0.2],
                                       [dataDict['traffic[pingjungongjiaoxianlu_max]'], 0.8])

        # 平均打分打分
        pingjundafen_value = getScore(array[:, 3], [dataDict['traffic[pingjunjiaotongdefen_min]'], 0.2],
                                      [dataDict['traffic[pingjunjiaotongdefen_max]'], 0.8])

        # 平均轨道数目打分
        pingjunguidao_value = getScore(array[:, 4], [dataDict['traffic[pingjunguidaojiaotong_min]'], 0.2],
                                       [dataDict['traffic[pingjunguidaojiaotong_max]'], 0.8])

        # 站点数目	平均距离	平均线路数目	平均打分	平均轨道数目
        traffic_score = [dataDict['traffic[traffic_zhandianshumu]'], dataDict['traffic[traffic_pingjunjuli]'],
                         dataDict['traffic[traffic_pingjungongjiaoxianlu]'],
                         dataDict['traffic[traffic_pingjunjiaotongdefen]'],
                         dataDict['traffic[traffic_pingjunguidaojiaotong]']]
        # 交通各个因素的权重为：
        traffic_weight = get_weight(traffic_score)

        traffic_values = []
        for i in range(0, len(array)):
            a1 = zhandianshumu_value[i]
            a2 = pinjunjuli_value[i]
            a3 = pingjunxianlu_value[i]
            a4 = pingjundafen_value[i]
            a5 = pingjunguidao_value[i]
            traffic_value = traffic_weight * [a1, a2, a3, a4, a5]
            traffic_values.append(sum(traffic_value))
        # show_score(traffic_values,'交通指数')
        return traffic_values

    def community_score():
        '''
        楼盘硬件设施方面
        :return:
        '''
        # 容积率
        # 容积率得分，容积率是越小越好
        # 容积率分为:独立别墅为0.2~0.5;
        # 联排别墅为0.4~0.7;
        # 6层以下多层住宅为0.8~1.2;
        # 11层小高层住宅为1.5~2.0;
        # 18层高层住宅为1.8~2.5;
        # 19层以上住宅为2.4~4.5;
        # 住宅小区容积率小于1.0的，为非普通住宅.
        df = pd.read_csv('all.csv', index_col=0)
        plotRate_array = df.loc[:, ['容积率']].values
        plotRate_value = getScore(plotRate_array[:, 0], [dataDict['community[plotRate_min]'], 0.8],
                                  [dataDict['community[plotRate_max]'], 0.3])

        # 绿化率
        greenRate_array = df.loc[:, ['绿化率']].values
        greenRate_value = getScore(greenRate_array[:, 0], [dataDict['community[greenRate_min]'], 0.4],
                                   [dataDict['community[greenRate_max]'], 0.8])

        # 车位比
        parkProportion_array = df.loc[:, ['车位比']].values
        parkProportion_value = getScore(parkProportion_array[:, 0], [dataDict['community[parkProportion_min]'], 0.2],
                                        [dataDict['community[parkProportion_max]'], 0.8])

        # 容积率  绿化率	车位比
        community_score = [dataDict['community[community_plotRate]'], dataDict['community[community_greenRate]'],
                           dataDict['community[community_parkProportion]']]
        community_weight = get_weight(community_score)

        community_values = []
        for i in range(0, len(parkProportion_array)):
            a1 = plotRate_value[i]
            a2 = greenRate_value[i]
            a3 = parkProportion_value[i]
            community_value = community_weight * [a1, a2, a3]
            community_values.append(sum(community_value))
        # show_score(community_values,'小区设施')
        return community_values

    def muti_score():
        # 建筑类型得分
        # '高层', '花园洋房', '别墅_建筑类型', '多层','住宅','商住','店铺','购物中心','商业街'
        buildingType_score = [dataDict['muti[buildType_gaoCeng]'],
                              dataDict['muti[buildType_huaYuanYangFang]'],
                              dataDict['muti[buildType_bieShu]'],
                              dataDict['muti[buildType_duoCeng]'],
                              dataDict['muti[buildType_zhuZhai]'],
                              dataDict['muti[buildType_shangZhu]'],
                              dataDict['muti[buildType_dianPu]'],
                              dataDict['muti[buildType_gouWuZhongXin]'],
                              dataDict['muti[buildType_shangYeJie]']]
        buildingType_weight = get_weight(buildingType_score)

        df = pd.read_csv('all.csv', index_col=0)
        bulidingType_array = df.loc[:, ['高层', '花园洋房', '别墅_建筑类型', '多层', '住宅', '商住', '店铺', '购物中心', '商业街']].values
        buildingType_value = []
        for i in bulidingType_array:
            buildingType_value.append(np.sum(buildingType_weight * i))
        buildingType_value = minMax(buildingType_value)

        # 楼盘特征得分
        # 其他0，低总价15, 公交枢纽14, 低密度13, 婚房12, 投资地产11,
        #  公园10, 商场9, 公寓8, 大型超市7, 轨交房6, 大型社区5, 品牌开发商4, 改善房3, 学校2, 刚需房1
        feature_score = [1,
                         dataDict['muti[feature_diZongJia]'],
                         dataDict['muti[feature_gongJiaoShuNiu]'],
                         dataDict['muti[feature_diMiDu]'],
                         dataDict['muti[feature_hunFang]'],
                         dataDict['muti[feature_touZiDiChan]'],
                         dataDict['muti[feature_gongYuan]'],
                         dataDict['muti[feature_shangChang]'],
                         dataDict['muti[feature_gongYu]'],
                         dataDict['muti[feature_daXingChaoShi]'],
                         dataDict['muti[feature_guiJiaoFang]'],
                         dataDict['muti[feature_daXingSheQu]'],
                         dataDict['muti[feature_pinPaiKaiFaShang]'],
                         dataDict['muti[feature_gaiShanFang]'],
                         dataDict['muti[feature_xueXiao]'],
                         dataDict['muti[feature_gangXuFang]']]
        feature_weight = get_weight(feature_score)

        feature_array = df.loc[:,
                        ['其他', '刚需房', '学校', '改善房', '品牌开发商', '大型社区', '轨交房', '大型超市', '公寓', '商场', '公园', '投资地产', '婚房',
                         '低密度', '公交枢纽', '低总价']].values
        feature_value = []
        for i in feature_array:
            feature_value.append(np.sum(feature_weight * i))
        feature_value = minMax(buildingType_value)

        # 户型类型得分
        # 1室	2室	3室	4室	5室	6室	7室	别墅	商户
        houseType_score = [dataDict['muti[houseType_1]'],
                           dataDict['muti[houseType_2]'],
                           dataDict['muti[houseType_3]'],
                           dataDict['muti[houseType_4]'],
                           dataDict['muti[houseType_5]'],
                           dataDict['muti[houseType_6]'],
                           dataDict['muti[houseType_7]'],
                           dataDict['muti[houseType_bieShu]'],
                           dataDict['muti[houseType_shangHu]']]
        houseType_weight = get_weight(houseType_score)

        houseType_array = df.loc[:, ['1室', '2室', '3室', '4室', '5室', '6室', '7室', '别墅_户型', '商户']].values
        houseType_value = []
        for i in houseType_array:
            value = i * houseType_weight
            value = np.sum(value)
            houseType_value.append(value)
        houseType_value = getScore(houseType_value, [0.4, 0.2], [2, 0.8])

        # 建筑类型 楼盘特征 户型
        muti_score = [dataDict['muti[muti_buildType]'], dataDict['muti[muti_feature]'],
                      dataDict['muti[muti_houseType]']]
        muti_weight = get_weight(muti_score)

        muti_values = []
        for i in range(0, len(bulidingType_array)):
            a1 = buildingType_value[i]
            a2 = feature_value[i]
            a3 = houseType_value[i]
            muti_value = muti_weight * [a1, a2, a3]
            muti_values.append(sum(muti_value))
        # show_score(muti_values,'楼盘多样性')
        return muti_values

    def location_score():
        def manhatten(point1, point2):
            def haversine(point1, point2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
                """
                Calculate the great circle distance between two points
                on the earth (specified in decimal degrees)
                """
                # 将十进制度数转化为弧度
                lon1 = point1[0]
                lat1 = point1[1]
                lon2 = point2[0]
                lat2 = point2[1]
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

                # haversine公式
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * asin(sqrt(a))
                r = 6371  # 地球平均半径，单位为公里
                return c * r

            p = [point2[0], point1[1]]
            return haversine(point1, p) + haversine(p, point2)

        # 地域得分
        region_score = [1,
                        dataDict['location[region_nanGuan]'],
                        dataDict['location[region_zhaoYang]'],
                        dataDict['location[region_jingYue]'],
                        dataDict['location[region_lvYuan]'],
                        dataDict['location[region_erDao]'],
                        dataDict['location[region_gaoXin]'],
                        dataDict['location[region_jingKai]'],
                        dataDict['location[region_kuanCheng]'],
                        dataDict['location[region_qiKai]'],
                        1]
        region_weight = get_weight(region_score)

        df = pd.read_csv('all.csv', index_col=0)
        region_array = df.loc[:, ['其他', '南关', '朝阳', '净月', '绿园', '二道', '高新', '经开', '宽城', '汽开', '长春周边']].values
        region_value = []
        for i in region_array:
            region_value.append(np.sum(region_weight * i))
        region_value = minMax(region_value)

        # 平均曼哈段距离
        # 经纬度
        jw_array = df.loc[:, ['经度', '纬度']].values
        points = [[125.330665, 43.917598],
                  [125.313642, 43.898338],
                  [125.457405, 43.808973],
                  [125.453503, 43.778884],
                  [125.307394, 43.876633],
                  [125.296092, 43.870066],
                  [125.322061, 43.872801],
                  [125.330277, 43.897067],
                  [125.31221, 43.882698],
                  [125.453503, 43.778884]]
        jw_value = []
        for i in jw_array:
            manhatten_distance = 0
            for j in points:
                manhatten_distance += manhatten(i, j)
            # 平均曼哈顿距离

            avg_manhatten_distance = manhatten_distance / len(points)
            jw_value.append(avg_manhatten_distance)

        jw_value = getScore(jw_value, [dataDict['location[manhatten_min]'], 0.7],
                            [dataDict['location[manhatten_max]'], 0.4])

        # 地域得分 平均曼哈顿距离
        location_score = [dataDict['location[location_region]'], dataDict['location[location_manhatten]']]
        location_weight = get_weight(location_score)

        location_values = []
        for i in range(0, len(region_array)):
            a1 = region_value[i]
            a2 = jw_value[i]
            location_value = location_weight * [a1, a2]
            location_values.append(sum(location_value))
        # show_score(location_values,'地域指数')
        return location_values

    # # 字符改为数值
    # for key, value in list(dataDict.items()):
    #     dataDict[key] = float(value)

    ps = price_score()
    ts = traffic_score()
    cs = community_score()
    ms = muti_score()
    ls = location_score()
    ps=np.array(ps).reshape(242,)

    V = [dataDict['final[final_price]'],
         dataDict['final[final_traffic]'],
         dataDict['final[final_community]'],
         dataDict['final[final_muti]'],
         dataDict['final[final_location]']]
    W = get_weight(V)

    M = []
    data = []
    df = pd.read_csv('all.csv', index_col=0)
    names = df.loc[:, ['楼盘名称']].values[:, 0]
    prices = df.loc[:, ['单价']].values[:, 0]
    for i in range(0, len(ps)):
        a1 = ps[i]
        a2 = ts[i]
        a3 = cs[i]
        a4 = ms[i]
        a5 = ls[i]
        q = W * [a1, a2, a3, a4, a5]
        m = sum(q)
        M.append(m)
        name = names[i]
        price = prices[i]
        data.append([name, price, a1, a2, a3, a4, a5, m, price / m])
    df = pd.DataFrame(data, columns=['楼盘名称', '价格', '价格指数', '交通指数', '小区设施', '楼盘多样性', '地域指数', '楼盘综合水平', '价格/综合水平'])
    df.to_excel('comprehensive evaluation.xlsx')
    return data


if __name__ == '__main__':
    '''
    you can mark your personal score in daraDict
    '''
    dataDict = {'final[final_price]': 8,
                'final[final_traffic]': 6,
                'final[final_community]': 5,
                'final[final_muti]': 3,
                'final[final_location]': 4,
                'price[each_price_min]': 6000,
                'price[each_price_max]': 12000,
                'price[property_price_min]': 0.5,
                'price[property_price_max]': 1.4,
                'traffic[zhandianshumu_min]': 2,
                'traffic[zhandianshumu_max]': 6,
                'traffic[pingjunjuli_min]': 500,
                'traffic[pingjunjuli_max]': 1200,
                'traffic[pingjungongjiaoxianlu_min]': 2,
                'traffic[pingjungongjiaoxianlu_max]': 5,
                'traffic[pingjunjiaotongdefen_min]': 3,
                'traffic[pingjunjiaotongdefen_max]': 6,
                'traffic[pingjunguidaojiaotong_min]': 1,
                'traffic[pingjunguidaojiaotong_max]': 3,
                'community[plotRate_min]': 0.8,
                'community[plotRate_max]': 2.2,
                'community[greenRate_min]': 0.5,
                'community[greenRate_max]': 1.1,
                'community[parkProportion_min]': 0.5,
                'community[parkProportion_max]': 1.0,
                }
    dataDict['price[price_each]']=7
    dataDict['price[price_property]']=3

    dataDict['traffic[traffic_zhandianshumu]']=5
    dataDict['traffic[traffic_pingjunjuli]']=6
    dataDict['traffic[traffic_pingjungongjiaoxianlu]']=4
    dataDict['traffic[traffic_pingjunjiaotongdefen]']=8
    dataDict['traffic[traffic_pingjunguidaojiaotong]']=3

    dataDict['community[community_plotRate]']=4
    dataDict['community[community_greenRate]']=7
    dataDict['community[community_parkProportion]']=5

    dataDict['muti[muti_buildType]']=5
    dataDict['muti[muti_feature]']=5
    dataDict['muti[muti_houseType]']=5
    dataDict['location[manhatten_min]']=1000
    dataDict['location[manhatten_max]']=3000

    dataDict['location[location_region]']=7
    dataDict['location[location_manhatten]']=4

    dataDict['muti[buildType_gaoCeng]'] = 1
    dataDict['muti[buildType_huaYuanYangFang]'] = 0
    dataDict['muti[buildType_bieShu]'] = 0
    dataDict['muti[buildType_duoCeng]'] = 0
    dataDict['muti[buildType_zhuZhai]'] = 0
    dataDict['muti[buildType_shangZhu]'] = 1
    dataDict['muti[buildType_dianPu]'] = 1
    dataDict['muti[buildType_gouWuZhongXin]'] = 0
    dataDict['muti[buildType_shangYeJie]'] = 1

    dataDict['muti[feature_diZongJia]'] = 1
    dataDict['muti[feature_gongJiaoShuNiu]'] = 1
    dataDict['muti[feature_diMiDu]'] = 0
    dataDict['muti[feature_hunFang]'] = 1
    dataDict['muti[feature_touZiDiChan]'] = 1
    dataDict['muti[feature_gongYuan]'] = 1
    dataDict['muti[feature_shangChang]'] = 0
    dataDict['muti[feature_gongYu]'] = 0
    dataDict['muti[feature_daXingChaoShi]'] = 1
    dataDict['muti[feature_guiJiaoFang]'] = 1
    dataDict['muti[feature_daXingSheQu]'] = 1
    dataDict['muti[feature_pinPaiKaiFaShang]'] = 0
    dataDict['muti[feature_gaiShanFang]'] = 1
    dataDict['muti[feature_xueXiao]'] = 1
    dataDict['muti[feature_gangXuFang]'] = 1

    dataDict['muti[houseType_1]'] = 1
    dataDict['muti[houseType_2]'] = 1
    dataDict['muti[houseType_3]'] = 0
    dataDict['muti[houseType_4]'] = 1
    dataDict['muti[houseType_5]'] = 0
    dataDict['muti[houseType_6]'] = 0
    dataDict['muti[houseType_7]'] = 1
    dataDict['muti[houseType_bieShu]'] = 0
    dataDict['muti[houseType_shangHu]'] = 0

    dataDict['location[region_nanGuan]'] = 7
    dataDict['location[region_zhaoYang]'] = 9
    dataDict['location[region_jingYue]'] = 5
    dataDict['location[region_lvYuan]'] = 3
    dataDict['location[region_erDao]'] = 4
    dataDict['location[region_gaoXin]'] = 4
    dataDict['location[region_jingKai]'] = 3
    dataDict['location[region_kuanCheng]'] = 2
    dataDict['location[region_qiKai]'] = 2
    data = result(dataDict)
    print(data)
