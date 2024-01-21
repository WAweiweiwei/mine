#coding:utf-8
'''
计算特征：
=====10个
['pos',
'QIAN880131', 'BRYS930101', 'VENM980101', 'BASU010101','TOBD000101', 'MICC010101',
 'num_Ji',
 'front_dps',
 'LR']

=====20个
['pos',
 'DESM900101', 'QIAN880128', 'QIAN880131', 'DOSZ010102','BRYS930101', 'VENM980101', 'BASU010101', 'TOBD000101', 'TOBD000102','BONM030105', 'MICC010101', 'SIMK990104', 'ZHAC000103',
'num_L','num_Ji',
 'front_dps',
 'idr',
 'LR',
 'elm_mod']
'''
from collections import defaultdict

import pandas as pd
import os
import sys
import numpy as np
from numpy import sort
import copy
import math
import cmath
import requests
import io
import json
import jsonpath

import logging

from pandas import DataFrame

from idr_pred import config,logconfig


logconfig.setup_logging()
log = logging.getLogger("idrPred.extract_feature")
# constant
a_list = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
aa_list = [i + j for i in a_list for j in a_list]
c_list = ('A', 'T', 'C', 'G')
cc_list = [i + j for i in c_list for j in c_list]
log.debug("aaindex path: %s", os.path.abspath(config.aa_index_path))
aaindex = pd.read_csv(config.aa_index_path, sep="\t", header=None, names=['name'] + aa_list, index_col='name')
aaindex = aaindex.T

# 计算LR
def get_GO(id):
    requestURL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={}".format(id)

    r = requests.get(requestURL, headers={"Accept": "application/json"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()

    responseBody = r.text
    responseBody = json.loads(responseBody)
    gos = jsonpath.jsonpath(responseBody, '$..goId')
    if gos!=False:
        #查找祖先
        ancestor = pd.read_csv(config.ancestor_path, sep=',', header=None, low_memory=False).set_index(0)

        l = []
        for i in gos:
            if i in ancestor.index:
                l = l + ancestor.loc[i, :].tolist()
            l.append(i)
        l = list(set(l))
        if np.nan in l:
            l.remove(np.nan)
        if 'all' in l:
            l.remove('all')
        gos = ','.join(l)

    res = {"GO": gos}
    return res

# 根据cv训练集计算所有的LR值

    # 根据cv训练集计算所有的LR值
def calculate_LR(df1):
    #     def calculate_LR(df1,df2):
    """
    df1:cv training set
    df2:cv test set
    """
    # log ((2+c)/(1+c)) + log ((2+c)/ (1+c)), {c==1}

    # 有害和中性注释的字典
    p = {}
    n = {}
    for index, row in df1.iterrows():
        if (pd.isna(row['GO'])):
            continue
        for i in row['GO'].split(';'):
            if i not in p.keys():
                p[i] = 1
                n[i] = 1
            if (row['ClinSigSimple'] == 1):
                p[i] += 1
            else:
                n[i] += 1
    #     for index,row in df2.iterrows():
    #         if(pd.isna(row['GO'])):
    #             continue
    #         for i in row['GO'].split(','):
    #             if i not in p.keys():
    #                 p[i]=1
    #                 n[i]=1
    #             if(row['ClinSigSimple']==1):
    #                 p[i]+=1
    #             else:
    #                 n[i]+=1

    l = copy.deepcopy(p)
    for i in l.keys():
        l[i] = math.log(p[i] / n[i])


    # 求和计算每个蛋白的lr
    def LR_add(x):
        sum = 0
        if (pd.isna(x)):
            return sum
        for i in x.split(';'):
            sum = sum + l[i]
        return sum

    df1['LR_score'] = df1['GO'].apply(lambda x: LR_add(x))
    #     df2['LR'] = df2['GO'].apply(lambda x:LR_add(x))
    df1 = df1.drop(columns=['GO'])
    #     df2 = df2.drop(columns=['GO'])
    return df1


# 根据cv训练集计算所有的LR值
def calculate_PA(df1, df2):
    """
       df1:cv training set
       df2:cv test set
       """
    # log ((2+c)/(1+c)) + log ((2+c)/ (1+c)), {c==1}

    # 有害和中性注释的字典
    p = {}
    n = {}
    for index, row in df1.iterrows():
        if (pd.isna(row['site'])):
            continue
        for i in row['site'].split(','):
            if i != '':
                if i not in p.keys():
                    p[i] = 1
                    n[i] = 1
                if (row['ClinSigSimple'] == 1):
                    p[i] += 1
                else:
                    n[i] += 1
    #     for index,row in df2.iterrows():
    #         if(pd.isna(row['site'])):
    #             continue
    #         for i in row['site'].split(','):
    #             if i!='':
    #                 if i not in p.keys():
    #                     p[i]=1
    #                     n[i]=1
    #                 if(row['ClinSigSimple']==1):
    #                     p[i]+=1
    #                 else:
    #                     n[i]+=1

    s = copy.deepcopy(p)
    for i in s.keys():
        s[i] = math.log(p[i] / n[i])
    s

    # 求和计算每个蛋白的pa
    def PA_add(x):
        sum = 0
        if (pd.isna(x)):
            return sum
        for i in x.split(','):
            if i != '':
                sum = sum + s[i]
        return sum

    df1['PA_score'] = df1['site'].apply(lambda x: PA_add(x))
    #     df2['PA'] = df2['site'].apply(lambda x:PA_add(x))
    df1 = df1.drop(columns=['site'])
    #     df2 = df2.drop(columns=['site'])
    return df1
def check_aa(seq, aa):
    seq = "".join(seq.split())
    aaf = aa[0]  # from-wild
    aai = int(str(aa[1:-1]).strip('\n')) # index-pos
    aat = aa[-1]  # to-mut
    return seq, aaf, aat, aai

#record 错误信息记录：
def msg_find(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    msg = ""
    if aaf not in a_list:
        msg = "aa error, origin of aa is invalid."
    if aat not in a_list:
        msg = "aa error, nutation of aa is invalid."
    if aai < 1 or aai > len(seq):
        msg = "aa error, index of aa is invalid."
    if seq[aai - 1] != aaf:
        msg = "aa error, seq[{}] = {}, but origin of aa = {}".format(aai, seq[aai - 1], aaf)
    return msg

# aaindex
def get_aaindex(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    res = aaindex.loc["{}{}".format(aaf, aat), :]
    # 10个特征时所需的aaindex
    print(res['ANDN920101'],res['QIAN880131'], res['BRYS930101'], res['VENM980101'], res['BASU010101'],res['TOBD000101'], res['MICC010101'])
    # 20个特征时所需的aaindex
    print(res['DESM900101'], res['QIAN880128'], res['QIAN880131'], res['DOSZ010102'],res['BRYS930101'], res['VENM980101'], res['BASU010101'], res['TOBD000101'], res['TOBD000102'],res['BONM030105'], res['MICC010101'], res['SIMK990104'], res['ZHAC000103'])
    return res.to_dict()

# first
def get_pos_1(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    pos_1=0
    if aai==1:
        pos_1=1
    res = {"pos_1": pos_1}
    return res

# neighborhood
def get_neighborhood_features(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)

    def find_win(seq, aai):
        # 确定边界
        index = aai - 1
        front = 0 if index - 11 < 0 else index - 11
        after = len(seq) if index + 12 > len(seq) - 1 else index + 12
        return seq[front: after]

    def get_count_a(win):
        """ count number of aa in windows"""
        a_dict = defaultdict(int)
        for i in win:
            a_dict[i] += 1  # 递增
        return {'num_' + i: a_dict[i] for i in a_list}

    win = find_win(seq, aai)
    count_a = get_count_a(win)
    nei_feature = count_a
    # 1.NonPolarAA:Number of nonpolar neighborhood residues
    nei_feature['num_noPolar'] = nei_feature['num_' + a_list[0]] + nei_feature['num_' + a_list[4]] + nei_feature[
        'num_' + a_list[5]] + nei_feature['num_' + a_list[7]] + nei_feature['num_' + a_list[9]] + nei_feature[
                                    'num_' + a_list[10]] + nei_feature['num_' + a_list[12]] + nei_feature[
                                    'num_' + a_list[17]] + nei_feature['num_' + a_list[18]] + nei_feature[
                                    'num_' + a_list[19]]
    # 2.PolarAA:Number of polar neighborhood residues
    nei_feature['num_Polar'] = nei_feature['num_' + a_list[1]] + nei_feature['num_' + a_list[11]] + nei_feature[
        'num_' + a_list[13]] + nei_feature['num_' + a_list[15]] + nei_feature['num_' + a_list[16]]
    # 3.ChargedAA:Number of charged neighborhood residues
    nei_feature['num_Charged'] = nei_feature['num_' + a_list[2]] + nei_feature['num_' + a_list[3]] + nei_feature[
        'num_' + a_list[6]] + nei_feature['num_' + a_list[8]] + nei_feature['num_' + a_list[14]]
    # 4.PosAA:Number of Positive charged neighborhood residues
    nei_feature['num_Pos'] = nei_feature['num_' + a_list[2]] + nei_feature['num_' + a_list[3]]
    # 5.NegAA:Number of Negative charged neighborhood residues
    nei_feature['num_Neg'] = nei_feature['num_' + a_list[6]] + nei_feature['num_' + a_list[8]] + nei_feature[
        'num_' + a_list[14]]
    print(nei_feature['num_L'],nei_feature['num_Polar'])
    return nei_feature

# Dipeptide features
def get_dps(seq,aa):
        import numpy as np
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        seq, aaf, aat, aai = check_aa(seq, aa)
        s = r"""
        library('protr')
        library('DT')
        extractCTD = function (x) c(extractCTDC(x), extractCTDT(x), extractCTDD(x))

         funcdict   = c(
        'dc'     = 'extractDC')

        fs = function(path){

         seq <- scan(textConnection(path), what = 'complex', blank.lines.skip = TRUE)

         aaa <- c("dc")

         exec = paste0('t(sapply(seq, ', funcdict[as.character(aaa)], '))')
         outlist = vector('list', length(exec))
         n = length(exec)
         for (i in 1L:n) {
           outlist[[i]] = eval(parse(text = exec[i]))
         }

         out = do.call(cbind, outlist)
         return(out)
        }
        """
        robjects.r(s)

        r = robjects.r["fs"](seq)

        # print(r)
        # print(type(r))
        rr = pandas2ri.rpy2py_floatvector(r) #<class 'numpy.ndarray'>
        # print(type(rr))
        feature_name = ['AA','RA','NA','DA','CA','EA','QA','GA','HA','IA','LA','KA','MA','FA','PA','SA','TA','WA','YA','VA','AR','RR','NR','DR','CR','ER','QR','GR','HR','IR','LR','KR','MR','FR','PR','SR','TR','WR','YR','VR','AN','RN','NN','DN','CN','EN','QN','GN','HN','IN','LN','KN','MN','FN','PN','SN','TN','WN','YN','VN','AD','RD','ND','DD','CD','ED','QD','GD','HD','ID','LD','KD','MD','FD','PD','SD','TD','WD','YD','VD','AC','RC','NC','DC','CC','EC','QC','GC','HC','IC','LC','KC','MC','FC','PC','SC','TC','WC','YC','VC','AE','RE','NE','DE','CE','EE','QE','GE','HE','IE','LE','KE','ME','FE','PE','SE','TE','WE','YE','VE','AQ','RQ','NQ','DQ','CQ','EQ','QQ','GQ','HQ','IQ','LQ','KQ','MQ','FQ','PQ','SQ','TQ','WQ','YQ','VQ','AG','RG','NG','DG','CG','EG','QG','GG','HG','IG','LG','KG','MG','FG','PG','SG','TG','WG','YG','VG','AH','RH','NH','DH','CH','EH','QH','GH','HH','IH','LH','KH','MH','FH','PH','SH','TH','WH','YH','VH','AI','RI','NI','DI','CI','EI','QI','GI','HI','II','LI','KI','MI','FI','PI','SI','TI','WI','YI','VI','AL','RL','NL','DL','CL','EL','QL','GL','HL','IL','LL','KL','ML','FL','PL','SL','TL','WL','YL','VL','AK','RK','NK','DK','CK','EK','QK','GK','HK','IK','LK','KK','MK','FK','PK','SK','TK','WK','YK','VK','AM','RM','NM','DM','CM','EM','QM','GM','HM','IM','LM','KM','MM','FM','PM','SM','TM','WM','YM','VM','AF','RF','NF','DF','CF','EF','QF','GF','HF','IF','LF','KF','MF','FF','PF','SF','TF','WF','YF','VF','AP','RP','NP','DP','CP','EP','QP','GP','HP','IP','LP','KP','MP','FP','PP','SP','TP','WP','YP','VP','AS','RS','NS','DS','CS','ES','QS','GS','HS','IS','LS','KS','MS','FS','PS','SS','TS','WS','YS','VS','AT','RT','NT','DT','CT','ET','QT','GT','HT','IT','LT','KT','MT','FT','PT','ST','TT','WT','YT','VT','AW','RW','NW','DW','CW','EW','QW','GW','HW','IW','LW','KW','MW','FW','PW','SW','TW','WW','YW','VW','AY','RY','NY','DY','CY','EY','QY','GY','HY','IY','LY','KY','MY','FY','PY','SY','TY','WY','YY','VY','AV','RV','NV','DV','CV','EV','QV','GV','HV','IV','LV','KV','MV','FV','PV','SV','TV','WV','YV','VV']

        result = {feature_name[i]: list(rr[0])[i] for i in range(len(feature_name))}
        return result

# length
def get_length(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    res = {"len": len(seq)}
    return res

# idr
def get_idr(uid,seq,aa):
    print(uid,seq)
    sum_len=0
    seq, aaf, aat, aai = check_aa(seq, aa)
    length =len(seq)
    data = pd.read_csv('data/dis_uni.csv')
    # data = pd.read_csv('idr_pred/data/dis_uni.csv')
    # data = pd.read_csv(r'D:\anaconda\envs\pytorch1\IDR-web\idr\data\dis_uni.csv')
    for i in range(len(data)):
        # print(uid)
        # print(data['uniprot'][i])
        if data['uniprot'][i] == uid:
            # print(data['uniprot'][i] , str(uid))
            idrs = data['value'][i]
            sp = str(idrs).split(';')
            count = len(sp)
            for j in range(count):
                # print(len(sp[j].split('-')),sp[j])
                idr_len = int(sp[j].split('-')[1]) - int(sp[j].split('-')[0]) + 1
                sum_len += idr_len
                print(int(sp[j].split('-')[1]), int(sp[j].split('-')[0]))
            # print(uniprot[i], sum_len)
            num = round(sum_len / length, 3)
            # print('idr',num)
            break
        else: num=0
    res = {"idr": num}
    return res

# slim
def get_slim(uid,seq,aa):

    seq, aaf, aat, aai = check_aa(seq, aa)
    # data = pd.read_csv('idr_pred/data/elm.csv')
    data = pd.read_csv('data/elm.csv')
    # data = pd.read_csv('disprot/data/2022.6/result/feature/mangce_93_all_lr_la_ELM.csv')
    uniprot = data['uniprot']
    # 'elm_CLV','elm_DEG','elm_DOC','elm_LIG','elm_MOD','elm_TRG'
    clv = data['CLV']
    deg = data['DEG']
    doc = data['DOC']
    lig = data['LIG']
    mod = data['MOD']
    trg = data['TRG']
    length = len(seq)
    sum_len = 0
    idr_len = 0
    a = []
    elm_name = ['clv','deg','doc','lig','mod','trg']
    elm_class = [data['CLV'],data['DEG'],data['DOC'],data['LIG'],data['MOD'],data['TRG']]
    i=1
    for i in range(len(data)):
        if uid == data['uniprot'][i]:
            a=[]

            for elm in elm_class:

                # print('{{{{{{{{{{{{{{{{',i,'}}}}}}}}}}}}}}}}}}}}}')

                if elm[i] != '+':
                    sp = str(elm[i]).split(';')
                    count = len(sp)
                    for j in range(count):
                        # print(len(sp[j].split('-')),sp[j])
                        elm_len = int(sp[j].split('-')[1]) - int(sp[j].split('-')[0]) + 1
                        sum_len += elm_len
                        # print(int(sp[j].split('-')[1]), int(sp[j].split('-')[0]))
                    # print(uniprot[i], sum_len)
                    num = round(sum_len / length, 3)

                    # print(uniprot[i]+','+str(num),length)
                    s = uniprot[i] + ',' + str(num) + '\n'
                    a.append(num)
                    # path = 'ing/SLiM/SUM/mangce_93/'+name
                    # with open(path, 'a') as f:
                    #     f.writelines(s)
                    sum_len = 0
                    # print('1',a)
                elif elm[i] == '+' :
                    # print(elm[i])
                    num=0
                    # print(uniprot[i]+','+'0')
                    a.append(num)
                    # print('2',a)

                        # path = 'ing/SLiM/SUM/mangce_93/' + name
                        # s = uniprot[i] + ',' + '0' + '\n'
                        # with open(path, 'a') as f:
                        #     f.writelines(s)
                # i+=1
            res = {'elm_clv': a[0], 'elm_deg': a[1], 'elm_doc': a[2], 'elm_lig': a[3], 'elm_mod': a[4],
                   'elm_trg': a[5]}
            return res
        else:
            a=[0,0,0.1,0.3,0,0]
            # print('3',a)

    res = {'elm_clv':a[0],'elm_deg':a[1],'elm_doc':a[2],'elm_lig':a[3],'elm_mod':a[4],'elm_trg':a[5]}
    return res

def get_Site_1(id):
    if '-' in str(id):
        rawData = 'None'
        return rawData
    site=''
    if not os.path.exists('../Site'):
        os.makedirs('../Site')
    url = "https://www.uniprot.org/uniprot/{}.gff".format(id)
    urlData = requests.get(url).content
    rawData = pd.read_csv(io.StringIO('\n'.join(urlData.decode('utf-8').split('\n')[2:])), sep="\t", header=None,
                          names=['ID', 'database', 'site', 'from', 'to', '1', '2', '3', '4', '5'])
    return rawData

def get_Site_2(rawData, seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    l = []
    # print(type(rawData))
    if str(type(rawData)) == "<class 'str'>":
        res = {"site": 'None'}
        return res
    for indexs2 in rawData.index:
        if rawData.loc[indexs2]['from'] <= aai and rawData.loc[indexs2]['to'] >= aai:
            l.append(rawData.loc[indexs2, 'site'])
    l = list(set(l))
    site = ','.join(l)
    res = {"site": site}
    return res

#pos
def get_residue(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    res = {"pos": aai}
    return res


def get_nutationAll(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    nutationAll = pd.DataFrame(columns=[i[0] + '_' + i[1] for i in aa_list], index=aa_list).fillna(0)
    np.fill_diagonal(nutationAll.values, 1)
    return nutationAll.loc[[aaf+aat]].to_dict(orient='records')[0]

def get_groupAll(seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    group = {
        'VILFMWYC': ['V', 'I', 'L', 'F', 'M', 'W', 'Y', 'C'],
        'DE': ['D', 'E'],
        'RKH': ['R', 'K', 'H'],
        'GP': ['G', 'P'],
        'NQS': ['N', 'Q', 'S'],
        'AT': ['A', 'T']
    }

    # 颠倒键值对用于映射
    group_r = {}
    for k, v in group.items():
        for i in v:
            group_r[i] = k
    # 全对角矩阵
    groupAll = pd.DataFrame(
        index=[i + '_' + j for i in group.keys() for j in group.keys()],
        columns=[i + '_' + j for i in group.keys() for j in group.keys()],
    ).fillna(0)
    np.fill_diagonal(groupAll.values, 1)
    return groupAll.loc[[group_r[aaf]+'_'+group_r[aat]]].to_dict(orient='records')[0]

def get_sift4g_file(id, seq):
    with open('seq.fa', 'w') as file_object:
        file_object.write('>'+id+'\n')
        file_object.write(seq+'\n\n')
    if not os.path.exists('../sift_out_file'):
        os.makedirs('../sift_out_file')
    os.system('sift4g -q ./seq.fa -d ../sift4g/uniprot_sprot.fasta --out ./sift_out_file/')
    filename = 'sift_out_file/{}.SIFTprediction'.format(id)
    # filename = 'idr_pred/sift_out_file/{}.SIFTprediction'.format(id)
    # filename = './sift_out_file/{}.SIFTprediction'.format(id)
    filename1 = 'sift_out_file/{}.SIFTprediction1'.format(id)
    # filename1 = 'idr_pred/sift_out_file/{}.SIFTprediction1'.format(id)
    fin = open(filename, 'r')
    # fin = open(filename, 'r')
    a = fin.readlines()
    for i in a:
        i = i.strip()
    fout = open(filename1, 'w')
    # fout = open(filename1, 'w')
    l = a[5:]
    l.insert(0, 'A  B  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  X  Y  Z  *  -\n')
    b = ''.join(l)
    fout.write(b)

    sift = pd.read_csv(filename1,sep='  ')
    # print('sift',sift)
    # sift[aat][aai+1]
    return sift

def get_sift4g_hits(id, seq, aa):
    seq, aaf, aat, aai = check_aa(seq, aa)
    with open('seq.fa', 'w') as file_object:
        file_object.write('>'+id+'\n')
        file_object.write(seq+'\n\n')
    if not os.path.exists('../subst'):
        os.makedirs('../subst')
    if not os.path.exists('../sift_out'):
        os.makedirs('../sift_out')
    with open('../subst/{}.subst'.format(id), 'w') as file_object:
        file_object.write(aa)
    os.system('sift4g -q ./seq.fa --subst ../subst/ -d ../sift4g/uniprot_sprot.fasta --out ../sift_out/')
    files = ['../sift_out/{}.SIFTprediction'.format(id)]

    sift = []
    for file in files:
        with open(file) as f:
            # 读取文件
            tmp = f.read()
            tmp = tmp.strip().split('\n')  # 每行分割
            # gi序号
        (filepath, tempfilename) = os.path.split(file)
        (filename, extension) = os.path.splitext(tempfilename)
        #     组成json
        # print('tmp',tmp)
        for i in tmp:

            tmp_1 = []
            if '\t' in i:
                tmp_1 = i.split('\t')
                # print('tmp_1',tmp_1)
                sift.append({
                    'swissMatch': int(tmp_1[5]),
                    'sift4g': float(tmp_1[2])
                }
                )
            # print('sift',sift)
            # print('score',float(tmp_1[2]))
    return sift[0]
    # return sift[0]['swissMatch']


def get_all_features(n,seq, aa):
# def get_all_features(n, seq, aa):
    # print('收集特征')
        li = []
    # err_list = []

        # print('有GO')
        flag='*'
        GOs=''
        Sites=''
        sift_hits=0
        sift=None


        # print(len(n))
        for i in range(len(n)):
                id_ = n[i]
                seq_ = seq[i]
                aa_ = aa[i]

                # print('=',aa_)
                features = {}
                features.update({"id" : id_})
                # features.update({"msg": msg_find(seq_, aa_)})
            # if msg_find(seq_, aa_)=="":
                if id_ != flag:
                    # sift_hits = get_sift4g_hits(id_, seq_, aa_)
                    # sift = get_sift4g_file(id_, seq_)
                    GOs = get_GO(id_)

                    Sites = get_Site_1(id_)
                    flag = id_
                seq1, aaf1, aat1, aai1 = check_aa(seq_, aa_)
                # print("hits:{}".format(sift_hits),'score:{}'.format(sift[aat1][aai1-1]))

                # 3 突变位点 野生氨基酸 突变氨基酸
                # features.update(get_residue(seq_, aa_))
                #
                # # 1 长度 1
                # features.update(get_length(seq_, aa_))
                # # 617 aaindex 617
                # features.update(get_aaindex(seq_, aa_))
                # # 400 氨基酸之间的突变如A_C 400个
                # features.update(get_nutationAll(seq_, aa_))
                # # 36  物化性质组的特征
                # features.update(get_groupAll(seq_, aa_))
                # # 25 领域特征
                # features.update(get_neighborhood_features(seq_, aa_))
                # 2 进化分数（命中数+sift）
                # features.update({'sift4g':sift[aat1][aai1-1]})
                # features.update({"swissMatch": sift_hits,'sift4g':sift[aat1][aai1-1]})
                # 1 注释特征GO
                features.update(GOs)
                # # 1 是否首位突变
                # features.update(get_pos_1(seq_, aa_))
                # # 1 位点特征site
                features.update(get_Site_2(Sites, seq_, aa_))
                # # 1 无序区域特征
                # features.update(get_idr(id_,seq_,aa_))
                # # 6 短线性基序
                # features.update(get_slim(id_,seq_,aa_))
                # # 2 二肽特征
                # features.update(get_dps(seq_,aa_))

                # c = {'ClinSigSimple': clin[i]}
                # # isdel = DataFrame(c)
                # # df2 = pd.concat([df2,isdel],axis=1)
                # features.update(c)
                df_features = pd.DataFrame([features])
                li.append(df_features)

            # else:
            #     df_features = pd.DataFrame([features])
            #     # err_list.append(df_features)
            #     li.append(df_features)

        df2 = pd.concat(li)
        df2 = df2.reset_index()
        del df2['index']


        df2.to_csv(R'E:\pythonProject\untitled\2023.9.16\data\PON_ALL_20230916_GO_SITE.csv',mode='w', index=False)
        # df2.to_csv('all_features_2.csv',mode='w', index=False)

        return df2

#
if __name__ == '__main__':
    # data1 = pd.read_csv(r'E:\pythonProject\untitled\predict_对比\NON-IDR\ClinVar\data\clinvar.csv')
    # seq = pd.read_csv(r"E:\pythonProject\untitled\out\2023-01-31=18-21-43.csv")
    data1 = pd.read_csv(r'E:\pythonProject\untitled\predict_对比\NON-IDR\2022.1117\all_400wild_lr_pa_elm.csv')
    # data1 = pd.read_csv(r'E:\pythonProject\untitled\2023.9.16\data\blind_20230916.csv')
    # data1 = pd.read_csv(r'E:\pythonProject\untitled\2023.8.28\data\Mauno_test_2_uid+seq.csv')
    # seq = pd.read_csv(r"E:\pythonProject\untitled\2023.8.28\data\Mauno_test_seq.csv")
    uid = data1['uniprot']
    # wild = data1['wild']
    mut = data1['variation']
    # pos = data1['pos']
    seq = data1['uniprot']
    # seq = data1['seq']
    get_all_features(uid, seq, mut)
    n=0
    # ClinSigSimple = data1['ClinSigSimple']
    # for i in range(len(data1)):
    #     get_all_features(uid, seq, mut)
    #

    #     aa = str(wild[i])+str(pos[i])+str(mut[i])
    #     fasta = pd.read_csv(r'E:\pythonProject\untitled\out\train_1008.csv')
    #
    #     # print(uid[i],fasta['uniprot'][i])
    #     # print(uid[i]==fasta['uniprot'][i])
    #     if uid[i] == fasta['uniprot'][i]:
    #         seq = str(fasta['simple_fasta'][i])
    #
    #         df_go = get_Site_2(get_Site_1(uid[i]),seq,aa)
    # # with open(r'E:\pythonProject\IDRPred\idr_pred\data\train_go.csv','a') as f:
    # #     s=str(df_go) +'\n'
    # #     f.writelines(df_go)
    #         print(uid[i],'=',df_go)
    #
    #     else:
    #         print('no',uid[i])
    #     n+=1
    # print('n:',n)
    # print(get_all_features(uid,seq,mut))
    # print(get_idr(uid,seq,mes))


