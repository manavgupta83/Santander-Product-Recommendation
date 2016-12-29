
# coding: utf-8

# In[469]:

import pandas as pd
import time
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import datetime

# setting pandas env variables to display max rows and columns
pd.set_option('display.max_columns',1000) 
pd.set_option('display.max_rows',1000)


# In[470]:

##################################### Feature Translation
# fecha_dato	The table is partitioned for this column
# ncodpers	Customer code
# ind_empleado	Employee index: A active, B ex employed, F filial, N not employee, P pasive
# pais_residencia	Customer's Country residence
# sexo	Customer's sex
# age	Age
# fecha_alta	The date in which the customer became as the first holder of a contract in the bank
# ind_nuevo	New customer Index. 1 if the customer registered in the last 6 months.
# antiguedad	Customer seniority (in months)
# indrel	1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
# ult_fec_cli_1t	Last date as primary customer (if he isn't at the end of the month)
# indrel_1mes	Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential)
#                 ,3 (former primary), 4(former co-owner)
# tiprel_1mes	Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
# indresi	Residence index (S (Yes) or N (No) if the residence country is the same than the bank country)
# indext	Foreigner index (S (Yes) or N (No) if the customer's birth country is different than the bank country)
# conyuemp	Spouse index. 1 if the customer is spouse of an employee
# canal_entrada	channel used by the customer to join
# indfall	Deceased index. N/S
# tipodom	Addres type. 1, primary address
# cod_prov	Province code (customer's address)
# nomprov	Province name
# ind_actividad_cliente	Activity index (1, active customer; 0, inactive customer)
# renta	Gross income of the household
# segmento	segmentation: 01 - VIP, 02 - Individuals 03 - college graduated
# ind_ahor_fin_ult1	Saving Account
# ind_aval_fin_ult1	Guarantees
# ind_cco_fin_ult1	Current Accounts
# ind_cder_fin_ult1	Derivada Account
# ind_cno_fin_ult1	Payroll Account
# ind_ctju_fin_ult1	Junior Account
# ind_ctma_fin_ult1	Más particular Account
# ind_ctop_fin_ult1	particular Account
# ind_ctpp_fin_ult1	particular Plus Account
# ind_deco_fin_ult1	Short-term deposits
# ind_deme_fin_ult1	Medium-term deposits
# ind_dela_fin_ult1	Long-term deposits
# ind_ecue_fin_ult1	e-account
# ind_fond_fin_ult1	Funds
# ind_hip_fin_ult1	Mortgage
# ind_plan_fin_ult1	Pensions
# ind_pres_fin_ult1	Loans
# ind_reca_fin_ult1	Taxes
# ind_tjcr_fin_ult1	Credit Card
# ind_valo_fin_ult1	Securities
# ind_viv_fin_ult1	Home Account
# ind_nomina_ult1	Payroll
# ind_nom_pens_ult1	Pensions
# ind_recibo_ult1	Direct Debit

train_data = pd.read_csv('train_ver2.csv.zip', compression='zip',header=0, sep=',', quotechar='"')
test_data = pd.read_csv('test_ver2.csv.zip', compression='zip',header=0, sep=',', quotechar='"')



# In[608]:

##### I want to train model on June 2015 data to finally make predictions for June 2016 data
train_data_final = train_data[train_data.fecha_dato.isin(['2015-05-28', '2015-06-28'])]

train_data_final1 = train_data[train_data.fecha_dato.isin(['2016-05-28'])]
test_data_final = test_data[test_data.fecha_dato.isin(['2016-06-28'])]

test_data_final = train_data_final1.append(test_data_final)

del train_data_final1

list_of_preds = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1'
                 ,'ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1'
                 ,'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1'
                 ,'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1'
                 ,'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']


preds_map = {'ind_cco_fin_ult1' : 0,'ind_cder_fin_ult1' : 1,'ind_cno_fin_ult1' : 2,'ind_ctju_fin_ult1' : 3
             ,'ind_ctma_fin_ult1' : 4,'ind_ctop_fin_ult1' : 5,'ind_ctpp_fin_ult1' : 6,'ind_deco_fin_ult1' : 7
             ,'ind_deme_fin_ult1' : 8,'ind_dela_fin_ult1' : 9,'ind_ecue_fin_ult1' : 10,'ind_fond_fin_ult1' : 11
             ,'ind_hip_fin_ult1' : 12,'ind_plan_fin_ult1' : 13,'ind_pres_fin_ult1' : 14,'ind_reca_fin_ult1' : 15
             ,'ind_tjcr_fin_ult1' : 16,'ind_valo_fin_ult1' : 17,'ind_viv_fin_ult1' : 18,'ind_nomina_ult1' : 19
             ,'ind_nom_pens_ult1' : 20,'ind_recibo_ult1' : 21}


train_data_final = train_data_final.sort_values(by = ['ncodpers','fecha_dato'], ascending = True)
test_data_final = test_data_final.sort_values(by = ['ncodpers','fecha_dato'], ascending = True)

for cols in list_of_preds:

    ######################################## TRAINING DATA ################################################
    #get the flag if the product was present in previous month
    train_data_final[cols+str('_added')] = train_data_final[cols].diff().fillna(0).astype(int)

    #flag if the current observation is for new customer
    train_data_final['new_cust'] = (train_data_final.ncodpers == train_data_final.ncodpers.shift(1)).astype(int)
    
    #if current observation is new customer, then the '_added' flag to be set to 0
    train_data_final.loc[train_data_final['new_cust'] == 0, [cols+str('_added')]] = 0

    #only assign positive flag to products which were added since last month
    train_data_final.loc[train_data_final[cols+str('_added')] < 0, [cols+str('_added')]] = 0

    if cols == list_of_preds[0]:
        train_data_final['total_additions'] = train_data_final[cols+str('_added')]
    else:
        train_data_final['total_additions'] = train_data_final['total_additions'] + train_data_final[cols+str('_added')]

    ####assign May month values to June for the target columns
    train_data_final[cols+str('_prev')] = train_data_final[cols].shift(1).fillna(0).astype(int)

     # for the customers without May values, we need to re-assign these numbers as 0
    train_data_final.loc[(train_data_final['new_cust'] == 0) & (train_data_final['fecha_dato'] == '2015-06-28'), [cols+str('_prev')]] = 0
    

    ######################################## TEST DATA ################################################

    #flag if the current observation is for new customer
    test_data_final['new_cust'] = (test_data_final.ncodpers == test_data_final.ncodpers.shift(1)).astype(int)
    
    ####assign May month values to June for the target columns
    test_data_final[cols+str('_prev')] = test_data_final[cols].shift(1).fillna(0).astype(int)

    # for the customers without May values, we need to re-assign these numbers as 0
    test_data_final.loc[(test_data_final['new_cust'] == 0) & (test_data_final['fecha_dato'] == '2016-06-28'), [cols+str('_prev')]] = 0

#keep the data for only month of June and only for customers who added a new product in June for training data
train_data_final = train_data_final[(train_data_final['fecha_dato'] == '2015-06-28') & (train_data_final['total_additions'] > 0)]

####keep only JUne 2016 data for test data
test_data_final = test_data_final[(test_data_final['fecha_dato'] == '2016-06-28')]

train_data_final.drop(list_of_preds,axis = 1,inplace = True)
test_data_final.drop(list_of_preds,axis = 1,inplace = True)

train_data_final.drop(['total_additions','new_cust'],axis = 1,inplace = True)
test_data_final.drop(['new_cust'],axis = 1,inplace = True)

train_data_final.rename(columns = lambda x : x.replace('_added',''), inplace = True)


all_vars = train_data_final.columns.values

keep_vars = list(set(all_vars) - set(list_of_preds))

###this is where we create duplicate rows for each customer; each duplicate row corresponds to products bought by customer
train_data_final = pd.melt(train_data_final, id_vars= keep_vars, var_name='Product', value_name='Value')

train_data_final = train_data_final.sort_values(by = ['ncodpers','fecha_dato'], ascending = True)

train_data_final = train_data_final[train_data_final['Value'] > 0 ]

#this is because no one added 'ind_ahor_fin_ult1','ind_aval_fin_ult1' in the month of June 2015
list_of_preds = list_of_preds[2:]

train_data_final['product_seq'] = train_data_final['Product'].map(lambda x:preds_map[x]).astype(int)

train_data_final.drop(['Value','Product'], axis = 1, inplace = True)


canal_dict = {'KAI': 35,'KBG': 17,'KGU': 149,'KDE': 47,'KAJ': 41,'KCG': 59,
 'KHM': 12,'KAL': 74,'KFH': 140,'KCT': 112,'KBJ': 133,'KBL': 88,'KHQ': 157,'KFB': 146,'KFV': 48,'KFC': 4,
 'KCK': 52,'KAN': 110,'KES': 68,'KCB': 78,'KBS': 118,'KDP': 103,'KDD': 113,'KBX': 116,'KCM': 82,
 'KAE': 30,'KAB': 28,'KFG': 27,'KDA': 63,'KBV': 100,'KBD': 109,'KBW': 114,'KGN': 11,
 'KCP': 129,'KAK': 51,'KAR': 32,'KHK': 10,'KDS': 124,'KEY': 93,'KFU': 36,'KBY': 111,
 'KEK': 145,'KCX': 120,'KDQ': 80,'K00': 50,'KCC': 29,'KCN': 81,'KDZ': 99,'KDR': 56,
 'KBE': 119,'KFN': 42,'KEC': 66,'KDM': 130,'KBP': 121,'KAU': 142,'KDU': 79,
 'KCH': 84,'KHF': 19,'KCR': 153,'KBH': 90,'KEA': 89,'KEM': 155,'KGY': 44,'KBM': 135,
 'KEW': 98,'KDB': 117,'KHD': 2,'RED': 8,'KBN': 122,'KDY': 61,'KDI': 150,'KEU': 72,
 'KCA': 73,'KAH': 31,'KAO': 94,'KAZ': 7,'004': 83,'KEJ': 95,'KBQ': 62,'KEZ': 108,
 'KCI': 65,'KGW': 147,'KFJ': 33,'KCF': 105,'KFT': 92,'KED': 143,'KAT': 5,'KDL': 158,
 'KFA': 3,'KCO': 104,'KEO': 96,'KBZ': 67,'KHA': 22,'KDX': 69,'KDO': 60,'KAF': 23,'KAW': 76,
 'KAG': 26,'KAM': 107,'KEL': 125,'KEH': 15,'KAQ': 37,'KFD': 25,'KEQ': 138,'KEN': 137,
 'KFS': 38,'KBB': 131,'KCE': 86,'KAP': 46,'KAC': 57,'KBO': 64,'KHR': 161,'KFF': 45,
 'KEE': 152,'KHL': 0,'007': 71,'KDG': 126,'025': 159,'KGX': 24,'KEI': 97,'KBF': 102,
 'KEG': 136,'KFP': 40,'KDF': 127,'KCJ': 156,'KFR': 144,'KDW': 132,-1: 6,'KAD': 16,
 'KBU': 55,'KCU': 115,'KAA': 39,'KEF': 128,'KAY': 54,'KGC': 18,'KAV': 139,'KDN': 151,
 'KCV': 106,'KCL': 53,'013': 49,'KDV': 91,'KFE': 148,'KCQ': 154,'KDH': 14,'KHN': 21,
 'KDT': 58,'KBR': 101,'KEB': 123,'KAS': 70,'KCD': 85,'KFL': 34,'KCS': 77,'KHO': 13,
 'KEV': 87,'KHE': 1,'KHC': 9,'KFK': 20,'KDC': 75,'KFM': 141,'KHP': 160,'KHS': 162,
 'KFI': 134,'KGV': 43}


pais_dict = {'LV': 102,'CA': 2,'GB': 9,'EC': 19,'BY': 64,'ML': 104,'MT': 118,
 'LU': 59,'GR': 39,'NI': 33,'BZ': 113,'QA': 58,'DE': 10,'AU': 63,'IN': 31,
 'GN': 98,'KE': 65,'HN': 22,'JM': 116,'SV': 53,'TH': 79,'IE': 5,'TN': 85,
 'PH': 91,'ET': 54,'AR': 13,'KR': 87,'GA': 45,'FR': 8,'SG': 66,'LB': 81,
 'MA': 38,'NZ': 93,'SK': 69,'CN': 28,'GI': 96,'PY': 51,'SA': 56,'PL': 30,
 'PE': 20,'GE': 78,'HR': 67,'CD': 112,'MM': 94,'MR': 48,'NG': 83,'HU': 106,
 'AO': 71,'NL': 7,'GM': 110,'DJ': 115,'ZA': 75,'OM': 100,'LT': 103,'MZ': 27,
 'VE': 14,'EE': 52,'CF': 109,'CL': 4,'SL': 97,'DO': 11,'PT': 26,'ES': 0,
 'CZ': 36,'AD': 35,'RO': 41,'TW': 29,'BA': 61,'IS': 107,'AT': 6,'ZW': 114,
 'TR': 70,'CO': 21,'PK': 84,'SE': 24,'AL': 25,'CU': 72,'UY': 77,'EG': 74,'CR': 32,
 'GQ': 73,'MK': 105,'KW': 92,'GT': 44,'CM': 55,'SN': 47,'KZ': 111,'DK': 76,
 'LY': 108,'AE': 37,'PA': 60,'UA': 49,'GW': 99,'TG': 86,'MX': 16,'KH': 95,
 'FI': 23,'NO': 46,'IT': 18,'GH': 88, 'JP': 82,'RU': 43,'PR': 40,'RS': 89,
 'DZ': 80,'MD': 68,-1: 1,'BG': 50,'CI': 57,'IL': 42,'VN': 90,'CH': 3,'US': 15,'HK': 34,
 'CG': 101,'BO': 62,'BR': 17,'BE': 12,'BM': 117}

nomprov_dict = {-1:'UNDEFINED','ALAVA':'ALAV','ALBACETE':'ALBA','ALICANTE':'ALIC','ALMERIA':'ALME','ASTURIAS':'ASTU'
                ,'AVILA':'AVIL','BADAJOZ':'BADA','BALEARS, ILLES':'BALE','BARCELONA':'BARC','BIZKAIA':'BIZK','BURGOS':'BURG'
                ,'CACERES':'CACE','CADIZ':'CADI','CANTABRIA':'CANT','CASTELLON':'CAST','CEUTA':'CEUT','CIUDAD REAL':'CIUD'
                ,'CORDOBA':'CORD','CORUÑA, A':'CORU','CUENCA':'CUEN','GIPUZKOA':'GIPU','GIRONA':'GIRO','GRANADA':'GRAN'
                ,'GUADALAJARA':'GUAD','HUELVA':'HUEL','HUESCA':'HUES','JAEN':'JAEN','LEON':'LEON','LERIDA':'LERI','LUGO':'LUGO'
                ,'MADRID':'MADR','MALAGA':'MALA','MELILLA':'MELI','MURCIA':'MURC','NAVARRA':'NAVA','OURENSE':'OURE'
                ,'PALENCIA':'PALE','PALMAS, LAS':'PALM','PONTEVEDRA':'PONT','RIOJA, LA':'RIOJ','SALAMANCA':'SALA'
                ,'SANTA CRUZ DE TENERIFE':'SANT','SEGOVIA':'SEGO','SEVILLA':'SEVI','SORIA':'SORI','TARRAGONA':'TARR'
                ,'TERUEL':'TERU','TOLEDO':'TOLE','VALENCIA':'VALE','VALLADOLID':'VALL','ZAMORA':'ZAMO','ZARAGOZA':'ZARA'}


emp_dict = {'N':1,-1:9999,'A':2,'B':3,'F':4,'S':5}
indfall_dict = {'N':1,-1:9999,'S':2}
sexo_dict = {'V':'V','H':'H',-1:'UNDEF'}
tiprel_dict = {'A':1,-1:9999,'I':2,'P':3,'N':4,'R':5}
indresi_dict = {'N':1,-1:9999,'S':2}
indext_dict = {'N':1,-1:9999,'S':2}
conyuemp_dict = {'N':1,-1:9999,'S':2}
segmento_dict = {-1:'UNDEF','01 - TOP':'VIP','02 - PARTICULARES': 'INDV','03 - UNIVERSITARIO':'GRAD'}



def processdata(df):
    
    df.replace(' NA', -2, inplace=True)
    df.replace('         NA', -2, inplace=True)
    df.replace('     NA', -2, inplace=True)
    df.fillna(-1, inplace=True)

    df['pais_residencia'] = df['pais_residencia'].map(lambda x: pais_dict[x]).astype(np.int8)
    df['canal_entrada'] = df['canal_entrada'].map(lambda x: canal_dict[x]).astype(np.int16)
    df['ind_empleado'] = df['ind_empleado'].map(lambda x: emp_dict[x])
    df['indfall'] = df['indfall'].map(lambda x: indfall_dict[x])
    df['sexo'] = df['sexo'].map(lambda x: sexo_dict[x])
    df['tiprel_1mes'] = df['tiprel_1mes'].map(lambda x: tiprel_dict[x])
    df['indresi'] = df['indresi'].map(lambda x: indresi_dict[x])
    df['indext'] = df['indext'].map(lambda x: indext_dict[x])
    df['conyuemp'] = df['conyuemp'].map(lambda x: conyuemp_dict[x])
    df['segmento'] = df['segmento'].map(lambda x: segmento_dict[x])
    df['nomprov'] = df['nomprov'].map(lambda x:nomprov_dict[x])
    df['renta'] = df['renta'].fillna(0).astype(np.float64)
    
    df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 1

    df['fecha_dato_month'] = df['fecha_dato'].map(lambda x: int(x[5:7])).astype(np.int8)
    df['fecha_dato_year'] = df['fecha_dato'].map(lambda x: int(x[0:4]) - 2015).astype(np.int8)
    df['month_int'] = (df['fecha_dato_month'] + 12 * df['fecha_dato_year']).astype(np.int8)
    df.drop('fecha_dato',axis=1,inplace=True)

    
    df['fecha_alta'] = df['fecha_alta'].map(lambda x: '2020-00-00' if x == -1 else x)
    df['fecha_alta_month'] = df['fecha_alta'].map(lambda x: int(x[5:7])).astype(np.int8)
    df['fecha_alta_year'] = df['fecha_alta'].map(lambda x: int(x[0:4]) - 2015).astype(np.int8)
    df['fecha_alta_day'] = df['fecha_alta'].map(lambda x: int(x[8:10])).astype(np.int8)
    df['fecha_alta_month_int'] = (df['fecha_alta_month'] + 12 * df['fecha_dato_year']).astype(np.int8)
    df.drop('fecha_alta',axis=1,inplace=True)
    df.drop(['fecha_alta_month','fecha_alta_day','fecha_alta_month_int'], axis = 1, inplace = True)

    
    df['ult_fec_cli_1t'] = df['ult_fec_cli_1t'].map(lambda x: '2020-00-00' if x == -1 else x)
    df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(lambda x: int(x[5:7])).astype(np.int8)
    df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(lambda x: int(x[0:4]) - 2015).astype(np.int8)
    df['ult_fec_cli_1t_day'] = df['ult_fec_cli_1t'].map(lambda x: int(x[8:10])).astype(np.int8)
    df['ult_fec_cli_1t_month_int'] = (df['ult_fec_cli_1t_month'] + 12 * df['ult_fec_cli_1t_year']).astype(np.int8)
    df.drop('ult_fec_cli_1t',axis=1,inplace=True)
    
    #####DROP WORTHLESS FEATURES
#     df.drop(['nomprov'],axis = 1, inplace = True) #codprov and numprov has one to one mapping and CODPROV Is very significant feature
    df.drop(['conyuemp'],axis = 1, inplace = True) ###~99% are missing values
    
    df['age'] = df['age'].astype(int)
    df['antiguedad'] = df['antiguedad'].astype(int)

    all_cols = df.columns.values
    num_cols = df._get_numeric_data()
    cat_cols = list(set(all_cols) - set(num_cols))

    df = pd.get_dummies(df, columns = cat_cols)
    
    cols = df.columns.tolist()
    cols.sort()
    
    df = df[cols]
    
    return df

def getAge(df):
    mean_age = 40.
    min_age = 20.
    max_age = 90.
    range_age = max_age - min_age
    df['age'] = df['age'].fillna(mean_age)
    df.loc[df['age'] < min_age, 'age'] = min_age
    df.loc[df['age'] > max_age, 'age'] = max_age
    df['new_age'] = (df['age']-min_age) /range_age
    
    df.drop(['age'], axis = 1 , inplace = True)
    return df
   

def getCustSeniority(df):
    min_value = 0.
    max_value = 256.
    range_value = max_value - min_value
    missing_value = 0.
    df['antiguedad'] =df['antiguedad'].fillna(missing_value)
    df.loc[df['antiguedad'] < min_value, 'antiguedad'] = min_value
    df.loc[df['antiguedad'] > max_value, 'antiguedad'] = max_value
    
    df['new_antiguedad'] = (df['antiguedad']-min_value) /range_value
    df.drop(['antiguedad'], axis = 1 , inplace = True)
    return df
    

def getRent(df):

    min_value = 0.
    max_value = 1500000.
    range_value = max_value - min_value
    renta_dict = {-1:101850,1:101850,2:76895,3:60562,4:77815,5:78525,6:60155,7:114223,8:135149,9:87410,10:78691,11:75397,12:70359,13:61962
                  ,14:63260,15:103567,16:70751,17:100208,18:80489,19:100635,20:101850,21:75534,22:80324,23:67016,24:76339
                  ,25:59191,26:91545,27:68219,28:141381,29:89534,30:68713,31:101850,32:78776,33:83995,34:90843,35:78168
                  ,36:94328,37:88738,38:83383,39:87142,40:81287,41:94814,42:71615,43:81330,44:64053,45:65242,46:73463
                  ,47:92032,48:101850,49:73727,50:98827,51:333283,52:116469}
    df['mean_renta'] =df['cod_prov'].map(lambda x: renta_dict[x])

#     missing_value = 101850.

    df.loc[df['renta'] < min_value, 'renta'] = df['mean_renta']
    df.loc[df['renta'] > max_value, 'renta'] = max_value
    
       
    
    df['new_renta'] = (df['renta']-min_value) /range_value
    df.drop(['renta','mean_renta'], axis = 1 , inplace = True)
    
    df.drop('cod_prov',axis = 1, inplace = True)
    return df
    

train_data_final = processdata(train_data_final)   
train_data_final = getAge(train_data_final)
train_data_final = getCustSeniority(train_data_final)
train_data_final = getRent(train_data_final)

test_data_final = processdata(test_data_final)
test_data_final = getAge(test_data_final)
test_data_final = getCustSeniority(test_data_final)
test_data_final = getRent(test_data_final)


# In[609]:




def cust_eng_across_months(df,istrain,rollback,backward):
    
    #####get how many times a customers was engaged in each of the channels in past 5 months
    ##### for e.g, how many times in past 5 months for Credit Card,  how many times in past 5 months for Current Account etc..
    if istrain == 1:
        month_list = ['2015-01-28', '2015-02-28','2015-03-28', '2015-04-28','2015-05-28']
    else:
        month_list = ['2016-01-28', '2016-02-28','2016-03-28', '2016-04-28','2016-05-28']
    
    if backward == 1:
        month_list_new = month_list[::-1][:rollback]
    else:
        month_list_new = month_list[:rollback]
        
    print (month_list_new)
    
    t123 = df[df.fecha_dato.isin(month_list_new)]

    for cols in list_of_preds:

        t123[cols] = t123[cols].fillna(0)
        num_occur = t123.groupby(['ncodpers'])[cols].agg([(cols+str(backward)+str('_tval_sum_')+str(rollback),np.sum)])                                                    .astype(int).reset_index()

        if cols == list_of_preds[0]:
            across_months = num_occur
        else:
            across_months = pd.merge(across_months, num_occur, on = ['ncodpers'])

    del t123, num_occur
    
   
    return across_months

def cust_eng_across_channels(df,istrain):
    
    #####get how many times a customers was engaged in ACROSS channels in past 5 months
    #### for e.g. how many products in Jan, how many products in Feb etc..
    if istrain == 1:
        month_dict = {'2015-01-28': 'm1', '2015-02-28': 'm2','2015-03-28': 'm3', '2015-04-28': 'm4','2015-05-28': 'm5'}
    else:
        month_dict = {'2016-01-28': 'm1', '2016-02-28': 'm2','2016-03-28': 'm3', '2016-04-28': 'm4','2016-05-28': 'm5'}

    month_list = month_dict.keys()
    month_list.sort()
    
    t123 = df[df.fecha_dato.isin(month_list)]
    t123[cols] = t123[cols].fillna(0)
    t123['month_id'] = t123['fecha_dato'].map(lambda x: month_dict[x])
    t123['presence'] = t123[list_of_preds].apply(lambda x: sum(x), axis = 1).fillna(0).astype(int)
    
    across_channels = t123.pivot(index = 'ncodpers', columns = 'month_id', values = 'presence').reset_index()
    
    across_channels['diff_m5_m4'] = across_channels['m5'] - across_channels['m4']
    across_channels['diff_m5_m3'] = across_channels['m5'] - across_channels['m3']
    across_channels['diff_m5_m2'] = across_channels['m5'] - across_channels['m2']
    across_channels['diff_m5_m1'] = across_channels['m5'] - across_channels['m1']
    
    across_channels['diff_m4_m3'] = across_channels['m4'] - across_channels['m3']
    across_channels['diff_m4_m2'] = across_channels['m4'] - across_channels['m2']
    across_channels['diff_m4_m1'] = across_channels['m4'] - across_channels['m1']
    
    across_channels['diff_m3_m2'] = across_channels['m3'] - across_channels['m2']
    across_channels['diff_m3_m1'] = across_channels['m3'] - across_channels['m1']
    
    across_channels['diff_m2_m1'] = across_channels['m2'] - across_channels['m1']
       
    return across_channels

def lag_features(istrain):
    
    if istrain == 1:
        month_list = ['2015-01-28', '2015-02-28','2015-03-28', '2015-04-28','2015-05-28','2015-06-28']
        data = train_data[train_data.fecha_dato.isin(month_list)]
    else:
        month_list = ['2016-01-28', '2016-02-28','2016-03-28', '2016-04-28','2016-05-28','2016-06-28']
        data1 = train_data[train_data.fecha_dato.isin(month_list)]
        data2 = test_data[test_data.fecha_dato.isin(month_list)]
        data = data1.append(data2)
        del data1, data2
    
    vars_req = ['ncodpers','fecha_dato']
    vars_req.extend(list_of_preds)
    
    data = data[vars_req]
    
    for i in range(0,len(list_of_preds)):
        preds = list_of_preds[i]
        
        data[preds+str('_lag1')] = data.groupby(['ncodpers'])[preds].shift(1).fillna(9999).astype(int)
        data[preds+str('_lag2')] = data.groupby(['ncodpers'])[preds].shift(2).fillna(9999).astype(int)
        data[preds+str('_lag3')] = data.groupby(['ncodpers'])[preds].shift(3).fillna(9999).astype(int)
        data[preds+str('_lag4')] = data.groupby(['ncodpers'])[preds].shift(4).fillna(9999).astype(int)
        data[preds+str('_lag5')] = data.groupby(['ncodpers'])[preds].shift(5).fillna(9999).astype(int)
        
    month_list = month_list[::-1][:1] #pick the last element of list    
    
    data = data[data.fecha_dato.isin(month_list)]
    
    for i in range(0,len(list_of_preds)):
        preds = list_of_preds[i]
        data[preds+str('_lag_54')] = data[preds+str('_lag5')].astype(str) + '_' + data[preds+str('_lag4')].astype(str)
        data[preds+str('_lag_43')] = data[preds+str('_lag4')].astype(str) + '_' + data[preds+str('_lag3')].astype(str)
        data[preds+str('_lag_32')] = data[preds+str('_lag3')].astype(str) + '_' + data[preds+str('_lag2')].astype(str)
        data[preds+str('_lag_21')] = data[preds+str('_lag2')].astype(str) + '_' + data[preds+str('_lag1')].astype(str)

        data.loc[data[preds+str('_lag_54')].isin(['1_0','0_1','1_1']),preds+str('_54')] = 1
        data.loc[~data[preds+str('_lag_54')].isin(['1_0','0_1','1_1']),preds+str('_54')] = 0

        data.loc[data[preds+str('_lag_43')].isin(['0_1','1_1']),preds+str('_43')] = 1
        data.loc[~data[preds+str('_lag_43')].isin(['0_1','1_1']),preds+str('_43')] = 0

        data.loc[data[preds+str('_lag_32')].isin(['0_1','1_1']),preds+str('_32')] = 1
        data.loc[~data[preds+str('_lag_32')].isin(['0_1','1_1']),preds+str('_32')] = 0

        data.loc[data[preds+str('_lag_21')].isin(['0_1','1_1']),preds+str('_21')] = 1
        data.loc[~data[preds+str('_lag_21')].isin(['0_1','1_1']),preds+str('_21')] = 0

        data[preds+str('_total_adds')] = data[preds+str('_54')]+data[preds+str('_43')]+data[preds+str('_32')]+data[preds+str('_21')].astype(int)

        data.drop([preds+str('_lag_54'),preds+str('_lag_43'),preds+str('_lag_32'),preds+str('_lag_21')], axis = 1, inplace = True)
        
    
    data.drop(list_of_preds, axis = 1, inplace = True)
    data.drop('fecha_dato',axis = 1, inplace = True)
    
    return data
        
        
##### FEATURE : ACROSS MONTHS CUSTOMER ENGAGEMENT
for j in range(3,5): ###we take previous months engagement of customers for each product
    
    j += 1
       
    start_time = datetime.datetime.now()
    
    ###get values to be used to training data
    across_months = cust_eng_across_months(train_data, istrain = 1, rollback = j, backward = 1)
    train_data_final = pd.merge(train_data_final, across_months, on = ['ncodpers'], how = 'left')
    
    #get values to be used for test data
    across_months = cust_eng_across_months(train_data, istrain = 0, rollback = j, backward = 1)
    test_data_final = pd.merge(test_data_final, across_months, on = ['ncodpers'], how = 'left')
    
    print ('time taken for j %s is %s seconds' %(j, (datetime.datetime.now()-start_time)))
    
# FEATURE : ACROSS CHANNELS CUSTOMER ENGAGEMENT

start_time = datetime.datetime.now()
across_channels = cust_eng_across_channels(train_data,istrain = 1)
train_data_final = pd.merge(train_data_final, across_channels, on = ['ncodpers'], how = 'left')

across_channels = cust_eng_across_channels(train_data,istrain = 0)
test_data_final = pd.merge(test_data_final, across_channels, on = ['ncodpers'], how = 'left')
print ('time taken for across channel feature is %s seconds' %(datetime.datetime.now()-start_time))
   

# FEATURE : LAG VALUES FOR ALL CHANNELS
lag_data = lag_features(istrain = 1)
train_data_final = pd.merge(train_data_final, lag_data, on = ['ncodpers'], how = 'left')
del lag_data

lag_data = lag_features(istrain = 0)
test_data_final = pd.merge(test_data_final, lag_data, on = ['ncodpers'], how = 'left')
del lag_data
    
    ##### THIS SECTION PROVED TO BE INEFFECTIVE AND DIDN"T ADD TO MODEL PERFORMANCE (INFACT IT MADE THE MODEL SLIGHTLY WORSE)
#     ##create combination of new created variables
#     for i in range(0,len(additional_var_list)):
#         col1 = additional_var_list[i]
#         col1 = col1+str('_')+str(j)

#         for k in range(i+1,len(additional_var_list)):
#             col2 = additional_var_list[k]
#             col2 = col2+str('_')+str(j)
#             train_data_final['sum_combo_'+str(j)+str(i)+str(k)] = train_data_final[col1] + train_data_final[col2]
#             test_data_final['sum_combo_'+str(j)+str(i)+str(k)] = test_data_final[col1] + test_data_final[col2]





# In[515]:

####create the map@7 evaluation function

def apk(actual, predicted, k):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.sum([apk(a,p,k) for a,p in zip(actual, predicted)])/929615 ### hard coded the denominator with the population size!!



# In[647]:

###################### EXTREME GRADIENT BOOSTING MODEL ###############################

        ####################### CROSS VALIDATION FOR PARAMETER TUNING ###############
# id_x = train_data_final['ncodpers']

# y = train_data_final['product_seq']

# X = train_data_final.drop(['product_seq','ncodpers'], axis = 1)

# xgtrain   = xgb.DMatrix(X, label=y)

# # #testing xgb parameters using CV
# max_depth_p = [8]
# min_child_weight_p = [1]
# eta_p = [0.2]

# for i in range(0,len(max_depth_p)):
#     for j in range(0,len(min_child_weight_p)):
#         for k in range(0,len(eta_p)):

#             mdp = max_depth_p[i]
#             mcwp = min_child_weight_p[j]
#             etap = eta_p[k]

#             print ('-------- max depth %s , min_child_weight %s, eta %s' %(mdp,mcwp,etap))
#             param_cv = {'max_depth':mdp, 'min_child_weight':mcwp, 'eta':etap, 'gamma':0,'objective':'multi:softprob'
#                         ,'subsample':0.8, 'colsample_by_tree':0.8, 'silent': 0, 'scale_pos_weight':1, 'num_class': 22}
#             model_cv = xgb.cv(param_cv, xgtrain, num_boost_round= 100, nfold=5,metrics={'mlogloss'},early_stopping_rounds=30
#                               ,verbose_eval = 10,seed = 1729)
#             print ('------------')



        ############################################## MODEL TRAINING ########################################

id_x = train_data_final['ncodpers']

y = train_data_final['product_seq']

X = train_data_final.drop(['product_seq','ncodpers','fecha_alta_year'], axis = 1)

xgtrain = xgb.DMatrix(X, label = y)

ROUNDS = 50 ########### THIS PROBABLY MY BIG MISTAKE. SHOULD HAVE KEPT IT AT 100, AS SUGGESTED BY CV

param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.1
param['max_depth'] = 8
param['gamma'] = 0
param['lambda'] = 4
param['silent'] = 1
param['num_class'] = 22
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['seed'] = 1729


plst      = list(param.items())
watchlist = [(xgtrain, 'train')]

model_xgb = xgb.train(plst, xgtrain, ROUNDS, watchlist,early_stopping_rounds = 5, verbose_eval = 10)

######## predict training output
preds_train = model_xgb.predict(xgtrain)
preds_train = np.argsort(preds_train, axis = 1)
preds_train = np.fliplr(preds_train)[:,:7]

preds_train_check = pd.DataFrame(preds_train, columns = list('abcdefg'))
preds_train_check = pd.concat([preds_train_check, id_x], axis = 1)
preds_train_check = preds_train_check.drop_duplicates()

preds_train_check = preds_train_check.drop(['ncodpers'], axis = 1)
preds_train_check = preds_train_check.as_matrix()
preds_train_check = preds_train_check.tolist()


########### capture the actual outcome
actual_outcome = pd.concat([id_x,y], axis = 1)
actual_outcome = actual_outcome.pivot(index = 'ncodpers', columns = 'product_seq', values = 'product_seq')                        .reset_index().fillna(100).astype(int)
actual_outcome.drop(['ncodpers'], axis = 1, inplace = True)
actual_outcome = actual_outcome.as_matrix()
actual_outcome = actual_outcome.tolist()

########## evaluate the model since, objective function of the model is diffrent from the evaluation function
for i in range(0,len(actual_outcome)):
    
    new_list = actual_outcome[i]
    new_list = [x for x in new_list if x != 100]
    
    if i == 0:
        final_list = [0]
    
    final_list.append(new_list)
    
final_list.remove(0)

mapk(final_list, preds_train_check, k = 7)


         ############################################### FEATURE IMPORTANCE ########################################

feat_imp = pd.DataFrame(model_xgb.get_fscore().items(),columns=['feature', 'value'])
feat_imp = feat_imp.sort_values(by = ['value'], ascending = False)
feat_imp

        
         ############################################### MODEL PREDICTION ########################################

test_cust_id = test_data_final['ncodpers']
X_test = test_data_final.drop(['ncodpers','fecha_alta_year'], axis = 1)
xg_test = xgb.DMatrix(X_test)
del X_test
preds_test = model_xgb.predict(xg_test)
preds_test = np.argsort(preds_test,axis = 1)
preds_test = np.fliplr(preds_test)[:,:7]

list_of_preds = np.array(list_of_preds)
preds_final = [" ".join(list(list_of_preds[preds])) for preds in preds_test]

preds_output = pd.DataFrame({'ncodpers':test_cust_id,'added_products':preds_final})
preds_output.to_csv('santander_output_50.csv', index = False)


