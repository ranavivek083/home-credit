import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_colwidth', -1)
import warnings
warnings.filterwarnings("ignore")
import gc

credit_bal = pd.read_csv('credit_card_balance.csv')
def late_dpd(dpd):
    list_dpd = list(dpd)
    count_dpd = 0
    for i,j in enumerate(list_dpd):
        if j !=0:
            count_dpd+=1

    return count_dpd


dpd_group = credit_bal.groupby(by=['SK_ID_CURR', 'SK_ID_PREV']).apply(lambda x: late_dpd(x.SK_DPD)).reset_index().\
    rename(index = str, columns = {0:'NO_DPD'})

grp1 = dpd_group.groupby(by = ['SK_ID_CURR'])['NO_DPD'].mean().reset_index().\
        rename(index = str, columns = {'NO_DPD' : 'DPD_COUNT'})

credit_bal = credit_bal.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
del dpd_group
gc.collect()
print(credit_bal.dtypes, credit_bal.shape)

def min_pay(min_pay, actual_pay):
    M = list(min_pay)
    A = list(actual_pay)
    P = len(M)
    count_miss = 0
    for i in range(len(M)):
        if A[i]<M[i]:
            count_miss+=1
    return (100*count_miss)/P


miss_group = credit_bal.groupby(by = ['SK_ID_CURR']).\
    apply(lambda x: min_pay(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT))\
    .reset_index().rename(index = str, columns = { 0 : 'NET_MISSED_PAYMENT'})
credit_bal = credit_bal.merge(miss_group, on=['SK_ID_CURR'], how='left')
del miss_group
gc.collect()
print(credit_bal.dtypes, credit_bal.shape)



avg_group = credit_bal.groupby(by=['SK_ID_CURR'])['SK_DPD'].mean().reset_index()\
    .rename(index=str, columns={'SK_DPD': 'AVG_DPD'})
credit_bal = credit_bal.merge(avg_group, on=['SK_ID_CURR'], how='left')
del avg_group
gc.collect()
print(credit_bal.dtypes, credit_bal.shape)


loan_grp = credit_bal.groupby(by = ['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index().rename(index = str, columns = {'SK_ID_PREV': 'NO_LOANS'})
credit_bal = credit_bal.merge(loan_grp, on = ['SK_ID_CURR'], how = 'left')
del loan_grp
gc.collect()
print(credit_bal.dtypes, credit_bal.shape)

grp = credit_bal.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index().rename(index = str, columns = {'CNT_INSTALMENT_MATURE_CUM': 'NO_INSTALMENTS'})
grp1 = grp.groupby(by = ['SK_ID_CURR'])['NO_INSTALMENTS'].sum().reset_index().rename(index = str, columns = {'NO_INSTALMENTS': 'TOTAL_INSTALMENTS'})
credit_bal = credit_bal.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
del grp, grp1
gc.collect()

# Average Number of installments paid per loan

credit_bal['INSTALLMENTS_PER_LOAN'] = (credit_bal['TOTAL_INSTALMENTS']/credit_bal['NO_LOANS']).astype('uint32')
del credit_bal['TOTAL_INSTALMENTS']
del credit_bal['NO_LOANS']
gc.collect()

print(credit_bal.dtypes, credit_bal.shape)

del credit_bal['AMT_DRAWINGS_ATM_CURRENT']
del credit_bal['AMT_DRAWINGS_CURRENT']
del credit_bal['AMT_DRAWINGS_OTHER_CURRENT']
del credit_bal['AMT_DRAWINGS_POS_CURRENT']
del credit_bal['CNT_DRAWINGS_OTHER_CURRENT']
del credit_bal['CNT_DRAWINGS_POS_CURRENT']
del credit_bal['CNT_DRAWINGS_CURRENT']
del credit_bal['CNT_DRAWINGS_ATM_CURRENT']
del credit_bal['SK_DPD_DEF']
del credit_bal['SK_ID_PREV']

credit_bal = credit_bal.sort_values("SK_ID_CURR", ascending=True)

sk_curr = list(set((credit_bal['SK_ID_CURR'])))
sk_curr.sort()

avg_month_bal = credit_bal.groupby('SK_ID_CURR', as_index= False)['MONTHS_BALANCE'].mean()
avg_month_bal = list(avg_month_bal['MONTHS_BALANCE'])
amt_bal = credit_bal.groupby('SK_ID_CURR', as_index= False)['AMT_BALANCE'].mean()
amt_bal = list(amt_bal['AMT_BALANCE'])
amt_bal = credit_bal.groupby('SK_ID_CURR', as_index= False)['AMT_BALANCE'].mean()
amt_bal = list(amt_bal['AMT_BALANCE'])
credit_limit = credit_bal.groupby('SK_ID_CURR', as_index= False)['AMT_CREDIT_LIMIT_ACTUAL'].mean()
credit_limit = list(credit_limit['AMT_CREDIT_LIMIT_ACTUAL'])
min_instal = credit_bal.groupby('SK_ID_CURR', as_index= False)['AMT_INST_MIN_REGULARITY'].mean()
min_instal = list(min_instal['AMT_INST_MIN_REGULARITY'])

pay_curr = credit_bal.groupby('SK_ID_CURR', as_index= False)['AMT_PAYMENT_CURRENT'].mean()
pay_curr = list(pay_curr['AMT_PAYMENT_CURRENT'])

pay_curr_tot = credit_bal.groupby('SK_ID_CURR', as_index= False)['AMT_PAYMENT_TOTAL_CURRENT'].mean()
pay_curr_tot = list(pay_curr_tot['AMT_PAYMENT_TOTAL_CURRENT'])

princ_amt_rec = credit_bal.groupby('SK_ID_CURR', as_index= False)['AMT_RECEIVABLE_PRINCIPAL'].mean()
princ_amt_rec = list(princ_amt_rec['AMT_RECEIVABLE_PRINCIPAL'])

tot_rec = credit_bal.groupby('SK_ID_CURR', as_index= False)['AMT_TOTAL_RECEIVABLE'].mean()
tot_rec = list(tot_rec['AMT_TOTAL_RECEIVABLE'])

inst = credit_bal.groupby('SK_ID_CURR', as_index= False)['CNT_INSTALMENT_MATURE_CUM'].mean()
inst = list(inst['CNT_INSTALMENT_MATURE_CUM'])

miss_pay = credit_bal.groupby('SK_ID_CURR', as_index= False)['NET_MISSED_PAYMENT'].mean()
miss_pay = list(miss_pay['NET_MISSED_PAYMENT'])

dpd = credit_bal.groupby('SK_ID_CURR', as_index= False)['DPD_COUNT'].mean()
dpd = list(dpd['DPD_COUNT'])

av_dpd = credit_bal.groupby('SK_ID_CURR', as_index= False)['AVG_DPD'].mean()
av_dpd = list(av_dpd['AVG_DPD'])

inst_per_loan = credit_bal.groupby('SK_ID_CURR', as_index= False)['INSTALLMENTS_PER_LOAN'].mean()
inst_per_loan = list(inst_per_loan['INSTALLMENTS_PER_LOAN'])

data = []

for i in range(len(sk_curr)):
    data.append([sk_curr[i], avg_month_bal[i], amt_bal[i], credit_limit[i], min_instal[i], pay_curr[i], pay_curr_tot[i],
                 princ_amt_rec[i], tot_rec[i], inst[i], miss_pay[i], dpd[i], av_dpd[i], inst_per_loan[i]])

final_credit_data = pd.DataFrame(data, columns=['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL'
                            ,'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT'
                            ,'AMT_RECEIVABLE_PRINCIPAL', 'AMT_TOTAL_RECEIVABLE', 'CNT_INSTALMENT_MATURE_CUM'
                            ,'NET_MISSED_PAYMENT', 'DPD_COUNT', 'AVG_DPD', 'INSTALLMENTS_PER_LOAN'])

final_credit_data.isnull().sum()
final_credit_data['AMT_PAYMENT_CURRENT'].mean()
final_credit_data['AMT_PAYMENT_CURRENT'] = final_credit_data['AMT_PAYMENT_CURRENT'].fillna(17934.67856692355)

final_credit_data.to_csv('/Users/phanitejab/Documents/DS/Home Credit/home_credit_git/credit_card_balance.csv',index = False)