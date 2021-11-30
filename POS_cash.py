# importing packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import gc
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# loading csv data into dataframes
application = pd.read_csv('application_train.csv')
previous_application = pd.read_csv('previous_application.csv')
pos = pd.read_csv('POS_CASH_balance.csv')



def cat_summary(data, categorical_cols, target, number_of_classes=10):
    var_count = 0  # How many categorical variables will be reported?
    vars_more_classes = []  # Variables with more than a certain number of classes will be stored.
    for var in categorical_cols:
        if len(previous_application[var].value_counts()) <= number_of_classes:  # Choose by number of classes.
            print(pd.DataFrame({var: data[var].value_counts(),
                                "Ratio": 100 * data[var].value_counts() / len(data),
                                "TARGET_MEDIAN": data.groupby(var)[target].median()}), end="\n\n\n")
            var_count += 1
        else:
            vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


# understanding categorical data
cat_cols = [col for col in previous_application.columns if previous_application[col].dtypes == 'O']



previous_application.describe().T

# # Correlation Matrix
#sns.heatmap(corrmat,vmax = 1,square = True,annot = True,vmin = -1)

# checking null values
def missing_values_table(df_prev):
    mis_val = df_prev.isnull().sum()
    mis_val_percent = 100 * df_prev.isnull().sum() / len(df_prev)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    print("Your selected dataframe has " + str(df_prev.shape[1]) + " columns.\n"
                                                                   "There are " + str(
        mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    return mis_val_table_ren_columns
cat_summary(previous_application, cat_cols, "SK_ID_CURR")

# understanding numerical data
#corrmat = previous_application.corr()
#fig = plt.figure(figsize = (10,7))
#plt.show()

missing_values_table(previous_application)


missing_values_table(pos)
previous_application.info()

# Info
pos.info()

# some improvements
a = ['Family', 'Spouse, partner', 'Children', 'Other_B', 'Other_A', 'Group of people']
previous_application["NAME_TYPE_SUITE"] = previous_application["NAME_TYPE_SUITE"].replace(a, 'Accompanied')

b = ['Auto Accessories', 'Jewelry', 'Homewares', 'Medical Supplies', 'Vehicles', 'Sport and Leisure',
         'Gardening', 'Other', 'Office Appliances', 'Tourism', 'Medicine', 'Direct Sales', 'Fitness', 'Additional Service',
         'Education', 'Weapon', 'Insurance', 'House Construction', 'Animals']
previous_application["NAME_GOODS_CATEGORY"] = previous_application["NAME_GOODS_CATEGORY"].replace(b, 'others')

c = ['AP+ (Cash loan)', 'Channel of corporate sales', 'Car dealer']
previous_application["CHANNEL_TYPE"] = previous_application["CHANNEL_TYPE"].replace(c, 'Other_Channel')

d = ['Auto technology', 'Jewelry', 'MLM partners', 'Tourism']
previous_application["NAME_SELLER_INDUSTRY"] = previous_application["NAME_SELLER_INDUSTRY"].replace(d, 'Others')

e = ['Refusal to name the goal', 'Money for a third person', 'Buying a garage','Gasification / water supply',
     'Hobby','Business development','Buying a holiday home / land','Furniture','Car repairs',
     'Buying a home','Wedding / gift / holiday']
previous_application["NAME_CASH_LOAN_PURPOSE"] = previous_application["NAME_CASH_LOAN_PURPOSE"].replace(e, 'Other_Loan')

previous_application.groupby(["WEEKDAY_APPR_PROCESS_START","NAME_CONTRACT_STATUS"]).agg("count")

weekend = ["SATURDAY","SUNDAY"]

previous_application["WEEKDAY_APPR_PROCESS_START"] = previous_application["WEEKDAY_APPR_PROCESS_START"].apply(lambda x : "WEEKEND" if x in weekend else "WEEKDAY")

previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)


#* AMT_APPLICATION: For how much credit did client ask on the previous application.
#* AMT_CREDIT : Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our approval process he could have received different amount.
#* 1) NEW_AMT_CTEDIT_RATE: Received Amount Rate

previous_application['NEW_APP_CREDIT_RATE'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']
#* AMT_CREDIT
#* AMT_ANNUITY: Annuity of previous application.
#* 2) NEW_AMT_PAYMENT_RATE: The ratio of the loan received to the monthly payment.
previous_application['NEW_AMT_PAYMENT_RATE'] = previous_application['AMT_CREDIT'] / previous_application['AMT_ANNUITY']
# 3) NEW_APP_GOODS_RATE
previous_application['NEW_APP_GOODS_RATE'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_GOODS_PRICE']
# 4) NEW_CREDIT_GOODS_RATE
previous_application['NEW_CREDIT_GOODS_RATE'] = previous_application['AMT_CREDIT'] / previous_application['AMT_GOODS_PRICE']
# 5) NEW_RETURN_DAY
previous_application['NEW_RETURN_DAY'] =  previous_application['DAYS_DECISION'] + previous_application['CNT_PAYMENT'] * 30
# 6) NEW_DAYS_TERMINATION_DIFF
# Expected end day - days remaining or how many days are due.
previous_application['NEW_DAYS_TERMINATION_DIFF'] = previous_application['DAYS_TERMINATION'] - previous_application['NEW_RETURN_DAY']
# 7) NEW_DAYS_DUE_DIFF
# (According to the application date of the current application, when was the first due date of the previous application?) -
#(According to the application date of the current application, when should the first term of the previous application be?)
previous_application['NEW_DAYS_DUE_DIFF'] =  previous_application['DAYS_LAST_DUE_1ST_VERSION'] - previous_application['DAYS_FIRST_DUE']
# 8) NEW_APP_CREDIT_RATE_RATIO
#If x <= 1, a client got the desired loan or more.
previous_application["NEW_APP_CREDIT_RATE_RATIO"] = previous_application["NEW_APP_CREDIT_RATE"].apply(lambda x: 1 if(x<=1) else 0)
# 9) NEW_RETURN_DAY
# the day the loan was issued + the number of loan installments * 30 (the number of days of the loan)
previous_application['NEW_RETURN_DAY'] =  previous_application['DAYS_DECISION'] + previous_application['CNT_PAYMENT'] * 30
# 10) NEW_CNT_PAYMENT
# classifying the installment numbers of loans as  0-12 short | 12-60 medium | 60-120 long
previous_application["NEW_CNT_PAYMENT"]=pd.cut(x=previous_application['CNT_PAYMENT'], bins=[0, 12, 60,120], labels=["Kısa", "Orta", "Uzun"])
# 11) NEW_END_DIFF
# According to the application date of the current application, when was the expected end of the previous application?
# According to the application date of the current application, when was the last due date for the previous application?
previous_application["NEW_END_DIFF"] = previous_application["DAYS_TERMINATION"] - previous_application["DAYS_LAST_DUE"]



# one hot encoding
def one_hot_encoder(df_prev, nan_as_category = False):
    original_columns = list(df_prev.columns)
    categorical_columns = [col for col in df_prev.columns if df_prev[col].dtype == 'object']
    df_prev = pd.get_dummies(df_prev, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df_prev.columns if c not in original_columns]
    return df_prev, new_columns


previous_application, cat_cols = one_hot_encoder(previous_application, nan_as_category= True)

num_aggregations = {
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    'AMT_APPLICATION': ['min', 'max', 'mean'],
    'AMT_CREDIT': ['min', 'max', 'mean'],
    'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
    'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
    'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
    'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['mean', 'sum'],
    'NEW_APP_CREDIT_RATE': ['min', 'max', 'mean', 'var'],
    'NEW_AMT_PAYMENT_RATE': ['min', 'max', 'mean'],
    'NEW_CREDIT_GOODS_RATE': ['min', 'max', 'mean'],
    'NEW_RETURN_DAY': ['min', 'max', 'mean', 'var'],
    'NEW_DAYS_TERMINATION_DIFF': ['min', 'max', 'mean'],
    'NEW_END_DIFF': ['min', 'max', 'mean'],
    'NEW_RETURN_DAY': ['min', 'max', 'mean'],
    'NEW_APP_CREDIT_RATE_RATIO': ['min', 'max', 'mean'],
    'NEW_DAYS_DUE_DIFF': ['min', 'max', 'mean']
}

cat_aggregations = {}

for cat in cat_cols:
    cat_aggregations[cat] = ['mean']
prev_agg = previous_application.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

approved = previous_application[previous_application['NAME_CONTRACT_STATUS_Approved'] == 1]
approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
approved_agg.columns = pd.Index(['APR_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
refused = previous_application[previous_application['NAME_CONTRACT_STATUS_Refused'] == 1]
refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
refused_agg.columns = pd.Index(['REF_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
del refused, refused_agg, approved, approved_agg, previous_application
gc.collect()
features.to_csv('prev_app_features_v1.csv')

train_data = pd.read_csv('prev_app_features.csv')
train_data = train_data[['PREV_NEW_DAYS_TERMINATION_DIFF_MAX', 'APR_AMT_APPLICATION_MIN',
       'PREV_AMT_APPLICATION_MAX', 'APR_NEW_DAYS_TERMINATION_DIFF_MAX',
       'PREV_AMT_GOODS_PRICE_MAX', 'APR_AMT_GOODS_PRICE_MAX',
       'APR_AMT_APPLICATION_MAX', 'PREV_NEW_DAYS_TERMINATION_DIFF_MEAN',
       'APR_NEW_DAYS_TERMINATION_DIFF_MEAN',
       'APR_NEW_DAYS_TERMINATION_DIFF_MIN',
       'PREV_NEW_DAYS_TERMINATION_DIFF_MIN', 'APR_AMT_GOODS_PRICE_MIN',
       'PREV_AMT_CREDIT_MAX', 'APR_AMT_GOODS_PRICE_MEAN', 'APR_AMT_CREDIT_MAX',
       'PREV_AMT_GOODS_PRICE_MIN', 'APR_AMT_APPLICATION_MEAN',
       'APR_AMT_CREDIT_MEAN', 'PREV_AMT_GOODS_PRICE_MEAN',
       'APR_AMT_CREDIT_MIN', 'APR_AMT_ANNUITY_MIN', 'PREV_AMT_ANNUITY_MIN',
       'APR_AMT_ANNUITY_MAX', 'PREV_AMT_ANNUITY_MAX', 'APR_AMT_ANNUITY_MEAN',
       'PREV_AMT_ANNUITY_MEAN', 'APR_DAYS_DECISION_MIN',
       'PREV_AMT_CREDIT_MEAN', 'APR_NEW_RETURN_DAY_MEAN',
       'PREV_DAYS_DECISION_MIN', 'PREV_AMT_APPLICATION_MEAN',
       'PREV_NEW_RETURN_DAY_MIN', 'APR_NEW_RETURN_DAY_MIN',
       'APR_DAYS_DECISION_MEAN', 'APR_NEW_RETURN_DAY_MAX',
       'PREV_DAYS_DECISION_MEAN', 'PREV_NEW_RETURN_DAY_MEAN',
       'APR_DAYS_DECISION_MAX', 'PREV_DAYS_DECISION_MAX',
       'PREV_NEW_RETURN_DAY_MAX']]
#application_train = application[['SK_ID_CURR','TARGET']]
#train_data = features
#train_data = pd.merge(features,application_train,how='inner',on='SK_ID_CURR')
#train_data = train_data.drop('SK_ID_CURR', axis=1)
train_data = train_data.fillna(0)
from numpy import inf
train_data[train_data == inf] = 0
train_data[train_data == -inf] = 0

for c in train_data.columns:
    if train_data[c].dtype == np.float:
        train_data[c] = train_data[c].astype(int)

train_data.to_csv('prev_app_features_v2.csv')

#random forest
X = train_data.drop('TARGET', axis=1)
y = train_data['TARGET']

from sklearn.model_selection import train_test_split
# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

plt.barh(X.columns, rfc.feature_importances_)

sorted_idx = rfc.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], rfc.feature_importances_[sorted_idx])
# predictions
rfc_predict = rfc.predict(X_test)