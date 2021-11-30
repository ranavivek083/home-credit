# Importing libraries.
import datetime
import pandas as pd
import numpy as np
import statistics

# Reading bureau and bureau monthly balance data.
# bureau = pd.read_csv('./data/bureau.csv')
bureau_balance = pd.read_csv('./home-credit/data/bureau_balance.csv')

# Lowercasing the columns.
# bureau.columns = [x.lower() for x in bureau.columns]
bureau_balance.columns = [x.lower() for x in bureau_balance.columns]

# Sorting by sk_id and month.
bureau_balance = bureau_balance.sort_values(['sk_id_bureau', 'months_balance'])


def get_dpd_features(repayments):
    if_dpd = 0
    dpd_info = []
    num_unknowns = 0
    num_months = max([-x['months_balance'] + 1 for x in repayments])
    if_credit_closed = 0
    months_taken_to_close = 0
    avg_dpd = None
    max_dpd = None
    num_dpd = 0

    for i in range(0, len(repayments)):
        # Skip if credit is closed.
        if repayments[i]['status'] == 'C':
            if_credit_closed = 1
            break
        else:
            # For unknown status of repayment.
            if repayments[i]['status'] == 'X':
                num_unknowns = num_unknowns + 1
            else:
                if repayments[i]['status'] != '0':
                    # Delayed payment.
                    if_dpd = 1
                    dpd_info.append(int(repayments[i]['status']))

        months_taken_to_close = months_taken_to_close + 1

    if if_credit_closed == 0:
        months_taken_to_close = None

    if len(dpd_info) > 0:
        num_dpd = len(dpd_info)
        avg_dpd = statistics.mean(dpd_info)
        max_dpd = max(dpd_info)

    return {'total_months': num_months, 'if_dpd': if_dpd, 'num_unknowns': num_unknowns, 'if_credit_closed': if_credit_closed, 'num_months_closed': months_taken_to_close, 'avg_dpd': avg_dpd, 'max_dpd': max_dpd, 'num_dpd': num_dpd}


def get_features(bureau_balance, initial_months_threshold=3):
    # Container to store repayment features of a credit.
    credit_repayment_features = []
    credit_group = bureau_balance.groupby(['sk_id_bureau'], as_index=False)

    idx = 0
    for group_name, group in credit_group:
        idx = idx + 1
        if idx % 1000 == 0:
            print(datetime.datetime.now(), " Percentage groups processed: ", round(idx*100/credit_group.ngroups, 1))
        # Getting repayments as a list.
        repayments = group.to_dict('records')

        # Separating initial repayments and setting first 3 months as initial repayment months.
        months_initial_repayments = len(repayments)
        if months_initial_repayments > initial_months_threshold:
            months_initial_repayments = initial_months_threshold

        initial_repayment_features = get_dpd_features(repayments[0:months_initial_repayments])
        if_initial_dpd = initial_repayment_features['if_dpd']
        avg_initial_dpd = initial_repayment_features['avg_dpd']
        max_initial_dpd = initial_repayment_features['max_dpd']
        num_initial_dpd = initial_repayment_features['num_dpd']
        num_initial_unknowns = initial_repayment_features['num_unknowns']

        total_repayment_features = get_dpd_features(repayments)
        if_dpd = total_repayment_features['if_dpd']
        num_months = total_repayment_features['total_months']
        if_credit_closed = total_repayment_features['if_credit_closed']
        num_months_closed = total_repayment_features['num_months_closed']
        num_unknowns = total_repayment_features['num_unknowns']
        avg_dpd = total_repayment_features['avg_dpd']
        max_dpd = total_repayment_features['max_dpd']
        num_dpd = total_repayment_features['num_dpd']

        credit_repayment_features.append({'sk_id_bureau': group_name, 'if_initial_dpd': if_initial_dpd, 'avg_initial_dpd': avg_initial_dpd, 'max_initial_dpd': max_initial_dpd, 'num_initial_dpd': num_initial_dpd, 'num_initial_unknowns': num_initial_unknowns, 'if_credit_closed': if_credit_closed, 'if_dpd': if_dpd, 'months_elapsed': num_months, 'num_months_closed': num_months_closed, 'num_unknowns': num_unknowns, 'avg_dpd': avg_dpd, 'max_dpd': max_dpd, 'num_dpd': num_dpd})

    credit_repayment_features_df = pd.DataFrame(credit_repayment_features)
    credit_repayment_features_df.to_csv('./home-credit/data/credit_features.csv', index=False)















