import pandas as pd
import numpy as np
import datetime

bureau = pd.read_csv('./home-credit/data/bureau.csv')
credit_features = pd.read_csv('./home-credit/features/credit_features.csv')

# Lowercasing the columns.
bureau.columns = [x.lower() for x in bureau.columns]
bureau = bureau.rename(columns={'sk_id_bureau': 'sk_bureau_id'})

# Merge with bureau balance data.
bureau = pd.merge(bureau, credit_features, on=['sk_bureau_id'], how='left')
bureau[['amt_credit_sum_limit', 'amt_credit_sum', 'amt_credit_sum_debt', 'amt_credit_sum_overdue', 'amt_credit_max_overdue']] = bureau[['amt_credit_sum_limit', 'amt_credit_sum', 'amt_credit_sum_debt', 'amt_credit_sum_overdue', 'amt_credit_max_overdue']].fillna(0)

def get_bureau_features(bureau):
    bureau_features = []
    bureau_group = bureau.groupby(['sk_id_curr'], as_index=False)

    idx = 0
    for name, bgroup in bureau_group:
        print(idx)
        idx = idx + 1
        if idx % 1000 == 0:
            print(datetime.datetime.now(), " Percentage groups processed: ", round(idx * 100 / bureau_group.ngroups, 1))

        # Getting loans as a list.
        loans = bgroup.to_dict('records')
        num_active_loans = 0
        num_closed_loans = 0
        num_bad_debt_loans = 0
        num_sold_off_loans = 0
        num_past_due_active_loans = 0
        num_prolonged_loans = 0
        type_of_loans = {'Consumer credit': 0, 'Credit card': 0, 'Mortgage': 0, 'Car loan': 0, 'Microloan': 0,
                         'Loan for working capital replenishment': 0, 'Loan for business development': 0,
                         'Real estate loan': 0, 'Unknown type of loan': 0, 'Another type of loan': 0,
                         'Cash loan (non-earmarked)': 0, 'Loan for the purchase of equipment': 0,
                         'Mobile operator loan': 0, 'Interbank credit': 0,
                         'Loan for purchase of shares (margin lending)': 0}
        total_debt = 0
        total_credit = 0
        total_overdue = 0
        max_initial_dpd = bgroup['max_initial_dpd'].max()
        num_initial_dpd = bgroup['num_initial_dpd'].sum()
        num_loans_initial_dpd = bgroup['if_initial_dpd'].sum()
        max_dpd = bgroup['max_dpd'].max()
        num_dpd = bgroup['num_dpd'].sum()
        num_loans_dpd = bgroup['if_dpd'].sum()
        try:
            avg_max_overdue = round(bgroup['amt_credit_max_overdue'].sum()/bgroup.loc[bgroup['amt_credit_max_overdue'] > 0].shape[0], 2)
        except:
            avg_max_overdue = 0

        for loan in loans:
            if loan['credit_active'] == 'Closed':
                # Adding to closed loans.
                num_closed_loans = num_closed_loans + 1
            elif loan['credit_active'] == 'Active':
                # Adding to active loans.
                num_active_loans = num_active_loans + 1
                if loan['days_credit_enddate'] < 0:
                    num_past_due_active_loans = num_past_due_active_loans + 1
            elif loan['credit_active'] == 'Sold':
                num_sold_off_loans = num_sold_off_loans + 1
            else:
                num_bad_debt_loans = num_bad_debt_loans + 1

            if loan['cnt_credit_prolong'] > 0:
                num_prolonged_loans = num_prolonged_loans + 1

            type_of_loans[loan['credit_type']] = type_of_loans[loan['credit_type']] + 1
            total_credit = total_credit + loan['amt_credit_sum']
            total_debt = total_debt + loan['amt_credit_sum_debt']
            total_overdue = total_overdue + loan['amt_credit_sum_overdue']

        try:
            if total_debt < 0:
                ratio_debt_credit = 0
            else:
                ratio_debt_credit = round(total_debt/total_credit, 2)
        except:
            ratio_debt_credit = 0
        try:
            ratio_overdue_debt = round(total_overdue/total_debt, 2)
        except:
            ratio_overdue_debt = 0

        bureau_features.append({'sk_id_curr': name, 'num_active_loans': num_active_loans, 'num_closed_loans': num_closed_loans, 'num_bad_debt_loans': num_bad_debt_loans, 'num_sold_off_loans': num_sold_off_loans, 'num_past_due_active_loans': num_past_due_active_loans, 'num_prolonged_loans': num_prolonged_loans, 'total_debt': total_debt, 'total_credit': total_credit, 'total_overdue': total_overdue, 'avg_max_overdue': avg_max_overdue, 'max_initial_dpd': max_initial_dpd, 'num_initial_dpd': num_initial_dpd, 'num_loans_initial_dpd': num_loans_initial_dpd, 'max_dpd': max_dpd, 'num_dpd': num_dpd, 'num_loans_dpd': num_loans_dpd, 'ratio_debt_credit': ratio_debt_credit, 'ratio_overdue_debt': ratio_overdue_debt, **type_of_loans})

    bureau_features_df = pd.DataFrame(bureau_features)

    # Writing to csv.
    bureau_features_df.to_csv('./home-credit/features/bureau_features.csv', index=False)
