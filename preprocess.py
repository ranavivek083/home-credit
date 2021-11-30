import pandas as pd
import numpy as np

bureau_features = pd.read_csv('./home-credit/features/bureau_features.csv')
bureau_features = bureau_features.fillna(0)

bureau_features.to_csv('./home-credit/features/bureau_features.csv', index=False)

# Modifying features.
bureau_features['percentage_active_loans'] = round(bureau_features['num_active_loans']*100/(bureau_features['num_active_loans'] + bureau_features['num_closed_loans']), 2)

del(bureau_features['num_closed_loans'])

bureau_features = bureau_features[['sk_id_curr', 'num_active_loans', 'percentage_active_loans',
       'num_bad_debt_loans', 'num_sold_off_loans', 'num_past_due_active_loans',
       'num_prolonged_loans', 'total_debt', 'total_credit', 'total_overdue',
       'avg_max_overdue', 'max_initial_dpd', 'num_initial_dpd',
       'num_loans_initial_dpd', 'max_dpd', 'num_dpd', 'num_loans_dpd',
       'ratio_debt_credit', 'ratio_overdue_debt', 'Consumer credit',
       'Credit card', 'Mortgage', 'Car loan', 'Microloan',
       'Loan for working capital replenishment',
       'Loan for business development', 'Real estate loan',
       'Unknown type of loan', 'Another type of loan',
       'Cash loan (non-earmarked)', 'Loan for the purchase of equipment',
       'Mobile operator loan', 'Interbank credit',
       'Loan for purchase of shares (margin lending)']]

bureau_features.to_csv('./home-credit/features/bureau_features.csv', index=False)