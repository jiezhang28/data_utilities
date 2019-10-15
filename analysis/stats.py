import pandas as pd
import statsmodels.api as sm
from scipy import stats

def get_odds_ratio(frame, feature, target, target_value, ret_type=None):
    """
    Returns odds ratio for all values in the feature column against the target
    value

    Parameters
    ----------
    frame        : pandas DataFrame
    feature      : name of the feature column in the DataFrame
    target       : name of the target column in the DataFrame
    target_value : the specific target value to test against
    ret_type     : specifies how to format the results in the return. If 'dict'
                   then returns dictionary object that is dataframe-friendly.
                   DEFAULT: None, 2-d array.

    Return
    ------
    Odd ratios, Pvalue, and CI for each feature value against the target value
    """
    results = []
    feature_values = frame[feature].unique()
    for v in feature_values:
        odds_tbl = pd.DataFrame({
            v : frame[feature]==v,
            't': frame[target]==target_value,
            'cnt' : 1})

        odds_tbl = (odds_tbl
                    .groupby(['t', v])
                    .count()['cnt']
                    .unstack()
                    .fillna(0)
                    .sort_index(ascending=False)
                    .sort_index(ascending=False, axis=1))

        odds_ratio, p_value = stats.fisher_exact(odds_tbl)
        ci_lower, ci_upper = sm.stats.Table2x2(odds_tbl).oddsratio_confint()

        results.append([
            target_value,
            v,
            odds_ratio,
            p_value,
            ci_lower,
            ci_upper,
            odds_tbl
        ])

    if ret_type == 'dict':
        d = defaultdict(list)
        headers = ['target','value','odds_ratio','p_value','ci_lower','ci_upper','contingency_tbl']
        for r in results:
            for k, v in zip(headers, r):
                d[k].append(v)

        results = d

    return results

