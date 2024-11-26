import numpy as np
import pandas as pd
import statsmodels.api as sm 
import statsmodels  
import scipy 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error










def _durbin(resid: np.ndarray) -> str:
    dw_statistic = statsmodels.stats.stattools.durbin_watson(resid)
    print(f"Durbin-Watson Statistic: {dw_statistic}")
    
    if dw_statistic > 1.5 and dw_statistic < 2.5:
        print("There’s no significant autocorrelation in your residuals.")
    elif dw_statistic < 1.5:
        print("Presence of negative autocorrelation")
    else:
        print("Presence of positive autocorrelation")

def _ljbox(resid: np.ndarray, lags: int) -> str:
    ljung_box_test = statsmodels.stats.diagnostic.acorr_ljungbox(resid, lags=[lags], return_df=True)
    p_value = ljung_box_test['lb_pvalue'].iloc[0]
    test_stat = ljung_box_test['lb_stat'].iloc[0]
    
    print(f"Ljung-Box Test Statistic: {test_stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Autocorrelation detected in the residuals (reject null hypothesis).")
    else:
        print("No significant autocorrelation detected in the residuals (do not reject null hypothesis).")

def _jarqueb(resid: np.ndarray) -> str:
    jb_statistic, jb_pvalue = scipy.stats.jarque_bera(resid)
    print(f"Jarque-Bera Statistic: {jb_statistic}, p-value: {jb_pvalue}")
    if jb_pvalue < 0.05:
        print("Residuals are not normally distributed (reject null hypothesis).")
    else:
        print("Residuals are normally distributed (do not reject null hypothesis).")

def _shapiro(resid: np.ndarray) -> str:
    shp_stats, shap_pvalue = scipy.stats.shapiro(resid)
    print(f"Shapiro-Wilk Statistic: {shp_stats}, p-value: {shap_pvalue}")
    if shap_pvalue < 0.05:
        print("Residuals are not normally distributed (reject null hypothesis).")
    else:
        print("Residuals are normally distributed (do not reject null hypothesis).")

def _kolmogorov_sm(resid: np.ndarray) -> str:
    ks_statistic, p_value = scipy.stats.kstest(resid, 'norm', args=(np.mean(resid),np.std(resid)))
    print(f"Kolmogorov-Smirnov Statistic: {ks_statistic}, p-value: {p_value}")
    if p_value < 0.05:
        print("Residuals do not follow a normal distribution (reject null hypothesis).")
    else:
        print("Residuals are approximately normally distributed (do not reject null hypothesis).")

def _arch(series: np.ndarray) -> str:
    arch_stat, p_value = statsmodels.stats.diagnostic.het_arch(series)
    print(f"ARCH Test Statistic: {arch_stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Decision: Reject the null hypothesis. There is evidence of ARCH effects.")
    else:
        print("Decision: Fail to reject the null hypothesis. No significant ARCH effects found.")

def _white(resid: np.ndarray, exog: np.ndarray) -> str:
    # Perform White's test
    lm_stat, lm_pvalue, f_stat, f_pvalue = statsmodels.stats.diagnostic.het_white(resid=resid, exog=sm.add_constant(exog))
    
    # Print test results
    print(f"White Test Statistic (LM): {lm_stat}, p-value: {lm_pvalue}")
    print(f"White Test Statistic (F): {f_stat}, p-value: {f_pvalue}")
    
    # Interpret the result using the LM p-value
    if lm_pvalue < 0.05:
        return "Decision: Reject the null hypothesis. Evidence of heteroskedasticity."
    else:
        return "Decision: Fail to reject the null hypothesis. No significant heteroskedasticity found."

def _breuschpg(resid: np.ndarray, exog: np.ndarray) -> str:
    bp_stat, p_value, _, _ = statsmodels.stats.diagnostic.het_breuschpagan(resid=resid, exog_het=sm.add_constant(exog))
    print(f"Breusch-Pagan Test Statistic: {bp_stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Decision: Reject the null hypothesis. There is evidence of heteroskedasticity.")
    else:
        print("Decision: Fail to reject the null hypothesis. No significant heteroskedasticity found.")


def detect_multicollinearity(data: pd.DataFrame, threshold: float = 30) -> str:
    """
    Detect multicollinearity in a dataset using the Eigenvalue Method.
    
    Parameters:
    - data (pd.DataFrame): A DataFrame containing the independent variables (predictors).
    - threshold (float): The threshold for the condition number to indicate multicollinearity. Default is 30.
    
    Returns:
    - str: Message indicating whether multicollinearity is detected or not.
    """
    # Step 1: Calculate the correlation matrix
    corr_matrix = data.corr()

    # Step 2: Perform eigenvalue decomposition
    eigvals, _ = np.linalg.eig(corr_matrix)

    # Step 3: Calculate the condition number
    condition_number = max(eigvals) / min(eigvals)
    
    # Step 4: Check if the condition number exceeds the threshold
    if condition_number > threshold:
        return f"Multicollinearity detected. Condition number: {condition_number:.2f}. (Threshold: {threshold})"
    else:
        return f"No significant multicollinearity detected. Condition number: {condition_number:.2f}. (Threshold: {threshold})"
def calculate_vif(data:pd.DataFrame):
    """
    Calculate and interpret VIF for each variable in the DataFrame.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the independent variables.
    
    Returns:
    pd.DataFrame: DataFrame with VIF values and interpretation for each variable.
    """
    # Add a constant to the independent variables
    X = data
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Add interpretation column
    interpretations = []
    for vif in vif_data["VIF"]:
        if vif == 1:
            interpretation = "No multicollinearity"
        elif 1 < vif < 5:
            interpretation = "Moderate correlation, acceptable"
        elif 5 <= vif < 10:
            interpretation = "High correlation, possible multicollinearity concern"
        else:
            interpretation = "Strong multicollinearity, consider removing variable"
        interpretations.append(interpretation)
    
    vif_data["Interpretation"] = interpretations
    
    return vif_data



def _bootstrap(X, y, n_samples=1000):
    
    """
    Perform bootstrap sampling to fit OLS models and check normality of residuals.

    Parameters:
    X (np.ndarray or pd.DataFrame): Independent variables (including constant if necessary).
    y (np.ndarray or pd.Series): Dependent variable.
    n_samples (int): Number of bootstrap samples to generate.

    Returns:
    list: List of bootstrapped samples of X and y where residuals are normal.
    """
    
    X_, y_ = [], []

    for i in range(n_samples):
        idx = np.random.choice(range(len(y)), 1000, replace=True)
        model = sm.OLS(y[idx], X[idx]).fit()

        #normality test
        ks_statistic, p_value = scipy.stats.jarque_bera(model.resid)


        if p_value > 0.05:
            X_.append(X[idx])
            y_.append(y[idx])

        

    return X_, y_



def _fit_model_sm(estimator:object, X:np.ndarray=None, y:np.ndarray=None, data:pd.DataFrame=None,formula:str = None) -> pd.DataFrame:


    if formula:
        estimator = estimator.ols(formula= formula, data=data)
    fit_model = estimator.fit()
    results = fit_model.summary()

    return results, fit_model



# R-squared function
def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    ss_residual = np.sum((y_true - y_pred) ** 2)        # Residual sum of squares
    r2 = 1 - (ss_residual / ss_total)
    return r2

# RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# MAE function
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
