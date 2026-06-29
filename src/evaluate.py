import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate(y_true, y_pred_log):
    y_pred = np.expm1(y_pred_log)
    y_true_orig = np.expm1(y_true)
    
    r2 = r2_score(y_true_orig, y_pred)
    mae = mean_absolute_error(y_true_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred))
    
    print(f"R²:   {r2:.4f}")
    print(f"MAE:  €{mae:,.0f}")
    print(f"RMSE: €{rmse:,.0f}")
    return {"r2": r2, "mae": mae, "rmse": rmse}