import numpy as np
def compute_metrics(preds, gts):
    preds = np.concatenate(preds, axis=0).squeeze()
    gts = np.concatenate(gts, axis=0).squeeze()

    assert len(preds) == len(gts)
    print("cell num: ", len(preds))

    all_corr = []
    for i in range(gts.shape[1]):
        x = gts[:, i]
        y = preds[:, i]
        cor = np.corrcoef(x, y)[0][1]
        all_corr.append(cor)

    # Count NaN values
    nan_count = np.sum(np.isnan(all_corr))
    valid_corr = [corr for corr in all_corr if not np.isnan(corr)]  # Filter out NaN values
    num_valid = len(valid_corr)

    if num_valid > 0:
        metrics = {
            "nan_count": nan_count,
            "PCC-10": np.mean(sorted(valid_corr)[::-1][:min(10, num_valid)]) if num_valid >= 10 else float('nan'),
            "PCC-50": np.mean(sorted(valid_corr)[::-1][:min(50, num_valid)]) if num_valid >= 50 else float('nan'),
            f"PCC-{num_valid}": np.mean(valid_corr) if num_valid > 0 else float('nan'),
            "MSE": np.mean((gts - preds) ** 2),
            "MAE": np.mean(np.abs(gts - preds))
        }
    else:
        metrics = {
            "nan_count": nan_count,
            "PCC-10": float('nan'),
            "PCC-50": float('nan'),
            f"PCC-{num_valid}": float('nan'),
            "MSE": np.mean((gts - preds) ** 2),
            "MAE": np.mean(np.abs(gts - preds))
        }
    return metrics