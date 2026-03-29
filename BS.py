import numpy as np

def compute_iv_surface(K, S, VIX, H, L, C, C_prev):
    """
    Compute implied volatility for a given strike using enhanced quant model.

    Parameters:
    K : float  -> Strike price
    S : float  -> Spot price
    VIX : float -> VIX value (in %)
    H : float  -> High price
    L : float  -> Low price
    C : float  -> Current close
    C_prev : float -> Previous close

    Returns:
    IV (in %) for given strike
    """

    # --- Step 1: Convert VIX to decimal ---
    sigma_vix = VIX / 100.0

    # --- Step 2: Realized volatility (Parkinson) ---
    log_hl = np.log(H / L)
    sigma_daily = np.sqrt((log_hl ** 2) / (4 * np.log(2)))
    sigma_realized = sigma_daily * np.sqrt(252)

    # --- Step 3: Return shock ---
    r = np.log(C / C_prev)
    r_ann_sq = (r ** 2) * 252

    # --- Step 4: ATM IV (enhanced model) ---
    var_atm = (
        0.55 * sigma_vix**2 +
        0.25 * sigma_realized**2 +
        0.10 * sigma_vix**2 +
        0.10 * r_ann_sq
    )
    sigma_atm = np.sqrt(var_atm)

    # --- Step 5: Moneyness ---
    m = np.log(K / S)

    # --- Step 6: Skew & Smile parameters ---
    alpha = -0.25
    beta = 3.5

    # --- Step 7: Final IV ---
    sigma_k = sigma_atm * (1 + alpha * m + beta * m**2)

    return sigma_k * 100  # return in %