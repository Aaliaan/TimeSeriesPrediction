# Also known as triple exponential smoothing
# SERIES:
# Time aspect is irrelevant
# Time doesn't exist
# Only first, next, previous and last values
# Think of as a list of two-dimensional x and y values.
# x is order going up by 1 and y is the value
# OBSERVED vs EXPECTED:
# y_hat = expected value
# y_hat with subscript 4 will mean the expected value is 4
# METHOD:
# For series [1, 2, 3] we see each value is 1 greater than the previous
# In mathematical notation this is expressed as:
# hat{y}_{x + 1} = y_x + 1

# Naive method:
# Expected value is equal to the last value

series = [3, 10, 12, 13, 12, 10, 12]
# Naive method:
# Expected value is equal to the last value


# SIMPLE AVERAGE:
def average(series):
    return float(sum(series))/len(series)


# MOVING AVERAGE:
# Average of a sliding window 'n'
def moving_average(series, n):
    return average(series[-n:])


# WEIGHTED MOVING AVERAGE:
# Requires a list of weights that should add up to 1
def weighted_average(series, weights):
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series[-n-1] * weights[n]
    return result

# SIMPLE EXPONENTIAL SMOOTHING:
# Weights take average of all data points
# Exponentially get smaller as you go back in time
# \hat{y}_x = \alpha \cdot y_x + (1-\alpha) \cdot \hat{y}_{x-1} \\ RECURSIVE
# Alpha is a memory decay rate: higher the faster the method forgets


def exponential_smoothning(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])  # Just like the formula shown in LaTeX
    return result

# LEVEL:
# Expected value = baseline, intercept(y - intercept) or level
# Now we can't use \hat{y} instead use \ell
# TREND/SLOPE:
# In series the change in x will always be 1, therefore we calculate the slope or trend using:
# b = y_x - y_{x-1}
# ADDITIVE/MULTIPLICATIVE:
# y_x / y_{x-1}: getting thereby the ratio
# Method based on subtraction is additive and the method based on division is multiplicative
# DOUBLE EXPONENTIAL SMOOTHING:
# Exponential smoothing applied to both level and trend
# \ell_x = \alpha y_x + (1-\alpha)(\ell_{x-1} + b_{x-1}) - LEVEL
# b_x = \beta(\ell_x - \ell_{x-1}) + (1-\beta)b_{x-1}    - TREND
# \hat{y}_{x+1} = \ell_x + b_x                           - FORECAST
# \beta is the trend factor coefficient. As with \alpha, some values of \beta work better


def double_exponenital_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1-alpha) * (level + trend)
        result.append(level + trend)
    return result

# SEASON:
# If series is repetitive at regular intervals it is known as a season
# Required for Holt-Winters method to work
# Stock prices cannot be forcasted this way
# SEASON LENGTH:
# The number of data points after which a new season begins
# L to denote a season
# SEASONAL COMPONENT:
# Additional deviation from level and trend that reaps itself at the same offset into the season
# Seasonal component for every point in the season
# I.e if season length is 12 and there are 12 seasonal components we use s to denote the seasonal component
# ℓx=α(yx−sx−L)+(1−α)(ℓx−1+bx−1) : LEVEL
# bx=β(ℓx−ℓx−1)+(1−β)bx−1        : TREND
# sx=γ(yx−ℓx)+(1−γ)sx−L          : SEASONAL
# ŷ_x+m=ℓx+mbx+sx−L+1+(m−1)modL  : FORECAST
# \gamma is the smoothing factor for the seasonal component
# Expected value index is x + m, where m can be any integer - ability to forecast to any number of points
# Equation now consists of level, trend and seasonal component


series_l = [30, 21, 29, 31, 40, 48, 53, 47, 37, 39, 31, 29, 17, 9, 20, 24, 27, 35, 41, 38,
          27, 31, 27, 26, 21, 13, 21, 18, 33, 35, 40, 36, 22, 24, 21, 20, 17, 14, 17, 19,
          26, 29, 40, 31, 20, 24, 18, 26, 17, 9, 17, 21, 28, 32, 46, 33, 23, 28, 22, 27,
          18, 8, 17, 21, 31, 34, 44, 38, 31, 30, 26, 32]

# INITIAL TREND:
# We can observe many seasons and can extrapolate a better starting trend
# Most common practice is to compute the average of trend averages across seasons
# b_0 = \dfrac{1}{L}\left(\dfrac{y_{L+1}-y_1}{L}+\dfrac{y_{L+2}-y_2}{L}+...+\dfrac{y_{L+L}-y_L}{L}\right)


def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen


# INITIAL SEASON COMPONENTS:
# Complicated when it comes to initial values for the seasonal components
# Need to compute the average level for every observed season
# Divide every observed value by the average for the season it's in and finally average of the numbers across seasons
#
def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen * j:slen * j + slen]) / float(slen))
        # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen * j + i] - season_averages[j]
        seasonals[i] = sum_of_vals_over_avg / n_seasons
    return seasonals


def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result
