## Welcome to Machine Learning for Finance in Python

First, let's explore the data. Any time we begin a machine learning (ML) project, we need to first do some exploratory data analysis (EDA) to familiarize ourselves with the data. This includes things like:

    raw data plots
    histograms
    and more...

I typically begin with raw data plots and histograms. This allows us to understand our data's distributions. If it's a normal distribution, we can use things like parametric statistics.

There are two stocks loaded for you into pandas DataFrames: lng_df and spy_df (LNG and SPY). Take a look at them with .head(). We'll use the closing prices and eventually volume as inputs to ML algorithms.

Note: We'll call plt.clf() each time we want to make a new plot, or f = plt.figure().

## Instructions
Print out the first 5 lines of the two DataFrame (lng_df and spy_df) and examine their contents.
    Use the pandas library to plot raw time series data for 'SPY' and 'LNG' with the adjusted close price ('Adj_Close') -- set legend=True in .plot().
    Use plt.show() to show the raw time series plot (matplotlib.pyplot has been imported as plt).
    Use pandas and matplotlib to make a histogram of the adjusted close 1-day percent difference (use .pct_change()) for SPY and LNG.
```markdown

print(lng_df.head())  # examine the DataFrames
print(spy_df.head())  # examine the SPY DataFrame

# Plot the Adj_Close columns for SPY and LNG
spy_df['Adj_Close'].plot(label='SPY', legend=True)
lng_df['Adj_Close'].plot(label='LNG', legend=True, secondary_y=True)
plt.show()  # show the plot
plt.clf()  # clear the plot space

# Histogram of the daily price change percent of Adj_Close for LNG
lng_df['Adj_Close'].pct_change().plot.hist(bins=50)
plt.xlabel('adjusted close 1-day percent change')
plt.show()
```

# Correlations

Correlations are nice to check out before building machine learning models, because we can see which features correlate to the target most strongly. Pearson's correlation coefficient is often used, which only detects linear relationships. It's commonly assumed our data is normally distributed, which we can "eyeball" from histograms. Highly correlated variables have a Pearson correlation coefficient near 1 (positively correlated) or -1 (negatively correlated). A value near 0 means the two variables are not linearly correlated.

If we use the same time periods for previous price changes and future price changes, we can see if the stock price is mean-reverting (bounces around) or trend-following (goes up if it has been going up recently).

## Instructions

Using the lng_df DataFrame and its Adj_Close:

1. Create the 5-day future price (as 5d_future_close) with pandas' .shift().
2. Use pct_change() on 5d_future_close and Adj_Close to create the % price
change 5 days in the future (5d_close_future_pct), and the current -5-day % 
price change (5d_close_pct).
3. Examine correlations between the two 5-day percent price change columns 
with .corr() on lng_df.
4. Using plt.scatter(), make a scatterplot of 5d_close_pct vs 5d_close_future_pct.

```markdown
# Create 5-day % changes of Adj_Close for the current day, and 5 days in the future
lng_df['5d_future_close'] = lng_df['Adj_Close'].shift(-5)
lng_df['5d_close_future_pct'] = lng_df['5d_future_close'].pct_change(5)
lng_df['5d_close_pct'] = lng_df['Adj_Close'].pct_change(5)

# Calculate the correlation matrix between the 5d close pecentage changes (current and future)
corr = lng_df[['5d_close_pct', '5d_close_future_pct']].corr()
print(corr)

# Scatter the current 5-day percent change vs the future 5-day percent change
plt.scatter(lng_df['5d_close_pct'], lng_df['5d_close_future_pct'])
plt.show()
```
