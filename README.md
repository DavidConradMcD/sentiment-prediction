# Measuring the relationship between news headline sentiment and S&P 1200 movements

## Introduction
Predicting stock market movements has seen a rise in recent years due to the advancements predictive technologies such as machine learning, as well as the exponentially-growing data resources available to investors. The ability to “accurately” predict where a stock price (or stock market) is a seemingly profitable pursuit; one that pays high-performers very well. However, the notion that one can determine future price movements is still heavily scrutinized despite a growing trend towards predictive modelling. 

The Efficient Market Hypothesis (EMH) states that stock prices encompass all information about a given company and therefore cannot be predicted given their stochastic nature. However, although prediction methods are still far from where they need to be in terms of boasting consistent accuracy, there are still numerous academic studies, companies, and now, entire industries that argue against the EMH. 

Common market analysis techniques include technical analysis, fundamental analysis, and technological methods. Neural networks (machine learning strategies) fall into the latter and offer a greater ability to discover patterns in nonlinear and chaotic systems and can thus offer a more accurate prediction of market directions. 

We aim to explore a subset of machine learning stock market prediction techniques: sentiment analysis. By leveraging a machine learning algorithm called Vader Sentiment, we are able to construct daily sentiment scores for a subset of top global news headlines from 2009-2016. We use these scores as a proxy for investor opinion (or mood/sentiment) on a daily basis as it pertains to the global economy. We find that there exists a positive correlation between our sentiment score and the S&P 1200, and that including this sentiment score could aid in the predictive power of a stock index prediction model. 


## Data
We use three primary datasets for this project: The S&P Global 1200 index, the Reddit news dataset, and historical GDP metrics from the World Bank.

The S&P Global 1200 index is comprised of seven indices with stocks from 29 representative countries and is used as a benchmark for global equity markets. The seven indices which make up the S&P Global 1200 are: S&P Europe 350, S&P/TOPIX 150, S&P/TSX 60, S&P/ASX 50, S&P/Asia 50, S&P Latin America 40 and the S&P 700. 

The index captures approximately 70% of global market capitalization, making it a great source for which to compare global news headlines. 

![Dataset layout](https://i.pinimg.com/originals/39/43/dd/3943ddb09b74f4ec4b6cc0564b1e4715.png)




The Reddit news dataset is a collection of the top news headlines from around the world from 2008-2016. There are 25 top news headlines for each day (only counting weekdays since we are comparing it with S&P data) which provides a total of 41,975 news headlines. 

There are a total of 57 missing days of news headlines in this dataset. When merging this dataframe with the S&P Global 1200 dataframe, we removed the 57 missing days of data from both groups, which results in a balanced dataset.

For the World Bank data, we collected the GDPs of the top 75 countries (excluding regions) from 2009-2016. 


## Vader Sentiment Analyzer
To generate the sentiment scores, we use a pre-built model called Vader (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer. Vader is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media and other domains which primarily consist of shorter text strings.

The VADER sentiment lexicon is sensitive both the polarity (a floating number between -1 and +1. Minus one is extreme negativity, and plus one is extreme positivity) and the intensity (low to high) of sentiments expressed in text corpuses.

Vader makes use of a lexicon (list of words and corresponding sentiment scores) including more than 7,500 lexical features with validated valence scores that indicate both the sentiment polarity (positive/negative), and the sentiment intensity on a scale from –4 to +4. Sentiment ratings from 10 independent human raters (all pre-screened, trained, and quality checked to ensure accuracy) were generated to form the lexicon library. For example, the word "okay" has a positive valence of 0.9, "good" is 1.9, and "great" is 3.1, whereas "horrible" is –2.5, the frowning emoticon :( is –2.2, and "sucks" and it's slang derivative "sux" are both –1.5.

Applying the Vader algorithm to a string of text yields the following output:



![](https://i.pinimg.com/originals/d0/96/23/d096231b44f8fc4ce63e4540c7dcd77d.png)



Which returns a python dictionary containing a negative score, neutral score, positive score, and compound score. The compound score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive). This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence. Calling it a 'normalized, weighted composite score' is accurate. The formula:

-formula picture here

 is used to normalize the prior outputs, where x is the sum of the sentiment scores of the constituent words of the sentence and alpha is a normalization parameter that they set to 15.

Ultimately, we chose Vader because it offers a fairly robust sentiment analyzer which can accurately depict the sentiment of short text documents. In order to make the best use of this algorithm, we expanded the dictionary and applied a weighted average to the sentiment scores.

Merging world bank data with daily sentiment scores for global news headlines


![](https://i.pinimg.com/originals/ee/09/4f/ee094f68a54be19b819a9a3c609aa596.png)




Creating the merged data frame requires several steps which make use of efficient searching/sorting algorithms (See Appendix A). First, a sentiment score is applied to each headline, ranking how positive or negative it is (on a scale of -1 to 1). 

Next, a searching algorithm parses each headline to see if a country name is included. The algorithm loops through each headline, then loops through a list of lists, where each list element contains a subset of strings pertaining to that country name. For example: if a headline contains any elements in [‘United States’, ‘USA’, ‘ US ‘, ‘U.S’, ‘America’, ‘Obama’, ‘ Trump ‘,’ George Bush ‘] then it will return ‘TRUE’ for having mentioned the United States. It should be noted that elements such as ‘US’ are extracted as ‘ US ‘ (where there are spaces surround US) to ensure words such as: bUSy or defUSe are not considered as corresponding to the United States. 

The searching algorithm returns a list of lists, where each list element contains the corresponding country names. For example: the highest weighted article will have a corresponding list = [‘China’ , ’United States’] whereas the highest sentiment scoring article would have the list = [‘China’ , ‘N/A’] since there is only one country name mentioned. Next, the weight for each headline is equal to the sum of the GDPs (in billions) for the included countries in any given headline. The GDP used in this calculation corresponds to the year in which the news article was posted in order to account for changing degrees of country size throughout the sample period. This is why the GDP corresponding to China for the highest scoring sentiment headline is different than the GDP used for the highest weighted average headline: the articles appear in different years.

Finally, a weighted average is calculated by multiplying the weight by the sentiment score and dividing by the sum of the weights. 

Appendix A outlines this procedure in a visual manner. Applying a weighted average allows us to parse out headlines that don’t mention countries, and hence, focus primarily on headlines that would seemingly have a higher impact on the global economy. 

Appendix B shows our process for improving the accuracy of the sentiment score in regards to the problem at hand (explaining S&P 1200 price movements). We move from using Vader Lexicon on its own, then the Finance based Lexicon, then combining the Finance and Vader Lexicon, and finally to the weighted average Lexicon score, which applies a weighted average to the combined finance and Vader lexicon (a total of 14,500 words) which performs the best at normalizing the sentiment score distribution and hence, providing the most accurate measurement of the four methods we explored. Visually, it is clear that the Vader Lexicon is negatively-skewed, whereas the Finance Lexicon is more positively skewed. This is due to the fact that the word libraries to each Lexicon apply different scores to the same word. I.e. the Vader Lexicon applies a negative score to the word ‘cancer’ since it interprets this word as being negative, perhaps referring to someone being diagnosed with cancer. Whereas the Finance Lexicon applies a slightly positive score to the word ‘cancer’ because it interprets this phrase as referring to the pharmaceutical industry. Perhaps researchers discovered a new process for cancer research. Hence, in our model, the sentiment score takes into account that seemingly negative words may be used in a positive context, especially in the sense that the headlines dataset comments on global news, meaning that it is more likely to include articles talking about cancer research —  or the pharmaceutical industry —  as it is to include cases of someone (or a group of people) being diagnosed with cancer.





## Methodology


![](https://i.pinimg.com/originals/f4/11/54/f41154519c0ed1d3a74186983efd7368.png)



From the different sentiment manipulation, there were three different datasets obtained. The first comparison was between the different sentiment scores with the daily closing prices before moving on with the regression comparisons between the other indicators.
The model used here is a simple normal linear model:

y = β0 + β1 x1+ ε

closing price = β0 + β1sentiment score+ ε


![](https://i.pinimg.com/originals/b3/c2/21/b3c22192dae6b6ba95fd2399684990bb.png)



Figure 3.1 – Normal Linear Model
The interpretation of the coefficient here is that one unit of increase in the sentiment score increases the closing price by $148.42, but it is hard to set the unit of sentiment score to a specific number and the amount of increase is very large for the daily closing price.
Therefore, the logged normal linear model was used to see how the movement changes over the percent increase of sentiment. The log normal linear model is:
log(y) = β0 + β1 x1+ ε
log(closing price) = β0 + β1sentiment score + ε


![](https://i.pinimg.com/originals/bd/5b/3c/bd5b3cd9e97aac1cf4fd3c4a3444455c.png)



Figure 3.2 – logged Linear Model (Weighted Sentiment Scores)




The log linear model shows more consistent results therefore it is used in the rest of the regression models as well as for the different manipulations of the lexicon. By manipulating the lexicon, comparing the weighted sentiment scores with the none-weighted sentiment scores increased the correlation between the dependent and independent variables.



![](https://i.pinimg.com/originals/02/80/f7/0280f7c80db180f4bb30db7da7a04676.png)




Knowing that sentiment score is statistically significant towards the closing prices each day, three different indicators with close relations were searched and included in this analysis, the US Unemployment Rate, the US Urban CPI and the EU Area CPI. These indicators came in monthly versus the closing price data that were collected daily, when merging the two datasets together, the assumption of the unemployment rate, the CPIs of US and EU Area would be the same throughout the month so the data was copied through each day of the corresponding month.

A logged OLS regression was done for all indicators first to see the relations and significances using the model: 
log(y) = β0 + β1 x1+ β2 x2 + β3 x3 + ε
log(closing price) = β0 + β1US Unemployment Rate + β2 US CPI  + β3EU CPI + ε


![](https://i.pinimg.com/originals/98/24/ae/9824ae2da52bfc1db7f91b08f4966a47.png)



Figure 3.4 - logged Linear of Indicators on Closing Price
Most of the indicators are statistically significant for closing price so a regression on all independent variables was done with the following model: 
log(y) = β0 + β1 x1+ β2 x2 + β3 x3+ β4 x4 + ε
log(CP) = β0 + β1sentiment score + β2US Unemployment Rate + β3 US CPI + β4 EU CPI + ε





![](https://i.pinimg.com/originals/b8/3f/a7/b83fa7f203680d5a81c89e3075259df2.png)




Figure 3.5 - Logged linear regression of indicators on the closing price

Sentiment scores variable is not significant here as shown, after testing the relation between each indicator variable, it was found that sentiment score and unemployment rate was heavily correlated with each other. The decision of taking out the variable to continue regression was made. After taking out the unemployment rate variable, the regression model became: 
log(y) = β0 + β1 x1+ β2 x2 + β3 x3 + ε
log(closing price) = β0 + β1sentiment score + β2US CPI + β3 EU CPI + ε



![](https://i.pinimg.com/originals/ab/f8/08/abf80854799e73beab1a4310ed2e040d.png)



While conducting this analysis and reading different articles, the daily sentiment score is extremely volatile, therefore, the next step is to use the long-term moving averages of sentiment in a model. 




## Results
From the different methodologies above, the result as follows.
When comparing different manipulation of lexicon, the more words added to the dictionary the better the sentiment score which relates to higher significance towards the dependent variable. Moreover, there were news that is minor but made in the top 25 headlines so the manipulation of weighted country names was added to test the sentiment score. As shown in figure 3.2 and figure 3.3, the coefficient of sentiment score went from 0.049 to 0.092 when the sentiment score is weighted by the importance of each country using their size and GDP. The p-values for both regressions are extremely small so null hypothesis can be strongly rejected and it shows sentiment is a statistically significant variable in both cases. 
The global indicators were used as a base model for analysing the movement of closing prices, then adding the sentiment score to see the effect within the linear model. When using all three indicators, two of the independent variables become insignificant due to strong correlation between the independent variables. Putting it in the real world, unemployment rate should be correlated to the CPI of the country and have a strong effect on the sentiment score. After taking out the unemployment variable and regressing the other indicators on the closing price, the significance level grew. 
The regression result says, for every 1% increase of sentiment will increase the closing price by 0.2 cents while the US CPI would bring down closing price by 0.09 cents and EU CPI increases the closing price by 0.6 cents daily. This shows that if sentiment score on relative news is positive, meaning the news was good, then it will have the effect of increasing closing price of Dow Jones for that day. 



## Conclusions/Limitations
In conclusion, after the different analysis and methodologies, it concludes that the positive news can predict a rise in the closing price of Dow Jones on a daily basis. Comparing the results from this analysis with the articles reviewed, similar conclusions were made based on the different regression models and analysis. Not only with the result of the sentiment scores can affect the closing prices but also the different manipulations of lexicon can help reach better prediction of the movement of stocks. 
Some limitations were encountered during this project. Lag sentiment could have been done to see which lag period performs the best, also to see if the closing price today heavily depend on the closing price of yesterday. If time series were analysed, different time period indicators can be used to show the effect on sentiment scores and closing price. 
Moreover, the sentiment indicators were limited to news articles only, more data sources such as social media, buy/sell signals, and mutual fund flows could also be included. This can help to generate a better sentiment score towards the prediction of stock prices. In this project, it focused on a much larger market, it could be manipulated into focusing on a smaller subset of the market. If so then each news or indicator will have better connection and correlation with the market that implies better prediction model towards the daily stock closing price. 



## References
[1] Shah, D., Isah, H., & Zulkernine, F. (2018, December 10). Predicting the Effects of News Sentiments on the Stock Market .

[2] Shah, D., Isah, H., & Zulkernine, F. (2019, May 27). Stock Market Analysis: A Review and Taxonomy of Prediction Techniques.

[3] Wang, Z., & Lin, Z. (2018, November). Stock Market Prediction Analysis by Incorporating Social and News Opinion and Sentiment.

[4] Kalyanaraman, V., Kazi, S., Tondulkar, R., & Oswal, S. (2014, September). Sentiment Analysis on News Articles for Stocks.

[5] Wooldridge, J. M. (2015). Introductory econometrics: a modern approach (6th ed.). Boston, MA: Cengage Learning.


## Appendix A

![](https://i.pinimg.com/originals/3f/18/b9/3f18b9ea2e37a6849c1deed8c98afc52.png)


## Appendix B

![](https://i.pinimg.com/originals/0e/c8/56/0ec856e40d6caa1ed32fa926bac736c3.png)

Moving from Vader Lexicon to the Weighted average shows an decrease in upward/downward bias. For the Vader lexicon, sentiments look to mainly negative whereas with the finance-specific Lexicon, sentiments look to be only positive. Combining the two leads to a slightly better approximation, but still with few negative results. Applying a weighted average based on country name/size looks to show the best result for each day’s sentiment.


![](https://i.pinimg.com/originals/25/26/65/252665f06e61033dab01e1fbe34dc024.png)

Applying a 100-day simple moving average to the sentiment score shows a similar resemblance to that of daily closing prices of the S&P 1200.


![](https://i.pinimg.com/originals/42/45/12/424512c6a1e39b0825cca5d093cc3504.png)

Furthermore, applying a one year rolling average to both the S&P 1200 Closing price and the daily sentiment score show that the sentiment score looks to follow the general trend of the S&P 1200 but with much more volatility. 

This result helps to solidify the notion that there is some relationship between news sentiment and stock markets. Moving forward, we would look to use a correlation matrix for each combination of sentiment SMA’s to determine the best fit with the S&P 1200 and use the resulting metric in a predictive model trained on prior sentiments and closing prices. Clearly, a sentiment score can only predict a relatively small portion of volatility in a global market index given the massive scale of factors at play. The dataset we use in constructing this score is limited and does not contain a significant portion of stock-market related news. Using a dataset of relevant headlines exclusively should provide a better result given that it would be better suited for this type of classification problem. 


