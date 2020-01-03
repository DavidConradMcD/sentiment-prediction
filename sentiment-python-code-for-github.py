from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt, numpy as np
from statistics import mean
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
import datetime
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer

analyzer = SentimentIntensityAnalyzer()

pd.set_option('display.float_format', lambda x: '%.3f' % x)



reddit_news = pd.read_csv('/Users/david/Desktop/Desktop/Year 4 Fall/Econometrics/project-datasets/Combined_News_DJIA.csv')
reddit_news['Date'] = pd.to_datetime(reddit_news['Date'])
reddit_news = reddit_news[reddit_news['Date'] >= '2009-10-30']
reddit_news.reset_index(inplace=True)
reddit_news.drop(columns='index',inplace=True)
reddit_news.head()




reddit_news['Top1'] = [i.strip("b'") for i in reddit_news['Top1']]
reddit_news['Top2'] = [i.strip("b'") for i in reddit_news['Top2']]
reddit_news['Top3'] = [i.strip("b'") for i in reddit_news['Top3']]
reddit_news['Top4'] = [i.strip("b'") for i in reddit_news['Top4']]
reddit_news['Top5'] = [i.strip("b'") for i in reddit_news['Top5']]
reddit_news['Top6'] = [i.strip("b'") for i in reddit_news['Top6']]
reddit_news['Top7'] = [i.strip("b'") for i in reddit_news['Top7']]
reddit_news['Top8'] = [i.strip("b'") for i in reddit_news['Top8']]
reddit_news['Top9'] = [i.strip("b'") for i in reddit_news['Top9']]
reddit_news['Top10'] = [i.strip("b'") for i in reddit_news['Top10']]
reddit_news['Top11'] = [i.strip("b'") for i in reddit_news['Top11']]
reddit_news['Top12'] = [i.strip("b'") for i in reddit_news['Top12']]
reddit_news['Top13'] = [i.strip("b'") for i in reddit_news['Top13']]
reddit_news['Top14'] = [i.strip("b'") for i in reddit_news['Top14']]
reddit_news['Top15'] = [i.strip("b'") for i in reddit_news['Top15']]
reddit_news['Top16'] = [i.strip("b'") for i in reddit_news['Top16']]
reddit_news['Top17'] = [i.strip("b'") for i in reddit_news['Top17']]
reddit_news['Top18'] = [i.strip("b'") for i in reddit_news['Top18']]
reddit_news['Top19'] = [i.strip("b'") for i in reddit_news['Top19']]
reddit_news['Top20'] = [i.strip("b'") for i in reddit_news['Top20']]
reddit_news['Top21'] = [i.strip("b'") for i in reddit_news['Top21']]
reddit_news['Top22'] = [i.strip("b'") for i in reddit_news['Top22']]
reddit_news['Top23'] = [i.strip("b'") for i in reddit_news['Top23']]
reddit_news['Top24'] = [i.strip("b'") for i in reddit_news['Top24'].astype(str)]
reddit_news['Top25'] = [i.strip("b'") for i in reddit_news['Top25'].astype(str)]




### Loading in Financials for Dow Jones ###
financials = pd.read_csv("/Users/david/Downloads/^GSPC (1).csv")
financials.rename(columns={"Date": "datef"}, inplace=True)
financials.head()





### Importing S&P 1200 Global Index ###
sp_1200 = pd.read_csv("/Users/david/Downloads/sp_1200.csv")
sp_1200.plot()




### Updating Vader with Financial Lexicon
fin_lex = pd.read_json("/Users/david/Downloads/NTUSD-Fin/NTUSD_Fin_word_v1.0.json")
fin_lex = fin_lex[['token','market_sentiment']]
word = fin_lex['token']
sentiment = fin_lex['market_sentiment']
fin_lex.head()




### Updating the Lexicon with a short list of new words
new_dict = dict(zip(word, sentiment))
analyzer.lexicon.update(new_dict)
new_words = {
    'refuses': -1.2,
    'spill': -1,
    'extraordinary':1.5,
    'discovered':0.6,
    'discovery':0.6,
    'good':2.5, #bad is -2.5 so I made good +2.5 instead of +1.9
    'negotiate':0.8,
    'deadly':-0.5,
    'hospitalized':-0.8,
    'clash':-0.4,
}

analyzer.lexicon.update(new_words)




### Generating sentiment scores for each headline
vader_stats = []
hline_sent = []
for i in range(0,1679):
    for headline in reddit_news.iloc[i][2:27]:
        try:
            hline_sent.append([i,float(analyzer.polarity_scores(headline)['compound'])])
        except AttributeError as Error:
            pass

sentiment_scores = pd.DataFrame(hline_sent)
sentiment_scores.columns = ['day','sentiment']


### Parsing the world bank dataset ###
test1 = pd.read_csv('/Users/david/Downloads/world_bank.csv')
test2 = test1.iloc[:,53:61]
test3 = test1.iloc[:,0]
test5 = test2.join(test3, how='right')
columnsTitles = ['Country Name', '2012', '2013', '2014', '2015', '2016']


test6 = test5.dropna(how='any')

test7 = test6.sort_values('2012', ascending=False)
test8 = test7.reset_index()
test9 = test8.drop(columns='index')
test10 = test9[:120]
test10["Name Length"]= test10["Country Name"].str.len()
test11 = test10[test10["Name Length"] < 22]
test12 = test11[12:]
to_drop = ['Euro area', 'Lower middle income', 'Sub-Saharan Africa', 'IDA blend', 'IDA only', 'Arab World'          ,'South Asia','IDA total', 'Small states','Other small states']
#df3 = df2[~df2['category'].isin(to_drop)]
test13 = test12[~test12['Country Name'].isin(to_drop)]
pd.set_option('display.max_rows', 300)
test14 = test13.drop(columns = ['Name Length'])
test15 = test14.reset_index()
test16 = test15.drop(columns = ['index'])
world_bank = test16
world_bank.head()



c_name_list = pd.read_csv("/Users/david/Downloads/country_names - Sheet1.csv",header=None)
c_name_list.columns = ['name_one','name_two','test1','test2']

c_name_list1 = c_name_list.drop(columns=['test1','test2'])
countries_list = list(c_name_list1['name_one'])

world_bank3 = world_bank.sort_values(['Country Name'], ascending=[1])
world_bank4 = world_bank3.reset_index()
world_bank4 = world_bank4[world_bank4['Country Name'] != world_bank4['Country Name'][40]]
world_bank4 = world_bank4.reset_index()
world_bank = world_bank4.drop(columns = ['index', 'level_0'])


### Creating Lists to store country names
Algeria = []
Angola = []
Argentina = []
Australia = []
Austria = []
Azerbaijan = []
Bangladesh = []
Belarus = []
Belgium = []
Brazil = []
Bulgaria = []
Canada = []
Chile = []
China = []
Colombia = []
Croatia = []
Cuba = []
Czech_Republic = []
Denmark = []
Dominican_Republic = []
Ecuador = []
Egypt = []
Finland = []
France = []
Germany = []
Greece = []
Hong_Kong = []
Hungary = []
India = []
Indonesia = []
Iran = []
Iraq = []
Ireland = []
Israel = []
Italy = []
Japan = []
Kazakhstan = []
Korea = []
Kuwait = []
Libya = []
Luxembourg = []
Malaysia = []
Mexico = []
Morocco = []
Myanmar = []
Netherlands = []
New_Zealand = []
Nigeria = []
Norway = []
Oman = []
Pakistan = []
Peru = []
Philippines = []
Poland = []
Portugal = []
Puerto_Rico = []
Qatar = []
Romania = []
Russia = []
Saudi_Arabia = []
Singapore = []
Slovakia = []
South_Africa = []
Spain = []
Sri_Lanka = []
Sudan = []
Sweden = []
Switzerland = []
Thailand = []
Turkey = []
Ukraine = []
United_Arab_Emirates = []
United_Kingdom = []
United_States = []
Vietnam = []


Algeria1 = ['Algeria', 'Algerian']
Angola1 = ['Angola', 'Angolan']
Argentina1 = ['Argentina', 'Argentine']
Australia1 = ['Australia', 'Australian']
Austria1 = ['Austria', 'Austrian']
Azerbaijan1 = ['Azerbaijan']
Bangladesh1 = ['Bangladesh', 'Bangladeshi']
Belarus1 = ['Belarus', 'Belarusian']
Belgium1 = ['Belgium', 'Belgian']
Brazil1 = ['Brazil', 'Brazilian']
Bulgaria1 = ['Bulgaria', 'Bulgarian']
Canada1 = ['Canada', 'Canadian','Trudeau']
Chile1 = ['Chile', 'Chilean']
China1 =  ['China', 'Chinese', 'Xi Jinping', ' Jinping ']
Colombia1 = ['Colombia', 'Colombian']
Croatia1 = ['Croatia', 'Croatian']
Cuba1 = ['Cuba', 'Cuban']
Czech_Republic1 = ['Czech Republic', 'Czech']
Denmark1 = ['Denmark', 'Danish']
Dominican_Republic1 = ['Dominican Republic', 'Dominican']
Ecuador1 = ['Ecuador', 'Ecuadorian']
Egypt1 = ['Egypt', 'Egyptian']
Finland1 = ['Finland', 'Finnish']
France1 = ['France', 'French']
Germany1 = ['Germany', 'German']
Greece1 = ['Greece', 'Greek']
Hong_Kong1 = ['Hong Kong']
Hungary1 = ['Hungary', 'Hungarian']
India1 = ['India', 'Indian']
Indonesia1 = ['Indonesia', 'Indonesian']
Iran1 = ['Iran', 'Iranian']
Iraq1 = ['Iraq', 'Iraqi']
Ireland1 = ['Ireland', 'Irish']
Israel1 = ['Israel', 'Israeli']
Italy1 = ['Italy', 'Italian']
Japan1 = ['Japan', 'Japanese']
Kazakhstan1 = ['Kazakhstan', 'Kazakh']
Korea1 = ['Korea', 'Korean']
Kuwait1 = ['Kuwait']
Libya1 = ['Libya', 'Libyan']
Luxembourg1 = ['Luxembourg']
Malaysia1 = ['Malaysia', 'Malaysian']
Mexico1 = ['Mexico', 'Mexican']
Morocco1 = ['Morocco', 'Moroccan']
Myanmar1 = ['Myanmar']
Netherlands1 = ['Netherlands', "Netherland's"]
New_Zealand1 = ['New Zealand']
Nigeria1 = ['Nigeria', 'Nigerian']
Norway1 = ['Norway', 'Norwegian']
Oman1 = ['Oman', 'Omani']
Pakistan1 = ['Pakistan', 'Pakistani']
Peru1 = ['Peru', 'Peruvian']
Philippines1 = ['Philippines', 'Philippine']
Poland1 = ['Poland', 'Polish']
Portugal1 = ['Portugal', 'Portuguese']
Puerto_Rico1 = ['Puerto Rico', 'Peuerto Rican']
Qatar1 = ['Qatar']
Romania1 = ['Romania', 'Romanian']
Russia1 = ['Russia', 'Russian', 'Putin']
Saudi_Arabia1 = ['Saudi Arabia', 'Saudi Arabian']
Singapore1 = ['Singapore', 'Singaporean']
Slovakia1 = ['Slovakia', 'Slovak']
South_Africa1 = ['South Africa', 'South African']
Spain1 = ['Spain', 'Spanish']
Sri_Lanka1 = ['Sri Lanka', 'Sri Lankan']
Sudan1 = ['Sudan', 'Sudanese']
Sweden1 = ['Sweden', 'Swedish']
Switzerland1 = ['Switzerland', 'Swiss']
Thailand1 = ['Thailand', 'Thai']
Turkey1 = ['Turkey', 'Turkish']
Ukraine1 = ['Ukraine', 'Ukrainian']
United_Arab_Emirates1 = ['United Arab Emirates', 'Emirati']
United_Kingdom1 = ['United Kingdom', ' UK ', 'British', 'UK', 'Britain', 'Scotland', 'Scottish']
United_States1 = ['United States', 'American', 'U.S', ' USA ', 'Obama', 'Donald Trump', ' Trump ', ' Bush ']
Vietnam1 = ['Vietnam', 'Vietnamese']


final_list_countries = [Algeria1,Angola1,Argentina1,Australia1,Austria1,Azerbaijan1,Bangladesh1,Belarus1,Belgium1,\
                        Brazil1,Bulgaria1,Canada1,Chile1,China1,Colombia1,Croatia1,Cuba1,Czech_Republic1,Denmark1,\
                        Dominican_Republic1,Ecuador1,Egypt1,Finland1,France1,Germany1,Greece1,Hong_Kong1,Hungary1,\
                        India1,Indonesia1,Iran1,Iraq1,Ireland1,Israel1,Italy1,Japan1,Kazakhstan1,Korea1,Kuwait1,\
                        Libya1,Luxembourg1,Malaysia1,Mexico1,Morocco1,Myanmar1,Netherlands1,New_Zealand1,Nigeria1,\
                        Norway1,Oman1,Pakistan1,Peru1,Philippines1,Poland1,Portugal1,Puerto_Rico1,Qatar1,Romania1,\
                        Russia1,Saudi_Arabia1,Singapore1,Slovakia1,South_Africa1,Spain1,Sri_Lanka1,Sudan1,Sweden1,\
                        Switzerland1,Thailand1,Turkey1,Ukraine1,United_Arab_Emirates1,United_Kingdom1,United_States1,Vietnam1]


test_cty_list = []
for i in final_list_countries:
    for x in i:
        test_cty_list.append(x)



### Creating a list of lists that contains the specific keywords used
### in each Headline (allowing for multiples)
empty_country_list = [ [] for i in range(len(reddit_news)) ]

for day in range(len(empty_country_list)):
    for idx,i in enumerate(reddit_news.iloc[day][2:27]):
        if any(cname in i for cname in test_cty_list):
            empty_country_list[day].append([i,day,[element for element in test_cty_list if element in i]])
        else:
            empty_country_list[day].append("Nope")



### Creating a list that contains the Real country names per each
### article, (not just the alternate names)
real_names = []
for i in range(len(empty_country_list)):
    for j in range(0,25):
        real_names.append([empty_country_list[i][j][2]])


real_names2 = []
for i in real_names:
    for j in i:
        real_names2.append(j)




### Creating a list that stores every instance of a country name appearing,
###given the actual country name that appears in the world_bank dataframe
prelim_list = [ [] for i in range(len(real_names2)) ]
final_list = []
for idx,i in enumerate(real_names2):
    if i == 'p':
        final_list.append([np.nan,np.nan])
        pass
    else:
        for r in range(0,75):
            try:
                if real_names2[idx][0] in final_list_countries[r]:
                    prelim_list[idx].append(final_list_countries[r][0])
                if real_names2[idx][1] in final_list_countries[r]:
                    prelim_list[idx].append(final_list_countries[r][0])
                else:
                    pass
            except IndexError as Error:
                    continue
    final_list.append(prelim_list[idx])



### Make it a 1679*2 Matrix
new_final = [x for x in final_list if x != []]
for i in range(len(new_final)):
    if str(new_final[i][0]) != 'nan' or len(new_final[i]) != 1:
        try:
            if new_final[i][0] == new_final[i][1]:
                new_final[i] = [new_final[i][0]]
        except IndexError as Error:
            continue
    else:
        pass
for i in new_final:
    if len(i) < 2:
        i.append(np.nan)
    if len(i) > 3:
        [i,np.nan]



### Turn the matrix into a dataframe
new_df = pd.DataFrame(new_final, columns = ['name one','name two'])


### Getting the GDP for each corresponding country (and corresponding year)
### and appending to a list
test_list_one = []
day_one = 0
day_list = [ [] for i in range(0,1679) ]
for idx,i in enumerate(day_list):
    for j in range(0,25):
        test_list_one.append(idx)

len(test_list_one[0:500])

### Getting Base GDP values for each mentioned country name
n1_gdp_base = []
for idx, i in enumerate(test_list_one):
    n1_gdp_base.append(world_bank[world_bank['Country Name'] == new_df['name one'][idx]][str(reddit_news['Date'][i].year)])

n2_gdp_base = []
for idx, i in enumerate(test_list_one):
    n2_gdp_base.append(world_bank[world_bank['Country Name'] == new_df['name two'][idx]][str(reddit_news['Date'][i].year)])

n1_gdp_main = []
for i in n1_gdp_base:
    try:
        n1_gdp_main.append(int(i)/1000000000)
    except TypeError as error:
        n1_gdp_main.append(np.nan)

n2_gdp_main = []
for i in n2_gdp_base:
    try:
        n2_gdp_main.append(int(i)/1000000000)
    except TypeError as error:
        n2_gdp_main.append(np.nan)




### Add the GDP values to the df of country names occurring in each article
new_df['name one gdp (billions)'] = n1_gdp_main
new_df['name two gdp (billions)'] = n2_gdp_main
new_df['sentiment'] = sentiment_scores['sentiment']



### Create weight column which sums the gdps for each headline. If no country mentioned,
### give a weight of 1 so that it will multiply properly.
test_articles_lst = []
for day in range(0,1679):
    for idx,j in enumerate(reddit_news.iloc[day][2:27]):
        test_articles_lst.append(j)


sent_test_lst = []
for i in test_articles_lst:
    try:
        sent_test_lst.append(analyzer.polarity_scores(i))
    except AttributeError as Error:
        pass




### Creating a column for weight * sentiment (where each weight is the sum of country weights for that given headline).
### If there are no country names mentioned, the weight*sentiment is simply just the sentiment score,
### giving it much less importance than those headlines mentioning a country
new_df['weight'] = new_df[['name one gdp (billions)', 'name two gdp (billions)']].sum(axis=1)
new_df['day'] = test_list_one
new_df.loc[new_df['weight'] == 0, 'weight'] = 1
new_df1 = new_df[new_df['name one'] == 'NaN']
new_df['weight * sentiment'] = new_df['sentiment'] * new_df['weight']
new_df.head()



### Change the weights back to zero for headlines that don't mention a country name
new_df.loc[new_df['weight'] == 1, 'weight'] = 0
new_df.head()



### Creating the Final Dataframe
sum_weights = [ [] for i in range(0,1679) ]
for i in range(0,1679):
    sum_weights[i].append(sum(new_df[(new_df['day'] == i)]['weight']))

sum_weights2 = []
for i in range(0,1679):
    sum_weights2.append(sum_weights[i][0])
sum_weights2


sum_weight_sentiment = [ [] for i in range(0,1679) ]
for i in range(0,1679):
    sum_weight_sentiment[i].append(sum(new_df[(new_df['day'] == i)]['weight * sentiment']))

sum_weight_sent2 = []
for i in range(0,1679):
    sum_weight_sent2.append(sum_weight_sentiment[i][0])
sum_weight_sent2




weighted_df = pd.DataFrame([sum_weights2,sum_weight_sent2])
weighted_df1 = weighted_df.transpose()
weighted_df1.columns = ['sum weights','sum sentiment']
weighted_df1.head()
weighted_df1['weighted avg'] = weighted_df1['sum sentiment']/weighted_df1['sum weights']
weighted_df1['weighted avg'].plot()
weighted_df2 = weighted_df1.join(reddit_news['Date'])

sp_1200['Date_sp'] = pd.to_datetime(sp_1200['Date_sp'])
sp_1200.head()


idx = pd.date_range('10-30-2009', '07-01-2016')


r = pd.date_range(start=sp_1200['Date_sp'].min(), end=sp_1200['Date_sp'].max())
r2 = pd.date_range(start=reddit_news['Date'].min(), end=reddit_news['Date'].max())
r3 = pd.date_range(start=reddit_news['Date'].min(), end=reddit_news['Date'].max())


sp_1200b = sp_1200.set_index('Date_sp').reindex(r).fillna('#NA').rename_axis('dt').reset_index()
sp_1200b[sp_1200b['Close_sp'] == '#NA'].head()


weighted_df3 = weighted_df2.set_index('Date').reindex(r2).fillna('#NA').rename_axis('dt').reset_index()


reddit_news_test = reddit_news.set_index('Date').reindex(r2).fillna('#NA').rename_axis('dt').reset_index()


reddit_news_test[reddit_news_test['dt'] == '2016-07-01']


df2 = pd.merge(sp_1200b,weighted_df3)


final_df = df2[(df2['Close_sp'] != '#NA') & (df2['weighted avg'] != '#NA')]
final_df.reset_index(inplace=True)


### Adding a binary column for sentiment pct change (1 is increase)
binary_sentiment = []
for i in final_df['weighted avg'].pct_change():
    if i > 0:
        binary_sentiment.append(1)
    else:
        binary_sentiment.append(0)
final_df['binary sentiment'] = binary_sentiment

binary_close = []
for i in final_df['Close_sp'].pct_change():
    if i > 0:
        binary_close.append(1)
    else:
        binary_close.append(0)
final_df['binary close'] = binary_close
final_df
