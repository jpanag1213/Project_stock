elif signal == 'obi_demo':
# todo: revise the obi signal here
print(paraset[0])
window = int(paraset[0])
self.allQuoteData[symbol].loc[:, 'obi'] = np.log(self.allQuoteData[symbol].loc[:, 'bidVolume1']) - np.log(
    self.allQuoteData[symbol].loc[:, 'askVolume1'])

self.allQuoteData[symbol].loc[:, 'obi1'] = np.log(self.allQuoteData[symbol].loc[:, 'bidVolume1'] +
                                                  self.allQuoteData[symbol].loc[:, 'bidVolume2']) - np.log(
    self.allQuoteData[symbol].loc[:, 'askVolume1'])

self.allQuoteData[symbol].loc[:, 'obi2'] = np.log(self.allQuoteData[symbol].loc[:, 'bidVolume1']) - np.log(
    self.allQuoteData[symbol].loc[:, 'askVolume1'] +
    self.allQuoteData[symbol].loc[:, 'askVolume2'])
# self.allQuoteData[symbol].loc[:, 'obi_' + str(window) + '_min'] = self.allQuoteData[symbol].loc[:,
#                                                                   'obi'].rolling(window * 60).mean()
self.allQuoteData[symbol].loc[:, 'obi_' + str(window) + '_min'] = self.allQuoteData[symbol].loc[:, 'obi'].diff(window)

askPriceDiff = self.allQuoteData[symbol]['askPrice1'].diff()
bidPriceDiff = self.allQuoteData[symbol]['bidPrice1'].diff()
midPriceChange = self.allQuoteData[symbol]['midp'].diff()

self.allQuoteData[symbol].loc[:, 'priceChange'] = 1
self.allQuoteData[symbol].loc[midPriceChange == 0, 'priceChange'] = 0

obi_change_list = list()
last_obi = self.allQuoteData[symbol]['obi'].iloc[0]
tick_count = 0
row_count = 0
for row in zip(self.allQuoteData[symbol]['priceChange'], self.allQuoteData[symbol]['obi']):
    priceStatus = row[0]
    obi = row[1]
    if (priceStatus == 1) or np.isnan(priceStatus):
        tick_count = 0
        last_obi = obi
    else:
        last_obi = self.allQuoteData[symbol]['obi'].iloc[row_count - tick_count]
        if tick_count <= window:
            tick_count = tick_count + 1

    row_count = row_count + 1
    obi_change = obi - last_obi
    obi_change_list.append(obi_change)

self.allQuoteData[symbol].loc[:, 'obi'] = obi_change_list
positivePos = (self.allQuoteData[symbol]['obi'] > float(paraset[2])) & (self.allQuoteData[symbol]['obi2'] > 1)
negativePos = (self.allQuoteData[symbol]['obi'] < -float(paraset[2])) & (self.allQuoteData[symbol]['obi1'] < -1)
self.allQuoteData[symbol].loc[positivePos, signal + '_' + str(window) + '_min'] = 1
self.allQuoteData[symbol].loc[negativePos, signal + '_' + str(window) + '_min'] = -1
self.allQuoteData[symbol].loc[(~positivePos) & (~negativePos), signal + '_' + str(window) + '_min'] = 0
# self.allQuoteData[symbol].loc[:,''] =
# self.allQuoteData[symbol].loc[:, 'obi' + str(window) + '_min_sum'] = self.allQuoteData[symbol].loc[:,'obi'].rolling(window * 60).sum()
# todo: 把几层obi当作一层看待，适合高价股？
print('Calculate obi here for symbol = ', symbol, 'with lbwindow = ', window)