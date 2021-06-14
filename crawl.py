import FinanceDataReader as fdr
from datetime import datetime

today = datetime.today().date()
ks11 = fdr.DataReader('KS11', today, today)
ks11['Date'] = ks11.index
ks11 = ks11[["Date", "Close", "Open", "High", "Low", "Volume", "Change"]]

ks11.to_csv("data/raw.csv", mode='a', index=False, header=False)

