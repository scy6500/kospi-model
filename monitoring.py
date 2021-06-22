import requests
from datetime import datetime
import FinanceDataReader as fdr
import smtplib
from email.mime.text import MIMEText
import preprocessing
import grid_search
import train
import commit


def send_email():
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login('scy6500@kookmin.ac.kr', '')
    msg = MIMEText('kospi pred model이 재학습 되었습니다.')
    msg['Subject'] = 'kospi pred model 재학습 알림'
    s.sendmail("scy6500@kookmin.ac.kr", "scy6500@kookmin.ac.kr", msg.as_string())
    s.quit()


while True:
    now = datetime.now()
    now = "{}시 {}분 {}초".format(str(now.hour), str(now.minute), str(now.second))
    if now == "17시 1분 1초":
        today = datetime.today().date()
        actual_low = float(fdr.DataReader('KS11', today, today)["Low"])
        predict_low = float(requests.get('https://main-kospi-model-serving-scy6500.endpoint.ainize.ai/predict')["result"])
        f = open("monitoring/monitoring.txt", 'a')
        result = "{} 예측 : {} 실제 : {}".format(today, predict_low, actual_low)
        f.write(result)
        f.close()
        if abs(actual_low - predict_low) > 100:
            preprocessing.prepreocessing()
            grid_search.main()
            train.main()
            send_email()
            commit.main()

