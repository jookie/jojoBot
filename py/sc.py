import schedule
import time
import os
from datetime import datetime

def run_script():
    current_time = datetime.now()
    if current_time.hour >= 9 and current_time.hour <= 16 and current_time.weekday() < 5:
        os.system("train.py")

schedule.every(1).minutes.do(run_script)

while True:
    schedule.run_pending()
    time.sleep(1)
