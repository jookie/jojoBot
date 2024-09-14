# train.py

import datetime
import pytz
def main():
    # Get current time in UTC 
    datetime_utc = datetime.datetime.now(pytz.utc)
    return datetime_utc    

if __name__ == "__main__":
    main()