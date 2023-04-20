import json
import os # wget needs to be installed
import datetime

def download():
    folderpath = './data/jsons'
    jurl = "https://nylottery.ny.gov/drupal-api/api/scratch_off_games?_format=json"
    today = datetime.datetime.today()
    
    day = today - datetime.timedelta(0 if today.time() > datetime.time(hour=6, minute=30) else 1)
    day = day.date()
    day_dateISO = day.isoformat().replace('-', '')
    day_fn = 'scratch_games_{}.json'.format(day_dateISO)
    filepath = '/'.join([folderpath, day_fn])
    # Make sure that wget is installed
    if day_fn not in os.listdir(folderpath):
        os.system("wget {} -O {}".format(jurl, filepath)

if __name__ == "__main__":
    download()
