import os
import datetime
from download_data import download

'''
Should I rewrite so that it takes a date param and can first look up for that?
    If the date doesn't exist, then it can get the latest date.
If the param doesn't exist, then it can get the list of all data.
'''
def get_data():
    folderpath = "./data/jsons"
    today = datetime.datetime.today()
    today_dateISO = today.date().isoformat().replace('-', '')
    today_fn = "scratch_games_{}.json".format(today_dateISO)
    if today_fn not in os.listdir(folderpath):
        download()

    return os.listdir(folderpath)


