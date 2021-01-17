# Import necessary modules
## re for mining for game numbers in the pdf
## tika for reading the pdf into Python as a readable format (as a dict)
## os for locating the pdf in the filesystem (might be unnecessary and maybe running wget in shell would suffice)
## datetime for using searching both for the most recent game list and also to filter within said list for
  ## available games
import re as reg
from os import listdir, system, chdir
from datetime import date, timedelta

## get pdf with list of games
#### NY website updates pdf every Monday and titles the pdf with that Monday's date, so we round to nearest Monday
def get_pdf(hui):
    # if date passed isn't Monday, then update it to most recent Monday
    if hui.weekday() > 0:
        hui -= timedelta(hui.weekday())
    # convert date from date type to string with the appropriate format for NY
    huistr = '{}.{}.{}'.format(hui.month, hui.day, hui.year)
    # generate url for getting pdf
    pre_filename = "Scratch-Off Game Reports as of {}.pdf".format(huistr)
    pdf_url = r"https://edit.nylottery.ny.gov/sites/default/files/{}".format(pre_filename)
    # call wget from shell to download pdf into cwd, but replace spaces with '%20' in url name to not mess with shell
    system("wget {} -O '{}'".format(pdf_url.replace(' ', '%20'), '../data/{}'.format(pre_filename)))
    # return filename for being passed to another function
    return "../data/{}".format(pre_filename)



## parse pdf for list of game IDs
def parse_pdf_for_ids(pdf_fn):
    from tika import parser
    do = parser.from_file(pdf_fn)['content']
    # regex to find all rows: re = reg.findall(r"(.+)\s([\d,]+)\s([\d/]+|Open)", do)
    ## main issue with this is that there are some rows with "\d{4}, \d{4}", the former of which won't be
    ## included with the other in the same tuple. Can fix that, but it might not be necessary.
    re = reg.findall(r"(.+)\s([\d,]+)\s(\d+/\d+/\d+|Open)", do)
    date_to_iso = lambda d: '20{}-{}-{}'.format(d.split('/')[-1], d.split('/')[0], d.split('/')[1])
    is_good = lambda x: date.fromisoformat(date_to_iso(x[-1])) > date.today() if x[-1] != 'Open' else True
    mi = [x for x in re if is_good(x)]
    game_ids = set(reg.findall(r"\d{4}", '\n'.join(' '.join(x) for x in mi)))
    # return set of game ids to be iterated through by another function
    return game_ids

## download webpages for list of game ids
#### do this in another program, at least for now
def download_game_data(game_id_set):
    pre_game_url = 'https://nylottery.ny.gov/scratch-off-game/?game='
    game_url = lambda gid: pre_game_url + gid
    




## main
def __main__():
    chdir('/home/daniel/Projects/mesprojets/Lottery/Complete/data')
    pdf_filename = get_pdf(date.today())
    game_ids = parse_pdf_for_ids(pdf_filename)

