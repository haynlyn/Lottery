import json
import pandas as pd
import numpy as np
import re as reg
import os

def download_data():
    jurl = "https://nylottery.ny.gov/drupal-api/api/scratch_off_games?_format=json"
    filepath = '/home/daniel/Projects/mesprojets/Lottery/Complete/data/scratch_games.json'
    os.system("wget {} -O {}".format(jurl, filepath))
    return filepath

clean_price = lambda price: np.float(reg.sub(r'\$|,','', price)) if price != np.nan else price
clean_iodds = lambda iodds: float(clean_price(reg.findall(r'[\d,\.]+', iodds)[1])) / float(clean_price(reg.findall(r'[\d\.]+', iodds)[0]))
bsearch = lambda f, l, h, g, e = 10**(-10):\
        (l+h)/2 if np.abs(f((l+h)/2) - g) <= e\
        else bsearch(f, l, (l+h)/2, g, e) if f((l+h)/2) - g > e\
        else bsearch(f, (l+h)/2, h, g, e)
k_to_thousand = lambda price: np.float(reg.findall(r'^\$(\d+)k', price.lower())[0]) * 1000

has_year = lambda time: 'year' in time or 'an.' in time or 'ann' in time or 'yr' in time
has_month = lambda time: 'month' in time or 'mo.' in time or 'mon' in time
has_week = lambda time: 'week' in time or 'wk' in time
has_day = lambda time: 'day' in time or 'daily' in time
has_life = lambda time: 'life' in time
has_time = lambda time: any([f(time) for f in [has_year, has_month, has_week, has_day, has_life]])
num_of_days = lambda time, age = 28: (100 - age) * 365.2425 if has_life(time) else 365.2425 if has_year(time) else 7 if has_week(time) else 30 if has_month(time) else 1 if has_day(time) else np.nan
num_of_years = lambda time: num_of_days(time) / 365.2425

def estimate_years_of_pay(prize, age = 28):
    return (100 - age)


def estimate_payment_frequency(prize, age = 28):
    try:
        prize = prize.lower()
        if 'life' in prize or 'annual' in prize:
            return (100 - age) * np.floor(num_of_days('year') / num_of_days(prize))
        elif 'for' in prize:
            val_arr = reg.findall(r'(a|\d+) (\w+) for (\d+\s)(\w+)', prize)[0]
            days_numerator = float(val_arr[2]) * num_of_days(val_arr[-1])
            days_denominator = num_of_days(val_arr[1]) * (1 if val_arr[0] == 'a' else float(val_arr[0]))
            return np.floor(days_numerator / days_denominator)
    except:
        #print(prize)
        #print("Error: unconventional prize amount (or it's another ticket, in which case ignore.)")
        return None

# Below was called "calculate_losing_tickets". Be sure to make any necessary changes throughout document.
def add_losing_ticket_info(game, how = 'dflt', rnd = 'yes'):
    # This calculates the number of tickets that won't generate a win. This is used to calculate the current odds, which
    # is necessary for calculating the actual odds of games with tickets that require additional DataFrames to be appended.
    # Account for both lottery games (with no diminishing ticket count and therefore having fewer columns).
    
    # Calculate total tickets from top prize.
    if how == 'top':
        T = game.iloc[0].overall_iodds * (game.iloc[0][['prizes_paid_out', 'prizes_remaining']].sum())
        t0 = T - game[['prizes_paid_out', 'prizes_remaining']].sum().sum()
    elif how in ['max', 'min', 'med', 'dflt', 'mean', 'avg']:
        preTi = pd.DataFrame(game[['prizes_paid_out', 'prizes_remaining']].sum(1)).join(game['overall_iodds'])
        preTi.rename(columns = {0: 'ti'}, inplace = True)
        func_dic = {'max': lambda pt: np.max(pt.prod(1)),\
                    'min': lambda pt: np.min(pt.prod(1)),\
                    'med': lambda pt: np.median(pt.prod(1)),\
                    'mean': lambda pt: np.mean(pt.prod(1)),\
                    'avg': lambda pt: np.mean(pt.prod(1)),\
                    'dflt': lambda pt:  (np.log(pt.overall_iodds)*pt.prod(1)).sum() / np.log(pt.overall_iodds).sum()}
        rnd_dic = {'yes': np.round, 'flr': np.floor, 'floor': np.floor, 'ceil': np.ceil, 'no': lambda x: x, 'up': np.ceil, 'down': np.floor, 'dwn': np.floor}
        t0 = rnd_dic[rnd](func_dic[how](preTi)) - preTi.ti.sum()
    else:
        print("'how' parameter not found. Try again.")
        return None
    percent_left = game.prizes_remaining.sum() / game[['prizes_remaining', 'prizes_paid_out']].sum().sum()
    r0 = percent_left * t0
    p0 = t0 - r0
    t0iodds = (1-(game.overall_iodds**-1).sum())**-1
    ticket_zero = pd.DataFrame(index = [len(game)], data = {'prize_amount': '$0', 'prizes_paid_out': p0, 'prizes_remaining': r0, 'overall_iodds': t0iodds, 'title': '0-from-calc'})
    return game.append(ticket_zero)
        
            


def account_for_ticket_prize(game):
    ticket_prize = game[game.prize_amount.apply(lambda x: ('Free' in x) and ('QP' in x or 'Quick Pick' in x))] 
    auxiliary_game_name = ticket_prize.prize_amount.iloc[0]
    if 'take 5' in auxiliary_game_name.lower():
        aux_game = pd.read_csv('/home/daniel/Projects/mesprojets/Lottery/Complete/data/auxiliary_games/take_5.csv')
    elif 'cash 4 life' in auxiliary_game_name.lower() or 'c4l' in auxiliary_game_name.lower():
        aux_game = pd.read_csv('/home/daniel/Projects/mesprojets/Lottery/Complete/data/auxiliary_games/cash_4_life.csv')
    else:
        print('Error: game name is not accounted for. Inspect.')
        return None
    # Convert iodds to odds
    aux_game['overall_odds'] = aux_game.overall_iodds**-1
    aux_game.drop('overall_iodds', inplace = True, axis = 1)
    pure_lotto_0 = pd.DataFrame(index = [len(aux_game)], data = {'prize_amount': '$0', 'title': 'pure-lotto-0', 'overall_odds' : 1 - aux_game.overall_odds.sum()})
    aux_game = aux_game.append(pure_lotto_0)
    # Account for games that return another ticket for the same game.
    if (aux_game.prize_amount == "$-1").any():
        same_game = aux_game[aux_game.prize_amount == "$-1"]
        aux_game.overall_iodds *= (1 - same_game.overall_odds.values)**-1
        aux_game.drop(same_game.index, inplace = True)

    aux_game.overall_odds *= ticket_prize.overall_odds.values
    game.drop(ticket_prize.index, axis = 0)
    return game.append(aux_game, ignore_index = True)
    















has_num_match = lambda prize: reg.match(r'\$\d+,\d+', prize)
has_k_match = lambda prize: reg.match(r'\$\d+k', prize)

'''
def clean_prize(prize):
    try:
        return clean_price(prize)
    except:
        #print("Can't clean normally.")
        try:
            #print(prize)
            prize = prize.lower()
            #print('Prize lowered')
            if has_time(prize):
                freq = estimate_payment_frequency(prize, age = 28)
            else:
                freq = 1
            #print('calculated freq')
            #print(freq)
            match_num, match_k = has_num_match(prize), has_k_match(prize)
            #print('made matches')
            if match_num:
                val = clean_price(match_num.group())
                #print(match_num)
            if match_k:
                val = k_to_thousand(match_k.group())
                #print(match_k)
            if ('tax' in prize) and ('free' in prize or 'paid' in prize or 'no' in prize):
                ##part = float(reg.findall(r'^[\dk]+', reg.sub(r'\$|,|\s', '', prize))[0])
                #print('tax. goal = {}'.format(val))
                val = bsearch(take_for_all_taxes, val, 2 * val, val, .01)
                #print('tax. end = {}'.format(val))
            cleaned_num = val * freq
            #print(cleaned_num)
            return cleaned_num
        except:
            print('Error: {}'.format(prize))
            return None
'''

take_for_all_taxes = lambda prize: prize * (1 - ((.24 + .0882) if prize > 5000 else 0))

def clean_prize(prize, payment = 'recurring'):
    try:
        print('cleaning plain')
        return clean_price(prize)
    except:
        try:
            # convert prize to lowercase
            print('converting')
            prize = prize.lower()
            # see if prize is written with k for thousands or no
            print('making_matches')
            match_num, match_k = has_num_match(prize), has_k_match(prize)
            # get value part
            if match_num:
                val = clean_price(match_num.group())
            if match_k:
                val = k_to_thousand(match_k.group())
            if ('tax' in prize) and ('free' in prize or 'paid' in prize or 'no' in prize):
                #val = bsearch(take_for_all_taxes, val, 1.5*val, val, .01)
                val /= 1 - (.24 + .0882)
            print('made val: {}'.format(val))

            # determine frequncy per year and number of years of payment
            ## first 2 conditions are for special cases
            if 'annual' in prize:
                per_year, num_years = 1, num_of_years('life')
            elif 'for' in prize:
                val_arr = reg.findall(r'(a|\d+) (\w+) for (\d+\s)?(\w+)', prize)[0]
                num_years = (1 if val_arr[2] == '' else float(val_arr[2])) * num_of_years(val_arr[-1])
                per_year = (1 if val_arr[0] == 'a' else num_of_years(val_arr[0])) / num_of_years(val_arr[1])
            # this assumes something like "$1k/wk/life"
            elif has_time(prize):
                if has_life(prize):
                    num_years = num_of_years('life')
                if has_week(prize):
                    per_year = num_of_years('week')**-1 
                elif has_day(prize):
                    per_year = 365.2425
            # if it's just a dollar amount then it should've been treated in the try, but just in case
            else:
                per_year, num_years = 1, 1
            print('have duration {} and frequency {}'.format(num_years, per_year))
            # now we take out for tax; payment method determines amount taken
            if payment == 'lump':
                print('lump')
                dough = val * num_years * per_year * (.6 if has_time(prize) else 1)
                dough = take_for_all_taxes(dough)
            else:
                print('recurring')
                dough_per_year = val * per_year
                dough_per_year = take_for_all_taxes(dough_per_year)
                dough = dough_per_year * num_years
            return dough
        except:
            print('Error with prize: {}'.format(prize))
            return None
               

'''
def clean_game_frame(df):
    # Convert iodds from str to float.
    df.iodds = df.overall_odds.apply(clean_iodds)
'''

def load_games(f):
    with open(f, 'rb') as jf:
        j = json.load(jf)

    num_name_dic, num_odds_dic, games = {}, {}, {}
    
    for g in j:
        print(g['game_number'])
        num_name_dic.setdefault(g['game_number'], g['title'].strip())

        try:
            num_odds_dic.setdefault(g['game_number'], clean_iodds(reg.findall(r'\d+ \D+ \d+.\d+', g['overall_odds'])[0]))
        except:
            pass

        game_df = pd.DataFrame(g['odds_prizes'],\
                        columns = ['prize_amount', 'prizes_paid_out',\
                        'prizes_remaining', 'overall_odds', 'title'], dtype = float)
        
        #clean game_df before adding to games
        #df = clean_game_frame(game_df)

        game_df.overall_odds = game_df.overall_odds.apply(clean_iodds)**-1
        #game_df.rename(columns = lambda x: x.replace('odds', 'iodds') if 'odds' in x else x, inplace = True)
        #game_df.overall_iodds = game_df.overall_iodds.apply(clean_iodds)
        games.setdefault(g['game_number'], game_df)

    return (num_name_dic, num_odds_dic, games)

'''
def make_game_dic(games):
    return {num: name for num, name in game for game in games}
'''
'''
def take_for_taxes(amt, tax):
    try:
        if amt < 0:
            return 0
        else:
            foo = tax[(tax.low <= amt) & (tax.high > amt)]
            return (foo.rate * (amt - foo.low)).values[0] + foo.rest.values[0]
    except:
        print("Error: amt is not a number >= 0. Fix.")
        return None
    
take_for_all_taxes = lambda amt, taxes: amt - np.sum([take_for_taxes(amt, t) for t in taxes])

# Read all things to account for in losses and then subtract that from the gross prize.
# TODO: make it so that it's loaded once and applied to all games in question.
def after_losses(game, amt = 'prize_amount'):
    folder = '/home/daniel/Projects/mesprojets/Lottery/Complete/data/losses'
    taxes = [pd.read_csv("{}/{}".format(folder, x)) for x in os.listdir(folder)]
    print('loaded taxes')
    toreturn = game[amt].apply(lambda val, tax_files = taxes: take_for_all_taxes(val, tax_files))
    return toreturn
'''
