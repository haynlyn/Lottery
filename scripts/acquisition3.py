
import json
import pandas as pd
import numpy as np
import re as reg
import os
import datetime


# Function for downloading the data.
def get_data():
    folderpath = '/home/daniel/Projects/mesprojets/Lottery/Complete/data/scratch_games'
    # Data is updated every day at 6:30AM. Make sure we have most recent file before making call to download.
    today = datetime.datetime.today()
    if today.time() <= datetime.time(hour = 6, minute = 30):
        print("Data not yet updated for today. Checking for yesterday's.")
        yesterday = today.date() - datetime.timedelta(days = 1)
        yesterday_date = yesterday.date().isoformat().replace('-', '')
        filepath = '/'.join([folderpath, 'scratch_games_{}.json'.format(yesterday_date)])
        if 'scratch_games_{}.json'.format(yesterday) not in os.listdir(folderpath):
            jurl = "https://nylottery.ny.gov/drupal-api/api/scratch_off_games?_format=json"
            print("Downloading yesterday's data from source.")
            os.system("wget {} -O {}".format(jurl, filepath))
        else:
            print("Yesterday's data found. Using.")
    else:
        print("Checking if today's data is already downloaded.")
        today_date = today.date().isoformat().replace('-', '')
        filepath = '/'.join([folderpath, 'scratch_games_{}.json'.format(today_date)])
        if 'scratch_games_{}.json'.format(today.date()).replace('-', '') not in os.listdir(folderpath):
            jurl = "https://nylottery.ny.gov/drupal-api/api/scratch_off_games?_format=json"
            print("Downloading today's data from source.")
            os.system("wget {} -O {}".format(jurl, filepath))
        else:
            print("Today's data found. Using.")
    return filepath






# A bunch of short, necessary functions.
## Transform strings of dollar amounts to floats.
clean_price = lambda price: np.float(reg.sub(r'\$|,','', price)) if price != np.nan else price

## Transform strings of "1 in XXX" to the value XXX
clean_iodds = lambda iodds: float(clean_price(reg.findall(r'[\d,\.]+', iodds)[1])) / float(clean_price(reg.findall(r'[\d\.]+', iodds)[0]))

## Binary search function, can be passed a complicated tax function as f in order to find a proper value. Not necessary for current data, but might be useful if scaled later.
'''bsearch = lambda f, l, h, g, e = 10**(-10):\
        (l+h)/2 if np.abs(f((l+h)/2) - g) <= e\
        else bsearch(f, l, (l+h)/2, g, e) if f((l+h)/2) - g > e\
        else bsearch(f, (l+h)/2, h, g, e)'''

## Function to transform prizes using "K" as "000" into numerical value
k_to_thousand = lambda price: np.float(reg.findall(r'^\$(\d+)k', price.lower())[0]) * 1000

## A bunch of quick, time-based functions for treating prizes with any sort of frequency.
## Used for determining frequency of prize for determining total pay and tax treatment.
has_year = lambda time: 'year' in time or 'an' in time or 'ann' in time or 'yr' in time
has_month = lambda time: 'month' in time or 'mo' in time or 'mon' in time
has_week = lambda time: 'week' in time or 'wk' in time
has_day = lambda time: 'day' in time or 'daily' in time
has_life = lambda time: 'life' in time
has_time = lambda time: any([f(time) for f in [has_year, has_month, has_week, has_day, has_life]])
num_of_days = lambda time, age = 28: (100 - age) * 365.2425 if has_life(time) else 365.2425 if has_year(time) else 7 if has_week(time) else 365.2425/12 if has_month(time) else 1 if has_day(time) else np.nan
num_of_years = lambda time: num_of_days(time) / 365.2425

## Functions to match whether a prizes is written as "K" or "000" for multiples of 1000.
has_num_match = lambda prize: reg.match(r'\$\d+,\d+', prize)
has_k_match = lambda prize: reg.match(r'\$\d+k', prize)






# Calculate the number of tickets in the scratch game that return $0 as a prize. This is necessary for estimating updated odds.
def add_losing_ticket_info(game, how = 'dflt', rnd = 'yes'):
    # Calculate total tickets from top prize.
    if how == 'top':
        T= game.iloc[0].overall_odds * (game.iloc[0][['prizes_paid_out', 'prizes_remaining']].sum())
        t0 = T - game[['prizes_paid_out', 'prizes_remaining']].sum().sum()
    # Calculate from set of estimated values.
    elif how in ['max', 'min', 'med', 'dflt', 'mean', 'avg']:
        preTi = pd.DataFrame(game[['prizes_paid_out', 'prizes_remaining']].sum(1)).join(game['overall_odds']**-1)
        preTi.rename(columns = {0: 'ti'}, inplace = True)
        # max: Take max
        # min: Take min
        # med: Take median
        # mean/avg: Take arithmetic average
        # dflt: Take weighted average with log of odds^-1 as weights. (Here, dflt(log(X)) == dflt(log(X^-1)), so we use log(X))
        func_dic = {'max': lambda pt: np.max(pt.prod(1)),\
                    'min': lambda pt: np.min(pt.prod(1)),\
                    'med': lambda pt: np.median(pt.prod(1)),\
                    'mean': lambda pt: np.mean(pt.prod(1)),\
                    'avg': lambda pt: np.mean(pt.prod(1)),\
                    'dflt': lambda pt:  (np.log(pt.overall_odds)*pt.prod(1)).sum() / np.log(pt.overall_odds).sum()}
        # Options for rounding: normal, down, down, up, none, up, down, down.
        rnd_dic = {'yes': np.round, 'flr': np.floor, 'floor': np.floor, 'ceil': np.ceil, 'no': lambda x: x, 'up': np.ceil, 'down': np.floor, 'dwn': np.floor}
        # Calculate estimated count of total 0_tickets by subtracting from our total value the sum of all other prizes.
        t0 = rnd_dic[rnd](func_dic[how](preTi)) - preTi.ti.sum()
    else:
        print("'how' parameter not found. Try again.")
        return None
    # In order to estimate the number of tickets remaining for prize_0, we calculate the proportion of all other tickets still in play and assume prize_0 were distributed similarly.
    percent_left = game.prizes_remaining.sum() / game[['prizes_remaining', 'prizes_paid_out']].sum().sum()
    # Use the ratio of tickets left from total tickets to estimate all remaining tickets for prize_0.
    r0 = percent_left * t0
    # Tickets paid out is simply the difference between the two.
    p0 = t0 - r0
    # prize_0's odds are 1 - the sum of all other odds.
    t0odds = 1 - game.overall_odds.sum()
    # Create singleton DataFrame for ticket_0 to be appended to the game, and return.
    ticket_zero = pd.DataFrame(index = [len(game)], data = {'prize_amount': '$0', 'prizes_paid_out': p0, 'prizes_remaining': r0, 'overall_odds': t0odds, 'title': '{}-0'.format('-'.join(game.title.iloc[0].split('-')[:2]))})
    return game.append(ticket_zero)
        
 




# Some games have the probability of returning another ticket as a prize. Use this to compile a fuller sense of the odds.
def account_for_ticket_prize(game):
    print(game)
    print('accounting for ticket prize')
    # Locate the row for the prize that returns a game.
    ticket_prize = game[game.prize_amount.str.contains(r'free.+qp|free.+quick|free.+fp', regex = True)]
    #ticket_prize = game[game.prize_amount.apply(lambda x: ('free' in x) and ('QP' in x or 'Quick Pick' in x))] 
    # Gather the name of the prize-game.
    auxiliary_game_name = ticket_prize.prize_amount.iloc[0]
    # Look up the corresponding file for the prize-game and use it to update odds.
    if 'take5' in auxiliary_game_name.lower().replace(' ', '') or 't5' in auxiliary_game_name.lower().replace(' ', ''):
        aux_game = pd.read_csv('/home/daniel/Projects/mesprojets/Lottery/Complete/data/auxiliary_games/take_5.csv')
    elif 'c4l' in auxiliary_game_name.lower().replace(' ', '') or 'cash4life' in auxiliary_game_name.lower().replace(' ', ''):
        aux_game = pd.read_csv('/home/daniel/Projects/mesprojets/Lottery/Complete/data/auxiliary_games/cash_4_life.csv')
    else:
        print('Error: game name is not accounted for. Inspect.')
        print("Couldn't load game for {}".format(game))
        return None
    print('loaded game')
    # Games are stored as having inverse-odds (iodds) rather than odds. We just invert to get odds and then drop iodds.
    # Convert iodds to odds
    aux_game['odds'] = aux_game.overall_iodds**-1
    aux_game.drop('overall_iodds', inplace = True, axis = 1)
    print('iodds converted to odds and dropped')
    # "pure lotto" refers to the games returned, as their odds don't change per se. Calculate the odds of losing in this game.
    ## prize_amount and title are self-explanatory. Odds of losing are just 1 - sum(odds of winning something).
    pure_lotto_0 = pd.DataFrame(index = [len(aux_game)], data = {'prize_amount': '$0', 'title': '{}-0'.format('-'.join(game.title.iloc[0].split('-')[:2] + aux_game.title.iloc[0].split('-')[:1])), 'odds' : 1 - aux_game.odds.sum()})
    print('calculated pl0')
    # Append to the loaded game the information for its losing odds.
    aux_game = aux_game.append(pure_lotto_0)
    # Account for games that return another ticket for the same game.
    if (aux_game.prize_amount == "$-1").any():
        print('ticket returns itself')
        # Access row for tickets that return another ticket of the game as a prize (i.e., Take 5)
        same_game = aux_game[aux_game.prize_amount == "$-1"]
        # If odds of getting the same game are Y, then the odds of any prize are divided by Y in order to account for the new universe.
        aux_game.odds /= (1 - same_game.odds.values)
        # Odds have been updated, so remove the ticket returning its parent-game so that the new probability universe is sound.
        aux_game.drop(same_game.index, inplace = True)
    # Odds for loaded game are then multiplied by the odds of the ticket_prize in the main game when looked at from the outer universe.
    aux_game['oodds'] = aux_game.odds * ticket_prize['oodds'].values
    aux_game['nodds'] = aux_game.odds * ticket_prize['nodds'].values
    print('calculated and updated odds')
    # Similarly, drop the ticket_prize from the main game since we've replaced it by its prizes and their odds.
    game_ret = game.drop(ticket_prize.index, axis = 0)
    print('dropped ticket_prize from main game.')
    # Append the updated loaded game to the main game to have an complete picture of its prizes and odds.
    game_ret = game_ret.append(aux_game, ignore_index = True)#[['prize_amount', 'title', 'nodds', 'oodds']]
    print('appended aux_game to main game')
    return game_ret
    





# Calculate deduction due to taxes. For NY, it is very simple. The below assumes not a resident of NYC or Yonkers.
take_for_all_taxes = lambda prize: prize * (1 - ((.24 + .0882) if prize > 5000 else 0))

# The main function for transforming prizes from their string representation to actual numbers. Involves calculating frequency of recurring payments and considers whether prize is lump-sum or not.
## Default is recurring since that's as the prizes are written.
def clean_prize(prize, payment = 'recurring'):
    try:
        # Simple transformation of "$12,345" to 12345.00
        return clean_price(prize)
    except:
        try:
            # Convert prize to lowercase for easier treatment.
            prize = prize.lower()
            # See if prize is written with k for thousands or not.
            match_num, match_k = has_num_match(prize), has_k_match(prize)
            # Get value part.
            if match_num:
                val = clean_price(match_num.group())
            if match_k:
                val = k_to_thousand(match_k.group())
            if ('tax' in prize) and ('free' in prize or 'paid' in prize or 'no' in prize):
                # Below is commented out for reasons explained at beginning of document.
                #val = bsearch(take_for_all_taxes, val, 1.5*val, val, .01)
                val /= 1 - (.24 + .0882)

            # Determine frequency per year and number of years of payment.
            ## First 2 conditions are for special cases.
            ## "Annual" prizes are only until end of life.
            if 'annual' in prize:
                per_year, num_years = 1, num_of_years('life')
            ## These prizes have a specific structure to them from which the relevant duration and frequency can be ascertained.
            elif 'for' in prize:
                val_arr = reg.findall(r'(a|\d+) (\w+) for (\d+\s)?(\w+)', prize)[0]
                num_years = (1 if val_arr[2] == '' else float(val_arr[2])) * num_of_years(val_arr[-1])
                per_year = (1 if val_arr[0] == 'a' else num_of_years(val_arr[0])) / num_of_years(val_arr[1])
            # This assumes something like "$1k/wk/life", from which the relevant duration and frequency can be ascertained.
            elif has_time(prize):
                # In experience, if it has life, then it will be for life, which serves as duration.
                if has_life(prize):
                    num_years = num_of_years('life')
                # We use this to calculate the frequency. We use each type of value simply because num_of_years(prize), if prize contains "life", would return estimation for years left in life rather than in smaller increment.
                if has_year(prize):
                    per_year = 1
                elif has_month(prize):
                    per_year = num_of_years('month')**-1
                elif has_week(prize):
                    per_year = num_of_years('week')**-1 
                elif has_day(prize):
                    per_year = 365.2425
            # If it's just a dollar amount then it should've been treated in the try, but just in case it wasn't, treat here.
            else:
                per_year, num_years = 1, 1
            # Now we take out for tax; payment method determines amount taken.
            if payment == 'lump':
                # Assume that we lose 40% on recurring payments for whatever reason. If it's not recurring, then it's lump-sum is default and multiplier should be 1.
                dough = val * num_years * per_year * (.6 if has_time(prize) else 1)
                # Take taxes on lump-sum.
                dough = take_for_all_taxes(dough)
            else:
                # First calculate amount earned per year.
                dough_per_year = val * per_year
                # Then take taxes on this yearly source of income.
                dough_per_year = take_for_all_taxes(dough_per_year)
                # Then multiply it by the number of years which this will be paid out.
                dough = dough_per_year * num_years
            return dough
        except:
            print('Error with prize: {}'.format(prize))
            return None
               




# This runs all of the above functions to make complete a game with all numeric prizes and their odds.
def clean_game(game_df, lump_or_rec = 'lump'):
    # Convert iodds "1 in X" to odds 1/X.
    game_df.overall_odds = game_df.overall_odds.apply(clean_iodds)**-1
    # Convert prize_amounts to lowercase for easier treatment of strings.
    game_df.prize_amount = game_df.prize_amount.str.lower()
    # Calculate 0 tickets.
    game_df = add_losing_ticket_info(game_df)
    # nodds = new odds
    game_df['nodds'] = game_df.prizes_remaining / game_df.prizes_remaining.sum()
    # oodds = old odds
    game_df['oodds'] = game_df.overall_odds
    # Drop overall_odds columns since it's extra baggage.
    game_df.drop('overall_odds', axis = 1, inplace = True)
    # Account for games with pure lotteries.
    if game_df.prize_amount.str.contains(r'free.+qp|free.+quick|free.+fp', regex = True).any():
        game_df = account_for_ticket_prize(game_df)
    # Drop all columns except prize_amount, nodds, oodds, paid_out, remaining, and title.
    game_df = game_df[['title', 'prize_amount', 'prizes_paid_out', 'prizes_remaining', 'nodds', 'oodds']].copy()
    # Clean prize values.
    game_df.prize_amount = game_df.prize_amount.apply(lambda x: clean_prize(x, lump_or_rec))
    return game_df

   




# Load games from filepath to JSON file of games.
def load_games(f):
    # Load JSON and read into variable.
    with open(f, 'rb') as jf:
        j = json.load(jf)

    # Initialize three dictionaries: one for the name, another for price, and the last for the actual prizes and odds.
    num_name_dic, num_price_dic, games = {}, {}, {}
    
    for g in j:
        game_num = int(g['game_number'])
        try:
            num_name_dic.setdefault(game_num, g['title'].strip())
        except:
            pass

        try:
            num_price_dic.setdefault(game_num, float(g['ticket_price']))
        except:
            pass

        try:
            # Load data into df.
            game_df = pd.DataFrame(g['odds_prizes'],\
                        columns = ['prize_amount', 'prizes_paid_out',\
                        'prizes_remaining', 'overall_odds', 'title'], dtype = float)

            # Clean game via above function.
            game_df = clean_game(game_df)

            # Add to dic
            games.setdefault(game_num, game_df)
        except:
            pass

    return (num_name_dic, num_price_dic, games)





# Define metrics for performance.
def measure(game_df, price, how = 'all'):
    odds = {x for x in game_df if x.endswith('odds')}
    if not ({'prize_amount'} | odds).issubset(game_df):
        print("Error: passed dataframe lacks necessary columns.")
        return -1
    if 10**16 - np.prod(game_df[odds].sum()) * 10**16 > 0:
        print("Error: passed dataframe's odds don't sum to 1.\n{}".format(game_df))
        print("Discrepancy:\t{}".format(10**16 * (1- np.prod(game_df[odds].sum()))))
        #return -2

    # Define functions to be run on df.
    ### erp/erps/erp$/erpd = (prize_amount * odds).sum() / Expected return per dollar spent.
    ### albe = game_df[game_df.prize_amount >= price][odds].sum() / Odds of breaking even or more.
    ### prft = game_df[game_df.prize_amount > price][odds].sum() / Odds of making profit.
    ##### sodds : single odds

    func_dic = {'albe': lambda df, sodds: df[df.prize_amount >= price][sodds].sum(),
            'prft': lambda df, sodds: df[df.prize_amount > price][sodds].sum(),
            'erp': lambda df, sodds: df[['prize_amount', sodds]].prod(1).sum()}

    if how not in (['all'] + list(func_dic)):
        print('Error: passed improper value for `how` parameter.')
        return -3
    elif how in func_dic:
        val = {'_'.join([how, sodds]): func_dic[how](game_df, sodds) for sodds in odds}
    else:
        val = {'_'.join([fun, sodds]): func_dic[fun](game_df, sodds) / price for fun in func_dic for sodds in odds}
    return val





# Aggregate all games of a given price into a single "game". Recalculate odds.
def analyze_price_performance(games, num_price_dic):
    # Calculate dictionary of prices mapping to an aggregate of games in said price.
    price_games_dic = {price: pd.concat([[games[num]] for num in games if num_price_dic[num] == price]) for price in num_price_dic.values()}
    
    # Update the oodds and nodds for each price.
    for price in price_games_dic:
        # Temporary assignment.
        df = price_games_dic[price]
        # Group by prize_amount and reset index to reduce size.
        df = df.groupby(by = 'prize_amount').sum().reset_index()
        # Oodds are just ti / ti.sum() for each i.
        df['oodds'] = df[['prizes_paid_out', 'prizes_remaining']].sum(1) / df[['prizes_paid_out', 'prizes_remaining']].sum().sum()
        # Nodds are just ri / ri.sum() for each i.
        df['nodds'] = df['prizes_remaining'] / df['prizes_remaining'].sum()
        price_games_dic[price] = df

    # Measure performance of each price in new dictionary.
    price_performances = {price: measure(price_games_dic[price], price) for price in price_games_dic}
    return price_performances



    

def main():
    # Get filepath for JSON data.
    print('Getting data.')
    json_filepath = get_data()

    # Get necessary info on games and put into three dics: one for mapping id to name, another from id to price, and last from id to DataFrame.
    print('Making game-relevant dictionaries.')
    num_name_dic, num_price_dic, games = load_games(json_filepath)




