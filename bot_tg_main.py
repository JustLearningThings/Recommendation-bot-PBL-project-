from sympy import false
import telebot
from telebot import types

import numpy as np
import pandas as pd
import json
from recommender import Recommender, Vectorizer, Dataset, process_database

from Places import Place

bot = telebot.TeleBot('5140244250:AAEm7uXuVcrJtEFMG9UW7AfY7G2Cjk7L_fQ')

user_dict_recommend = {'region' : None, 'speciality': None, 'alcohol': None, 'delivery': None,
             'parking': None, 'working_hours': None, 'coordinates': None}

# constants for the recommender
PATH_TO_DB = './dataset.xlsx'
recommendations_num = 5


# helper functions

def check_query(d):
    for v in d.items(): 
        if v[1] is None:
            return False
    
    return True

def reset_dict():
    global user_dict_recommend

    user_dict_recommend = {'region' : None, 'speciality': None, 'alcohol': None, 'delivery': None,
    'parking': None, 'working_hours': None, 'coordinates': None}


# bot functions

@bot.message_handler(content_types=['text'], commands=['start', 'recommend'])
def get_text_messages(message):
    keyboard = types.InlineKeyboardMarkup()

    button1 = types.InlineKeyboardButton(text='Yes', callback_data='r')
    keyboard.add(button1)

    bot.send_message(message.from_user.id, "Would you like me to recommend a place ?", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    global user_dict_recommend
    global recommendations_num

    # input for recommendation

    if call.data == "r":
        keyboard_reg = types.InlineKeyboardMarkup(row_width=2)
        item_r = types.InlineKeyboardButton("Rascanovca", callback_data='rascanovca')
        item_c = types.InlineKeyboardButton("Centru", callback_data='centru')
        item_tc = types.InlineKeyboardButton("Telecentru", callback_data='telecentru')
        item_cioc = types.InlineKeyboardButton("Ciocana", callback_data='ciocana')
        item_bot = types.InlineKeyboardButton("Botanica", callback_data='botanica')

        keyboard_reg.add(item_r, item_c, item_tc, item_cioc, item_bot)
        bot.send_message(call.message.chat.id, "Chose preferable region", reply_markup=keyboard_reg)

    list_reg = ['rascanovca', 'centru', 'telecentru', 'ciocana', 'botanica']

    if call.data in list_reg:
        user_dict_recommend['region'] = call.data

        keyboard_rec = types.InlineKeyboardMarkup(row_width=3)
        item1 = types.InlineKeyboardButton("Restaurant", callback_data='restaurant')
        item2 = types.InlineKeyboardButton("Cafe", callback_data='cafe')
        item3 = types.InlineKeyboardButton("Fast food", callback_data='fast food')
        item4 = types.InlineKeyboardButton("Japanese", callback_data='japanese')
        item5 = types.InlineKeyboardButton("Italian", callback_data='italian')
        item6 = types.InlineKeyboardButton("Mexican", callback_data='mexican')
        item7 = types.InlineKeyboardButton("Romanian / Moldavian", callback_data='romanian / moldavian')
        item8 = types.InlineKeyboardButton("European", callback_data='european')
        item9 = types.InlineKeyboardButton("Other cuisine", callback_data='other cuisine')
        item10 = types.InlineKeyboardButton("Grill", callback_data='grill')
        item11 = types.InlineKeyboardButton("Bar", callback_data='bar')
        item12 = types.InlineKeyboardButton("Canteen", callback_data='canteen')
        item13 = types.InlineKeyboardButton("Doesn't matter", callback_data="doesn't matter")

        keyboard_rec.add(item1, item2, item3, item4, item5, item6, item7, item8, item9,
                          item10, item11, item12, item13)
        bot.send_message(call.message.chat.id, "Chose speciality for recommendation", reply_markup=keyboard_rec)

    list_speciality = ['restaurant','cafe','fast food', 'japanese', 'italian', 'mexican', 'romanian / moldavian', 'european', 'other cuisine', 'grill', 'bar', 'canteen', "doesn't matter" ]

    if call.data in list_speciality:
        user_dict_recommend ['speciality'] = call.data

        keyboard = types.InlineKeyboardMarkup(row_width=2)
        item_y = types.InlineKeyboardButton("Yes", callback_data='y_a')
        item_n = types.InlineKeyboardButton("No", callback_data='n_a')
        item_dm = types.InlineKeyboardButton("Doesn't matter", callback_data='dm_a')
        keyboard.add(item_n, item_y, item_dm)
        bot.send_message(call.message.chat.id, "Presence of Alcohol?", reply_markup=keyboard)

    list_aclo = ['y_a','n_a','dm_a']

    if call.data in list_aclo:
        user_dict_recommend ['alcohol'] = call.data

        keyboard_del = types.InlineKeyboardMarkup(row_width=2)
        item_y = types.InlineKeyboardButton("Yes", callback_data='y_d')
        item_n = types.InlineKeyboardButton("No", callback_data='n_d')
        item_dm = types.InlineKeyboardButton("Doesn't matter", callback_data='dm_d')
        keyboard_del.add(item_n, item_y, item_dm)
        bot.send_message(call.message.chat.id, "Presence of delivery?", reply_markup=keyboard_del)

    list_del = ['y_d','n_d','dm_d']

    if call.data in list_del:
        user_dict_recommend['delivery'] = call.data

        keyboard_parking = types.InlineKeyboardMarkup(row_width=2)
        item_y = types.InlineKeyboardButton("Yes", callback_data='y_p')
        item_n = types.InlineKeyboardButton("No", callback_data='n_p')
        item_dm = types.InlineKeyboardButton("Doesn't matter", callback_data='dm_p')
        keyboard_parking.add(item_n, item_y, item_dm)
        bot.send_message(call.message.chat.id, "Presence of Parking?", reply_markup=keyboard_parking)

    list_park = ['y_p', 'n_p', 'dm_p']

    if call.data in list_park:
        user_dict_recommend['parking'] = call.data

        keyboard_wh = types.InlineKeyboardMarkup(row_width=2)
        item_y = types.InlineKeyboardButton("Yes", callback_data='y_h')
        item_n = types.InlineKeyboardButton("No", callback_data='n_h')
        item_dm = types.InlineKeyboardButton("Doesn't matter", callback_data='dm_h')
        keyboard_wh.add(item_n, item_y, item_dm)
        bot.send_message(call.message.chat.id, "Show working Hours?", reply_markup=keyboard_wh)

    list_wh = ['y_h', 'n_h', 'dm_h']

    if call.data in list_wh:
        user_dict_recommend['working_hours'] = call.data

        keyboard_coord = types.InlineKeyboardMarkup(row_width=2)
        item_y = types.InlineKeyboardButton("Yes", callback_data='y_c')
        item_n = types.InlineKeyboardButton("No", callback_data='n_c')
        item_dm = types.InlineKeyboardButton("Doesn't matter", callback_data='dm_c')
        keyboard_coord.add(item_n, item_y, item_dm)
        bot.send_message(call.message.chat.id, "Show coordinates?", reply_markup=keyboard_coord)

    list_coord = ['y_c', 'n_c','dm_c']

    if call.data in list_coord:
        user_dict_recommend['coordinates'] = call.data

    for key, values in user_dict_recommend.items():
        if values and values[:3] == "dm_":
            user_dict_recommend[key] = "Doesn't matter"
        elif values and values[0] == "y":
            user_dict_recommend[key] = "Yes"
        elif values and values[0] == "n":
            user_dict_recommend[key] = "No"

    # if all the necessary information has been gathered recommend
    if check_query(user_dict_recommend):
        # create json from the given information
        data = json.dumps(user_dict_recommend)

        # recommend
        recommendations = recommender.recommend_and_parse(dataset, data, recommendations_num)

        # display recommendations
        show_working_hours = user_dict_recommend['working_hours'].lower() in ['y', 'yes']
        show_coordinates = user_dict_recommend['coordinates'].lower() in ['y', 'yes']

        for r in recommendations:
            bot.send_message(
                call.message.chat.id, 
                Recommender.show_recommendation(r, show_working_hours, show_coordinates))

        # flush the dictionary
        reset_dict()

    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text='Your choice was saved', reply_markup=None)

# read the database and vectorize it
try:
    db = Place.getAllPlaces()
    db = [place.toDict() for place in db]

    dataset = {
        'name': [],
        'rating': [],
        'wrkh': [],
        'address': [],
        'sector': [],
        'spec': [],
        'has_alc': [],
        'has_park': [],
        'has_delivery': [],
        'coord': []
    }

    for place in db:
        # print(place.toDict())
        dataset['name'].append(place['name'])
        dataset['rating'].append(place['rating'])
        dataset['wrkh'].append(place['workingHours'])
        dataset['address'].append(place['address'])
        dataset['sector'].append(place['sector'])
        dataset['spec'].append(place['specialization'])
        dataset['has_alc'].append(place['hasAlcohol'])
        dataset['has_park'].append(place['hasPark'])
        dataset['has_delivery'].append(place['hasDelivery'])
        dataset['coord'].append(place['coordinates'])
except:
    dataset = process_database(PATH_TO_DB)

# create the recommender object that fits the database
recommender = Recommender(dataset.X)

bot.polling(none_stop=True, interval=0)