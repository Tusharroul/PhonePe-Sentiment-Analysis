import pandas as pd
import numpy as np
import os

'''
-----
NOTE:
-----
open data file from data_apple_store and google_play_store_data
'''
###################  phonepe data

## from google playstore
review = [result_phonepe[i]['content'] for i in range(len(result_phonepe))]

df_phonepe = pd.DataFrame({'Source':'Google_Play_store','Reviews':review})

## from apple store
phonepe_reviews

df_phonepe = df_phonepe.append(pd.DataFrame({'Source':'Apple_app_store','Reviews':phonepe_reviews}),ignore_index=True)

## from twitter
tweets_phonepe

df_phonepe = df_phonepe.append(pd.DataFrame({'Source':'Twitter','Reviews':tweets_phonepe}),ignore_index=True)

df_phonepe.to_csv(r'C:\Users\hp\Documents\R_and_PY_programming\tushar\binf_phonepe_project\phonepe_reviews.csv')



###################  google pay data

## from google playstore
review = [result_googlepay[i]['content'] for i in range(len(result_googlepay))]

df_googlepay = pd.DataFrame({'Source':'Google_Play_store','Reviews':review})

## from apple store
googlepay_reviews

df_googlepay = df_googlepay.append(pd.DataFrame({'Source':'Apple_app_store','Reviews':googlepay_reviews}),ignore_index=True)

## from twitter
tweets_phonepe1

df_googlepay = df_googlepay.append(pd.DataFrame({'Source':'Twitter','Reviews':tweets_phonepe1}),ignore_index=True)

df_googlepay.to_csv(r'C:\Users\hp\Documents\R_and_PY_programming\tushar\binf_phonepe_project\google_pay_reviews.csv')




###################  paytm data

## from google playstore
review = [result_paytm[i]['content'] for i in range(len(result_paytm))]

df_paytm = pd.DataFrame({'Source':'Google_Play_store','Reviews':review})

## from apple store
paytm_reviews

df_paytm = df_paytm.append(pd.DataFrame({'Source':'Apple_app_store','Reviews':paytm_reviews}),ignore_index=True)

## from twitter
tweets_phonepe2

df_paytm = df_paytm.append(pd.DataFrame({'Source':'Twitter','Reviews':tweets_phonepe2}),ignore_index=True)

df_paytm.to_csv(r'C:\Users\hp\Documents\R_and_PY_programming\tushar\binf_phonepe_project\paytm_reviews.csv')