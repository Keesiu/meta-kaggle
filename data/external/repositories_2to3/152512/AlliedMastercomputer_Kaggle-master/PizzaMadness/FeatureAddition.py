from PizzaManager import *
from datetime import datetime

#def addEditedFeature(df):
#    df['post_was_edited'] = (df.request_text!=df.request_text_edit_aware)
#    return df

def addTextLen(df):
    df['request_len'] = df.request_text_edit_aware.apply(len)
    df['title_len'] = df.request_title.apply(len)

def addPostHour(df):
    df['post_hour_utc'] = df.unix_timestamp_of_request_utc.apply(lambda x: (datetime.utcfromtimestamp(x).hour-10)%24)
    df.drop(['unix_timestamp_of_request_utc','unix_timestamp_of_request'],axis=1,inplace=True)

# Combine all text an remove first 9 character which are typically '[Request]'
def addTotalText(df):
    df['total_text'] = (df.request_title+' '+df.request_text_edit_aware).apply(lambda x: x[9:])
