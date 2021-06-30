from tweepy import OAuthHandler
import pandas as pd
import tweepy
import pre
import warnings
warnings.filterwarnings('ignore')


access_token = "1372225633154072576-30fWXQZ9DetK9jU6qJ31IKA9YFEIFr"
access_secret = "JC7aVA4Ae7Sc2E8yhllSsObKV4Pjkin5eufOnuGfeUgGp"
consumer_key = "GvUc8F9Mu4gw1XGIYLMzrPMkx"
consumer_secret = "IkYqjhQJoBWADKT3JP41lKymO4WtKrDR0tv8JEyHtjIH1D6tL6"

def printtweetdata(n, ith_tweet):
    print()
    print(f"Tweet {n}:")
    print(f"Username:{ith_tweet[0]}")
    print(f"Location:{ith_tweet[1]}")
    print(f"Follower Count:{ith_tweet[2]}")
    print(f"Total Tweets:{ith_tweet[3]}")
    print(f"Retweet Count:{ith_tweet[4]}")
    print(f"Tweet Text:{ith_tweet[5]}")
    print(f"Hashtags Used:{ith_tweet[6]}")

def scrape(words, numtweet):
    db = pd.DataFrame(columns=['user', 'text'])

    tweets = tweepy.Cursor(api.search, q=words, lang="en", tweet_mode='extended').items(numtweet)

    list_tweets = [tweet for tweet in tweets]
    i = 1
    for tweet in list_tweets:
        username = tweet.user.screen_name
        location = tweet.user.location
        followers = tweet.user.followers_count
        totaltweets = tweet.user.statuses_count
        retweetcount = tweet.retweet_count
        hashtags = tweet.entities['hashtags']
        

        try:
            text = tweet.retweeted_status.full_text
        except AttributeError:
            text = tweet.full_text
        hashtext = list()
        for j in range(0, len(hashtags)):
            hashtext.append(hashtags[j]['text'])

        if followers>10000:
            ith_tweet = [username, text]
            db.loc[len(db)] = ith_tweet

            i = i+1

    #print(db)
    p=pre.prediction(db)
    print(p)
    filename = 'scraped_tweets.csv'
    p.drop(p.index[p['val'] == 0.0], inplace=True)
    p.drop(['val'], axis=1, inplace=True)
    p.to_csv(filename)


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

print("Enter Twitter HashTag to search for")
words = input()

numtweet = 1000
scrape(words, numtweet)
print('\n\nScraping has completed!')



