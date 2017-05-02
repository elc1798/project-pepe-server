import json, urllib

token = "EAACEdEose0cBAP9GZBlEU9H40msYDAmykgZBKw4EAJzHKH3TE9uzUgZCmYZB6N3SQZBfY3473ZCI25CZBUCu9WV2NxC9JdU4uZBvNLXwvAablDHfOlXgeR9LEHUqyvZAyNZCiJ68t06ZAX2ZA1jErWGygKYuqpEZAkJMhN6Ou0IYZAO0hXNNEnCFI0pl0PN3oWJvgrhQ4ZD"
url = "https://graph.facebook.com/v2.9/1601592959856152/feed?limit=200&access_token=" + token

resp = urllib.urlopen(url)
data = json.loads(resp.read())

post_url = "https://graph.facebook.com/v2.9/%s?fields=attachments&access_token=" + token

for elem in data["data"]:
    resp = urllib.urlopen(post_url % (elem["id"],))
    post_data = json.loads(resp.read())

    if "attachments" in post_data and "data" in post_data["attachments"]:
        for attachment in post_data["attachments"]["data"]:
            if "media" in attachment and "image" in attachment["media"]:
                print attachment["media"]["image"]["src"]

