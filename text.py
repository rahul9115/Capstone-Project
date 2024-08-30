import http.client
import json

conn = http.client.HTTPSConnection("linkedin-data-api.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "747c1de4bbmsh8539df016695067p1910d9jsnd710b80a2d66",
    'x-rapidapi-host': "linkedin-data-api.p.rapidapi.com"
}

conn.request("GET", "/search-jobs?keywords=golang&locationId=92000000&datePosted=anyTime&sort=mostRelevant", headers=headers)

res = conn.getresponse()
data = res.read()
json_data = json.loads(data.decode('utf-8'))
with open("data.json","w") as file:
    json.dump(json_data,file)

print(data.decode("utf-8"))