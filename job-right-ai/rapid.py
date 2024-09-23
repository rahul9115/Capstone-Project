import http.client
import json
conn = http.client.HTTPSConnection("linkedin-data-api.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "2124cb6875msh38ecc4814e40470p11ee41jsn2c6006d4d820",
    'x-rapidapi-host': "linkedin-data-api.p.rapidapi.com"
}


def get_job_description(id):
    conn.request("GET", f"/get-job-details?id={id}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    json_data = json.loads(data.decode('utf-8'))
    with open("job_description.json","a") as file:
        json.dump(json_data,file)


def get_job_details(role):
    # conn.request("GET", f"/search-jobs?keywords={role}&locationId=92000000&datePosted=anyTime&sort=mostRelevant", headers=headers)
    # res = conn.getresponse()
    # data = res.read()
    # json_data = json.loads(data.decode('utf-8'))
    # with open("data.json","w") as file:
    #     json.dump(json_data,file)
    with open("data.json","r") as file:
        data=json.load(file)
    for i in range(2):
        get_job_description(data["data"][i]["id"])
get_job_details("DataScientist")