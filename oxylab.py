import requests, json, csv
from dotenv import load_dotenv
import os
def get_job():
    load_dotenv()
    API_credentials = (os.environ.get("OXYLAB_USERNAME"), os.environ.get("OXYLAB_PASSWORD"))
    payload = {
        'source': 'universal',
        'url': 'https://stackshare.io/jobs',
        'geo_location': 'United States',
        'render': 'html',
        'browser_instructions': [
            {
                'type': 'click',
                'selector': {
                    'type': 'xpath',
                    'value': '//button[contains(text(), "Load more")]'
                }
            },
            {'type': 'wait', 'wait_time_s': 2}
        ] * 13 + [
            {
                "type": "fetch_resource",
                "filter": "^(?=.*https://km8652f2eg-dsn.algolia.net/1/indexes/Jobs_production/query).*"
            }
        ]
    }
    response = requests.request(
        'POST',
        'https://realtime.oxylabs.io/v1/queries',
        auth=API_credentials, 
        json=payload, 
        timeout=180
    )
    print("Response",response.json())
    results = response.json()['results'][0]['content']
    print(results)
    data = json.loads(results)

    jobs = []
    for job in data['hits']:
        parsed_job = {
            'Title': job.get('title', ''),
            'Location': job.get('location', ''),
            'Remote': job.get('remote', ''),
            'Company name': job.get('company_name', ''),
            'Company website': job.get('company_website', ''),
            'Verified': job.get('company_verified', ''),
            'Apply URL': job.get('apply_url', '')
        }
        jobs.append(parsed_job)

    fieldnames = [key for key in jobs[0].keys()]
    with open('stackshare_jobs.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in jobs:
            writer.writerow(item)
get_job()