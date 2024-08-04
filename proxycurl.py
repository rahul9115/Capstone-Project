import requests
import json

api_key = 'iA5aUG0mAmmZcpJhGTmNww'
headers = {'Authorization': 'Bearer ' + api_key}
api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
params = {
    'linkedin_profile_url': 'https://www.linkedin.com/in/alok-saxena-2756697',
    'extra': 'include',
    'github_profile_id': 'include',
    'facebook_profile_id': 'include',
    'twitter_profile_id': 'include',
    'personal_contact_number': 'include',
    'personal_email': 'include',
    'inferred_salary': 'include',
    'skills': 'include',
    'use_cache': 'if-present',
    'fallback_to_cache': 'on-error',
}
response = requests.get(api_endpoint,
                        params=params,
                        headers=headers,verify=False)
print("The response",response.json())

with open("linkeldn.json","w") as file:
    json.dump(response.json(),file)