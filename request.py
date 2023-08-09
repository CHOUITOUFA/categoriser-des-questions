
import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'tags':50})

print(r.json())