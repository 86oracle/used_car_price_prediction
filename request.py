import requests

url = " https://localhost:5000/results"
r = requests.post(url, json={"name" :"Golf " ,"vehicleType":"kleinwagen" , "brand":"volkswagen" , "model":"golf" ,"notRepairedDamage": "yes","fuelType":"diesel" ,"kilometer" : 150000 , "monthOfRegistration":1, "yearOfRegistration":1993 , "power(in bhp)":120 , "postalCode":70435 })
print(r.json())