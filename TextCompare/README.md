A Text compare webservices using python Flask framework. Pass in two strings and return a calculated percentage of how similar they are. 
Source file is located in the src folder. See the requirement text file for required imports.

Docker container located, 

https://hub.docker.com/r/cpamieta/textcompare


Sample request using notebook. Request is a json format which requires two fields of text1 and text2, the reply is also in json. See sample below.

import requests

res = requests.post(
    url='http://localhost:5000/compare',
    json={
        'text1': "The easiest way to earn points with Fetch Rewards is to just shop for the products you already love. If you have any participating brands on your receipt, you'll get points based on the cost of the products. You don't need to clip any coupons or scan individual barcodes. Just scan each grocery receipt after you shop and we'll find the savings for you.",
        'text2': "The easiest way to earn points with Fetch Rewards is to just shop for the items you already buy. If you have any eligible brands on your receipt, you will get points based on the total cost of the products. You do not need to cut out any coupons or scan individual UPCs. Just scan your receipt after you check out and we will find the savings for you.",   
    }
)


res.json()
