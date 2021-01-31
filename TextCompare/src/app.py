#import os 
#import joblib
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)


class Compare(Resource):
    @staticmethod
    def post():
        #convert json to object
        data = request.get_json()
        text1 = data['text1']
        text2 = data['text2']
        #remove certain char from the string
        text1=text1.replace(",", "")
        text1=text1.replace(".", "")
        text1=text1.replace("?", "")
        text1=text1.replace("!", "")

        text2=text2.replace(",", "")
        text2=text2.replace(".", "")
        text2=text2.replace("?", "")
        text2=text2.replace("!", "")
        
        text1a = text1.split(" ")
        text2a = text2.split(" ")
 
 
        hashmap = {}
        count=0
        #Find the the text with the greatest amount of words.
        #Create a hashmap using that largest text where the words are the key and the value would be the count of occurrence.        
        #loop over the other text and if the word is in the hashmap, keep a count and remove a count from the hashmap
        #return a caluclated percent of similarity by the count of matching words by the total count from the longest text object.
        if(len(text1a) > len(text2a)):
            
            for word in text1a:
                if(hashmap.get(word) == None):                    
                    hashmap[word] = 1
                else:
                    hashmap[word] = hashmap[word]  +1

            for word2 in text2a:
                if(hashmap.get(word2) != None and hashmap.get(word2) != 0):
                    count +=1                 
                    hashmap[word] = hashmap[word]  -1
        else:           
            for word in text2a:
                if(hashmap.get(word) == None):                   
                    hashmap[word] = 1
                else:
                    hashmap[word] = hashmap[word]  +1
                   
            for word2 in text1a:
                if(hashmap.get(word2) != None and hashmap.get(word2) > 0):
                    count +=1                   
                    hashmap[word2] = hashmap[word2]  -1    

            cal = count / len(text2a)         
#return a json reply 
        return jsonify({
                'similarityPercent': cal
        })

api.add_resource(Compare, '/compare')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5000)