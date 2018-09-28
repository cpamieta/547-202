
# coding: utf-8

# In[1]:


import takahe
import ginger_python2
import nltk


# In[53]:


#Tokenize the words
text = nltk.word_tokenize("AI equipped machine has ever flown to space before project team members said. The mission of the bantam astronaut assistant known as CIMON short for Crew Interactive Mobile Companion is relatively short and modest. But its work off Earth could help pave the way for some pretty big things according to NASA officials. Having AI  having that knowledge base and the ability to tap into it in a way that is useful for the task that you are doing  is really critical for having humans further and further away from the planet Kirk Shireman NASAs International Space Station ISS program manager said during a prelaunch news conference yesterday June 28. We have to have autonomy he added. We will have to have tools like this to have the species successfully live far away from Earth. CIMON was developed by the European aerospace company Airbus on behalf of the German space agency which is known by its German acronym DLR. The robots AI is IBMs famous Watson system. CIMON is roughly spherical and weighs 11 lbs. 5 kilograms. The robot can converse with people and it knows whom its talking to thanks to facial recognition software. CIMON has a face of its own a simple cartoon one. The astronaut assistant is also mobile; once aboard the ISS CIMON will be able to fly around by sucking air in and expelling it through special tubes. Though CIMON is flexible enough to interact with anyone it is tailored to European Space Agency astronaut Alexander Gerst who arrived at the ISS aboard a Russian Soyuz spacecraft earlier this month. CIMONs mission calls for the robot to work with Gerst on three separate investigations. They will experiment with crystals work together to solve the Rubiks cube and perform a complex medical experiment using CIMON as an intelligent flying camera Airbus representatives wrote in a mission description earlier this year. CIMON will be a very involved partner in this work which will take a total of 3 hours. Alexander Gerst could say something like CIMON could you please help me perform a certain experiment? Could you please help me with the procedure? Philipp Schulien a CIMON system engineer at Airbus said during a different news conference yesterday. And then CIMON will fly towards Alexander Gerst and they will already start the communication. CIMON will be able to access lots of relevant information including photos and videos about the procedure in question. And the astronaut assistant is smart enough to deal with questions beyond the procedure that Gerst might have Schulien added. IMONs mission is a technology demonstration designed to show researchers how humans and machines can interact and collaborate in the space environment. It will be a while before intelligent robots are ready to do any really heavy lifting in the final frontier say helping astronauts repair damaged spacecraft systems or treating sick crew members. But that day is probably coming. For us this is a piece of the future of human space flight Christian Karrasch CIMON project leader at DLR said yesterday. If you go out to the moon or to Mars you cannot take all mankind and engineers with you Karrasch added. So the astronauts they will be on their own. But with an artificial intelligence you have instantly all the knowledge of mankind.")
text1 = nltk.word_tokenize("A beautiful space exploration friendship between human and machine may have just begun. Early this morning June 29 a small robot endowed with artificial intelligence AI launched on a two day trip to the International Space Station aboard Space X Dragon cargo capsule. No other AI equipped machine has ever flown to space before project team members said. The mission of the bantam astronaut assistant known as CIMON short for Crew Interactive Mobile Companion is relatively short and modest. But its work off Earth could help pave the way for some pretty big things according to NASA officials.")
text2 = nltk.word_tokenize("The White Houses use of a national security argument to justify the duties against a close ally along with President Trumps repeated belittling of both Mr. Trudeau and his trade policies has offended and angered Canadians. On social media they are calling for boycotts of American products and encouraging one another to look elsewhere for vacation destinations. Mr Trudeaus decision to retaliate won a rare endorsement from all three of Canadas major political parties.")
textList1 = ["The/DT wife/NN of/IN a/DT former/JJ U.S./NNP president/NN Bill/NNP Clinton/NNP Hillary/NNP Clinton/NNP visited/VBD China/NNP last/JJ Monday/NNP ./PUNCT", "Hillary/NNP Clinton/NNP wanted/VBD to/TO visit/VB China/NNP last/JJ month/NN but/CC postponed/VBD her/PRP$ plans/NNS till/IN Monday/NNP last/JJ week/NN ./PUNCT", "Hillary/NNP Clinton/NNP paid/VBD a/DT visit/NN to/TO the/DT People/NNP Republic/NNP of/IN China/NNP on/IN Monday/NNP ./PUNCT", 
"Last/JJ week/NN the/DT Secretary/NNP of/IN State/NNP Ms./NNP Clinton/NNP \
visited/VBD Chinese/JJ officials/NNS ./PUNCT"]                          
tokenText= nltk.pos_tag(text)

#convert list to string needed for formatting. Once formatting is done, convert into a list where each sentese is a index.
textString = " ".join(str(xx) for xx in tokenText)
textString=textString.replace("('.', '.')", "./PUNCT,")
textString=textString.replace("('?', '.')", "?/PUNCT,")
textString=textString.replace("('!', '.')", "!/PUNCT,")

textString=textString.replace("('", "")

textString=textString.replace("')", "")
textString=textString.replace("', '", "/")
#textList=textString.split(',')
textList = [x.strip() for x in textString.split(',')]

#delete the empty index
del textList[-1]
#textList[1]=textList[1].lstrip()
print(textList[2])


# In[55]:


#!/usr/bin/python

#sample code provided by “boudinfl/takahe.” GitHub,github.com/boudinfl/takahe

# Create a word graph from the set of sentences with parameters :
# - minimal number of words in the compression : 6
# - language of the input sentences : en (english)
# - POS tag for punctuation marks : PUNCT
compresser = takahe.word_graph( textList, 
                                nb_words =20, 
                                lang = 'en', 
                                punct_tag = "PUNCT" )

# Get the 50 best paths
candidates = compresser.get_compression(50)

# 1. Rerank compressions by path length (Filippova's method)
for cummulative_score, path in candidates:

# Normalize path score by path length
    normalized_score = cummulative_score / len(path)

    # Print normalized score and compression
    print round(normalized_score, 3), ' '.join([u[0] for u in path])

# Write the word graph in the dot format
compresser.write_dot('test.dot')

# 2. Rerank compressions by keyphrases (Boudin and Morin's method)
reranker = takahe.keyphrase_reranker( textList,  
                                      candidates, 
                                      lang = 'en' )

reranked_candidates = reranker.rerank_nbest_compressions()
b=reranked_candidates[len(reranked_candidates)-1]
print(b[1])
gg= ' '.join([u[0] for u in b[1]])
# Loop over the best reranked candidates
for score, path in reranked_candidates:
    
    
# Print the best reranked candidates
    print round(score, 3), ' '.join([u[0] for u in path])
    print round(score, 3), ' '.join([u[0] for u in path])


# In[27]:


#send text into grammar checker. 
ginger_python2.main(gg)

