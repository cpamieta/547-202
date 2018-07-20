                                                                Intro
Pinpointing the optimal time to buy or sell music festival tickets offers buyers the best deals and sellers the highest sale prices, respectively.  Predicting the best time to buy or sell music festival tickets can lead a seller to making a profit, or help a buyer find an affordable deal to attend a concert. By using Ebay API to gather data, this paper looks into different machine learning models to predict if the auction will sell at the current listed price and if a buyer should buy today or wait. 


                                                                Data Gathering
For the proof on concept, just the Ebay API was used to gather data. A PHP script was created that utilize the Ebay APi; tick.php. The end result of the tick.php script was a generated csv file named ticketDatafinal.csv. 


                                                                 Data Clean

Once the data was obtained, some formatting needed to take place. This also included feature engineering where new features where created using the data provided and domain knowledge. This was done in the DataClean.py. 


                                                                      Model

Last part was to create few different models to predict the price of the ticket. This can be seen in the Training+model.py . 

