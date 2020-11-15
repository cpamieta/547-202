Creating a ETL using google dataflow.

Steps of the ETL:

1)Get the ticket data off google data storage.

2)Transform the data using feature engineering on the data.

3)Store the final data on BigQuery 



Running dataFlow:

1) set gcloud config set project ticket-prediction

2)Navigate to the python code.

3) run,source /home/cpamieta/env/bin/activate

4)execute pyton script, python3 ./df06.py -p $DEVSHELL_PROJECT_ID -b ticket-prediction.appspot.com
