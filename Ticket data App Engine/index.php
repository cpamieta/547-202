<?php
//Getting Started with the Finding API: Finding Items by Keywords, 20019,[Sourcecode]. 
//http://developer.ebay.com/DevZone/finding/HowTo/GettingStarted_PHP_NV_XML/GettingStarted_PHP_NV_XML.html

error_reporting(E_ALL);  // Turn on all errors, warnings and notices for easier debugging

//Google cloud settings
use Google\Cloud\Storage\StorageClient;

$app = array();
$app['bucket_name'] = "ticket-prediction.appspot.com";
$app['project_id'] = getenv('GCLOUD_PROJECT');
// API request variables
$endpoint = 'https://svcs.ebay.com/services/search/FindingService/v1';  // URL to call
$version = '1.13.0';  // API version supported by your application
$appid = 'chrispam-ticketpr-PRD-56c5a7b3e-f41fdc91';  // Replace with your own AppID
$globalid = 'EBAY-US';  // Global ID of the eBay site you want to search (e.g., EBAY-DE)
$query = 'coachella';  // You may want to supply your own query
$safequery = urlencode($query);  // Make the query URL-friendly
$i = '0';  // Initialize the item filter index to 0

// Create a PHP array of the item filters you want to use in your request

// Check to see if the request was successful, else print an error
  $results = '';
  // If the response was loaded, parse it and build links  \
  $file = fopen("gs://".$app['bucket_name']."/ticketData.csv","a");
  
 
	
	
                                          
	$data = "test";
	  fputcsv($file,explode(',',$data));
	  

    // For each SearchResultItem node, build a link and append it to $results
  
  fclose($file);


?>
<html>
<head>
<title>eBay Search Results for /title>
<style type="text/css">body { font-family: arial,sans-serif;} </style>
</head>
<body>

<h1>eBay Search Results for </h1>

<table>
<tr>
  <td>
  </td>
</tr>
</table>

</body>
</html>