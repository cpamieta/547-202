<?php
//Getting Started with the Finding API: Finding Items by Keywords, 20019,[Sourcecode]. 
//http://developer.ebay.com/DevZone/finding/HowTo/GettingStarted_PHP_NV_XML/GettingStarted_PHP_NV_XML.html

error_reporting(E_ALL);  // Turn on all errors, warnings and notices for easier debugging

// API request variables
$endpoint = 'http://svcs.ebay.com/services/search/FindingService/v1';  // URL to call
$version = '1.13.0';  // API version supported by your application
$appid = 'Chrispam-Predicti-PRD-653c6de86-295ed2c6';  // Replace with your own AppID
$globalid = 'EBAY-US';  // Global ID of the eBay site you want to search (e.g., EBAY-DE)
$query = 'coachella';  // You may want to supply your own query
$safequery = urlencode($query);  // Make the query URL-friendly
$i = '0';  // Initialize the item filter index to 0

// Create a PHP array of the item filters you want to use in your request
$filterarray =
  array(
    array(
    'name' => 'MinPrice',
    'value' => '100',
    'paramName' => 'Currency',
    'paramValue' => 'USD'),

	    array(
    'name' => 'categoryId',
    'value' => '173634',
    'paramName' => '',
    'paramValue' => ''),

);

// Generates an indexed URL snippet from the array of item filters
function buildURLArray ($filterarray) {
  global $urlfilter;
  global $i;
  // Iterate through each filter in the array
  foreach($filterarray as $itemfilter) {
    // Iterate through each key in the filter
    foreach ($itemfilter as $key =>$value) {
      if(is_array($value)) {
        foreach($value as $j => $content) { // Index the key for each value
          $urlfilter .= "&itemFilter($i).$key($j)=$content";
        }
      }
      else {
        if($value != "") {
          $urlfilter .= "&itemFilter($i).$key=$value";
        }
      }
    }
    $i++;
  }
  return "$urlfilter";
} // End of buildURLArray function

// Build the indexed item filter URL snippet
buildURLArray($filterarray);

// Construct the findItemsByKeywords HTTP GET call 
$apicall = "$endpoint?";
$apicall .= "OPERATION-NAME=findCompletedItems";
$apicall .= "&SERVICE-VERSION=$version";
$apicall .= "&SECURITY-APPNAME=$appid";
$apicall .= "&GLOBAL-ID=$globalid";
$apicall .= "&keywords=$safequery";
$apicall .= "&paginationInput.pageNumber=1";
$apicall .= "&paginationInput.entriesPerPage=100";

$apicall .= "$urlfilter";

// Load the call and capture the document returned by eBay API
$resp = simplexml_load_file($apicall);

// Check to see if the request was successful, else print an error
if ($resp->ack == "Success") {
  $results = '';
  // If the response was loaded, parse it and build links  \
  $file = fopen("electricdaisycarnival1.txt","a");
  $c = count($resp->searchResult);
$l = $resp->paginationOutput;
$xx = $l->totalEntries;


    $results .= "<tr><td><img src=\"$\"></td><td><a href=\"$\">$xx</a></td></tr>";

  foreach($resp->searchResult->item as $item) {
   //foreach($resp->paginationOutput->item as $item) {

 

  
    $pic   = $item->galleryURL;
    $link  = $item->viewItemURL;
    $title = $item->title;
	    $subtitle = $item->subtitle;
	
  $listingInfo = $item->listingInfo;
   $shippingInfo = $item->shippingInfo;
     $shippingServiceCost = $shippingInfo->shippingServiceCost;
     
        $sellingStatus = $item->sellingStatus;
     $sellingState = $sellingStatus->sellingState;
     

  
  $endTime = $listingInfo->endTime;
    $startTime = $listingInfo->startTime;

  
  $sellingStatus = $item->sellingStatus;
    $convertedCurrentPrice = $sellingStatus->convertedCurrentPrice;
	    $currentPrice = $sellingStatus->currentPrice;

	
	

	$data = $title.'>'.$convertedCurrentPrice.">".$startTime.">".$endTime.">".$link.">".$subtitle.">".$shippingServiceCost.">".$sellingState;
	  fputcsv($file,explode(',',$data));

    // For each SearchResultItem node, build a link and append it to $results
   $results .= "<tr><td><img src=\"$pic\"></td><td><a href=\"$pic\">$title</a></td></tr>";
  }
  fclose($file);
}
// If the response does not indicate 'Success,' print an error
else {
  $results  = "<h3>Oops! The request was not successful. Make sure you are using a valid ";
  $results .= "AppID for the Production environment.</h3>";
}
?>

<!-- Build the HTML page with values from the call response -->
<html>
<head>
<title>eBay Search Results for <?php echo $query; ?></title>
<style type="text/css">body { font-family: arial,sans-serif;} </style>
</head>
<body>

<h1>eBay Search Results for <?php echo $query; ?></h1>

<table>
<tr>
  <td>
    <?php echo $results;?>
  </td>
</tr>
</table>

</body>
</html>