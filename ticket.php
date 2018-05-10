<?php
// This is the data you want to pass to Python
#"# of ticket", "CAR CAMPING PASS", "VIP", "VIP Parking", "weekend","days till","PriceRange",
 #                     "ticketPrice","trend","Price","Shuttle Passes","avg unsold price","avg sold price"  


$item='1,0,0,0,2,2,12,300,41,375,1,532.299,488.427';
$tmp = exec("python C:/xampp/htdocs/ticketpredictCopy/test.py" .$_POST["CARCAMPINGPASS"] .$_POST["vip"] .$_POST["weekend"] .$_POST["days"] .$_POST["price"] .$_POST["ShuttlePasses"]);
echo $tmp;
	
	

//echo $_POST["CARCAMPINGPASS"] .$_POST["vip"] .$_POST["weekend"] .$_POST["days"] .$_POST["price"]


?>