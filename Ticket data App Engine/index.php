<?php
/**
 * Copyright 2018 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

namespace Google\Cloud\Samples\AppEngine\Storage;

use Google\Auth\Credentials\GCECredentials;

require_once __DIR__ . '/vendor/autoload.php';

$bucketName = getenv('GOOGLE_STORAGE_BUCKET');
$projectId = getenv('GOOGLE_CLOUD_PROJECT');
$defaultBucketName = sprintf('%s.appspot.com', $projectId);

register_stream_wrapper($projectId);

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    write_stream($bucketName, 'ebayTic.csv', $_REQUEST['content']);

    header('Location: /');
    exit;
}

?>
<!DOCTYPE HTML>
<html>
  <head>
    <title>Ebay api</title>
  </head>

  <body>
    <h1>Ebay api</h1>

    <div>
        <form action="/write/stream" method="post">
            Ebay search term:<br />
            <textarea name="content"></textarea><br />
            <input type="submit" />
        </form>

    <div>

</html>

