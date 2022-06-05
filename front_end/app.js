var express = require('express'); // require express module to run frontend server 

var app = express(); // declarations

app.use(express.static('.')); // can access and use html pages in the current directory


app.get('/', function(req, res) {
    res.render('index.html'); // when we enter localhost:8000 in browser, we tell the app to render/display index.html
})


// boiler plate code for building a POST http request
async function postData(url = '', data = {}) {
    const response = await fetch(url, {
      method: 'POST', // *GET, POST, PUT, DELETE, etc.
      mode: 'cors', // no-cors, *cors, same-origin
      cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
      credentials: 'same-origin', // include, *same-origin, omit
      headers: {
        'Content-Type': 'application/json'
      },
      redirect: 'follow', // manual, *follow, error
      referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
      body: JSON.stringify(data) // body data type must match "Content-Type" header
    });
    return response.json(); // parses JSON response into native JavaScript objects
}

function saveRec(req) {
    console.log('saveReq: req = ' + req) // log the request value

    var response_from_server;

    // call the postData function with url = backend server url
    // and request body = json object with inputText = user entered data
    postData("http://localhost:5005/saverecording", { "filename": req })
        .then(data => { // we process the response received from server
            console.log(data); // JSON data parsed by `data.json()` call
            if(data['success'] == true) { // set response accordingly
              response_from_server = data['src']
              document.getElementById("spectrograph").src = response_from_server; 
              document.getElementById("specto-caption").innerText = response_from_server;
              document.getElementById("genspecto").classList.remove('btn-light');
	            document.getElementById("genspecto").classList.add('btn-success');
              document.getElementById("genspecto").innerText = "Conversion Successful";
              document.getElementById("getinference").disabled = false;
            } else {
              response_from_server = data['message']
              document.getElementById("genspecto").classList.remove('btn-light');
	            document.getElementById("genspecto").classList.add('btn-danger');
              document.getElementById("genspecto").innerText = "Conversion Failed"
            }
            console.log(response_from_server) // log response
            // set the read only response field with data received from server
            document.getElementById("genspecto").disabled = true
    });
      
}

function callInf(req) {
  console.log('callInf: req = ' + req) // log the request value

  var response_from_server;
  var google_card = document.getElementById('google_card');
  var google_pred = document.getElementById('google_pred');
  var google_raw = document.getElementById('google_raw');

  var dense_card = document.getElementById('dense_card');
  var dense_pred = document.getElementById('dense_pred');
  var dense_raw = document.getElementById('dense_raw');
  // call the postData function with url = backend server url
  // and request body = json object with inputText = user entered data
  postData("http://localhost:5005/inference", { "filename": req })
      .then(data => { // we process the response received from server
          console.log(data); // JSON data parsed by `data.json()` call
          if(data['success'] == true) { // set response accordingly
            google_pred.innerText += data["google_pred"];
            google_raw.innerText += data["google_raw"];
            dense_pred.innerText += data["dense_pred"];
            dense_raw.innerText += data["dense_raw"];

            if(data["google_result"] == true) {
              google_card.classList.add('text-bg-success');
            }
            else {
              google_card.classList.add('text-bg-danger');
            }

            if(data["dense_result"] == true) {
              dense_card.classList.add('text-bg-success');
            }
            else {
              dense_card.classList.add('text-bg-danger');
            }
          } else {
            google_pred.innerText += " ERROR";
            google_raw.innerText += " ERROR";
            dense_pred.innerText += " ERROR";
            dense_raw.innerText += " ERROR";
          }
          console.log(response_from_server) // log response
          // set the read only response field with data received from server
          document.getElementById("getinference").disabled = true;
  });
    
}

// VERY IMP. Tell the app which port to run on. Should NOT be the same as the backend server port
app.listen(8000);
