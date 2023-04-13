const videoElement = document.getElementById('videoElement');
const timerElement = document.getElementById('timerElement');
const constraints = { audio: true, video: true };
const MIN_RECORDING_TIME = 5; // in seconds
const MAX_RECORDING_TIME = 30; // in seconds
let startTime;
let elapsedTime = 0;
let timerInterval;
$(document).ready(function () {
    var data = $('input[name="data"]').val();
    if (data == "TIP") {
        alert("Please hold your hand infront of camera showing tips of your fingers!")
    }
});
// Load the Google API client library
gapi.load('client', init);

// Initialize the Google API client library
function init() {
    gapi.client.init({
        apiKey: 'AIzaSyCcLQnen8T8qLUNxVUqzynf2n82NVQau_s',
        clientId: '649981568433-pq20i12pb35i8ckq69d7kl1fti5bkkl6.apps.googleusercontent.com',
        discoveryDocs: ['https://www.googleapis.com/discovery/v1/apis/drive/v3/rest'],
        scope: 'https://www.googleapis.com/auth/drive.file'
    }).then(function () {
        // Authenticate the user with Google
        return gapi.auth2.getAuthInstance().signIn();
    }).then(function () {
        // Get the user's access token
        const accessToken = gapi.auth2.getAuthInstance().currentUser.get().getAuthResponse().access_token;

        // Start recording the video
        navigator.mediaDevices.getUserMedia(constraints)
            .then(function (mediaStream) {
                videoElement.srcObject = mediaStream;
                const mediaRecorder = new MediaRecorder(mediaStream);
                mediaRecorder.ondataavailable = function (e) {
                    chunks.push(e.data);
                }

                mediaRecorder.onstop = function () {
                    // Save the recorded video to Google Drive
                    uploadToDrive(accessToken, chunks);
                }

                // Stop recording after 10 seconds
                setTimeout(function () {
                    mediaRecorder.stop();
                }, 10000);

            })
            .catch(function (err) {
                console.error('Could not access media devices', err);
            });
    });
}
function uploadToDrive(accessToken, chunks) {
    const fileMetadata = {
        name: 'recording.mp4',
        parents: ['1k3iJVKL14PHo5JL8ukyhN4QC_hD-DZd4'] // Replace with the ID of the folder where you want to save the video
    };
    const media = {
        mimeType: 'video/mp4',
        body: new Blob(chunks, { type: 'video/mp4' })
    };
    gapi.client.drive.files.create({
        resource: fileMetadata,
        media: media,
        fields: 'webViewLink',
        access_token: accessToken
    }).then(function (response) {
        // Get the link to the uploaded file
        const link = response.result.webViewLink;
        console.log('File uploaded:', link);
        // Do something with the link, such as display it to the user or send it to your server
    }, function (error) {
        console.error('Error uploading file:', error);
    });
}

// navigator.mediaDevices.getUserMedia(constraints)
//     .then(stream => {
//         videoElement.srcObject = stream;
//     })
//     .catch(err => {
//         console.error('Error accessing media devices', err);
//     });

let mediaRecorder;
let chunks = [];

const recordButton = document.getElementById('recordButton');
const recordingIndicator = document.getElementById('recordingIndicator'); // Added element to show recording status
recordButton.addEventListener('click', () => {
    recordButton.disabled = true;
    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            startTime = Date.now();
            timerInterval = setInterval(updateTime, 1000);

            mediaRecorder.addEventListener('dataavailable', event => {
                chunks.push(event.data);
            });

            mediaRecorder.addEventListener('stop', () => {
                clearInterval(timerInterval);
                elapsedTime = 0;
                timerElement.textContent = '00:00:00';
                const blob = new Blob(chunks, { type: 'video/mp4' });
                const videoURL = URL.createObjectURL(blob);
                const downloadLink = document.createElement('a');
                downloadLink.href = videoURL;
                downloadLink.download = 'C:/games/recording.mp4';
                downloadLink.click();
                // window.location.href = '/test'; //redirect to test
            });

            mediaRecorder.start();
            recordingIndicator.textContent = 'Recording......';
        });
});
stopButton = document.getElementById('stopButton')
stopButton.addEventListener('click', () => {
    mediaRecorder.stop();
    clearInterval(timerInterval);
    elapsedTime = 0;
    timerElement.textContent = '00:00:00';
    recordingIndicator.textContent = '';
});

function updateTime() {
    elapsedTime = Math.floor((Date.now() - startTime) / 1000);
    const hours = Math.floor(elapsedTime / 3600);
    const minutes = Math.floor((elapsedTime - (hours * 3600)) / 60);
    const seconds = elapsedTime - (hours * 3600) - (minutes * 60);
    const timeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    timerElement.textContent = timeString;
    if (elapsedTime >= MAX_RECORDING_TIME) {
        stopRecording();
        return;
    }

    if (elapsedTime <= MIN_RECORDING_TIME) {
        stopButton.disabled = true;
    } else {
        stopButton.disabled = false;
    }
}

function stopRecording() {
    mediaRecorder.stop();
    clearInterval(timerInterval);
    elapsedTime = 0;
    timerElement.textContent = '00:00:00';
    recordingIndicator.textContent = '';
    recordButton.disabled = false;
}

