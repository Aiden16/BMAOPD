const videoElement = document.getElementById('videoElement');
const timerElement = document.getElementById('timerElement');
const constraints = { audio: true, video: true };
const MIN_RECORDING_TIME = 5; // in seconds
const MAX_RECORDING_TIME = 30; // in seconds
let startTime;
let elapsedTime = 0;
let timerInterval;

navigator.mediaDevices.getUserMedia(constraints)
    .then(stream => {
        videoElement.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing media devices', err);
    });

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
                downloadLink.download = 'recording.mp4';
                downloadLink.click();
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
