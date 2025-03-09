function startRecognition() {
    document.getElementById('letters').textContent = "Detectando...";
    document.getElementById('corrected').textContent = "Aguardando...";

    fetch('/start_recognition', { method: 'POST' })
        .then(res => res.json())
        .then(data => alert(data.status));
}

function fetchResult() {
    fetch('/get_recognition_result')
        .then(res => res.json())
        .then(data => {
            document.getElementById('letters').textContent = data.raw_word;
            document.getElementById('corrected').textContent = data.corrected_word;
        });
}

setInterval(fetchResult, 2000);
