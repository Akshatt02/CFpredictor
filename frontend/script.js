function fetchRecommendations() {
    const handle = document.getElementById('handle').value.trim();
    document.getElementById('results').innerHTML = `<p>Loading...</p>`;
    const req = { handle };
    document.getElementById('results').style.display = "block";
    fetch('http://localhost:5000', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        if (data.error) {
            document.getElementById('results').innerHTML = `<p class="error-text">Error: ${data.error}</p>`;
        } else {
            let resultHTML = `<p>Current Rating: <strong>${data.predicted_rating - data.rating_change}</strong></p>`;
            resultHTML += `<p>Predicted Rating after 5 contests: <strong>${data.predicted_rating}</strong></p>`;
            if (data.rating_change > 0) {
                resultHTML += `<p>Rating Change: <strong>+${data.rating_change}</strong></p>`;
            }
            else if (data.rating_change < 0) {
                resultHTML += `<p>Rating Change: <strong>${data.rating_change}</strong></p>`;
            } else {
                resultHTML += `<p>Rating Change: <strong>${data.rating_change}</strong></p>`;
            }
            resultHTML += `<h3>Recommended Problems:</h3><ul>`;
            data.recommendations.forEach(problem => {
                resultHTML += `<li><a href="${problem.url}" target="_blank">${problem.name}</a></li>`;
            });
            resultHTML += `</ul>`;
            document.getElementById('results').innerHTML = resultHTML;
        }
    })
    .catch(error => {
        document.getElementById('results').innerHTML = `<p class="error-text">Error: Unable to fetch data. Please try again later.</p>`;
    });
}