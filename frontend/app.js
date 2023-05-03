function preview() {
    const frame = document.getElementById('frame');
    frame.src = URL.createObjectURL(event.target.files[0]);
}

async function detect() {
    // Get the image file from the file input
    const url = "http://localhost:8000/detect/";
    const image_file = document.getElementById("formFile").files[0];

    // Create a FormData object and append the image file to it
    const form_data = new FormData();
    form_data.append("image", image_file);

    fetch(url, {
        method: "POST",
        body: form_data
    }).then(response => response.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            document.getElementById("display").src = url
        })
        .catch(error => console.error(error));
}

const detectButton = document.getElementById('detect-button');
detectButton.addEventListener('click', detect);