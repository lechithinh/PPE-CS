function preview() {
    const frame = document.getElementById('frame');
    frame.src = URL.createObjectURL(event.target.files[0]);
}

async function detect() {
    // Get the image file from the file input
    const image_file = document.getElementById("formFile").files[0];

    // Create a FormData object and append the image file to it
    const form_data = new FormData();
    form_data.append("image", image_file);

    try {
        // Send a POST request to the API endpoint to detect the image
        const response = await fetch("http://127.0.0.1:8000/detect_image/", {
            method: "POST",
            body: form_data,
        });
        console.log(response)
        // If the response is successful, get the response as an image
        const result_image = await response.blob();
        
        // Create a URL for the result image
        const result_image_url = URL.createObjectURL(result_image);

        // Set the src attribute of the result image element to the result image URL
        document.getElementById("display").src = result_image_url;
        
    } catch (error) {
        console.error(error);
    }
}

const detectButton = document.getElementById('detect-button');
detectButton.addEventListener('click', detect);