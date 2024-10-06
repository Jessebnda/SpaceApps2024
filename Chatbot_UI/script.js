// Evento para enviar mensajes
document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

// Evento para limpiar el chat
document.getElementById('clear-button').addEventListener('click', clearChat);

// Función para enviar mensajes
function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    if (message === '') return;

    displayMessage(message, 'user');
    userInput.value = '';

    // Llamada al modelo para obtener la respuesta
    getResponse(message).then(response => {
        displayMessage(response, 'bot');
    });
}

// Función para mostrar mensajes en el chat
function displayMessage(message, sender) {
    const chatWindow = document.getElementById('chat-window');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);
    messageElement.textContent = message;
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Función para interactuar con tu modelo
async function getResponse(message) {
    try {
        const response = await fetch('http://localhost:5000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        const data = await response.json();
        return data.response; // Asumiendo que tu API devuelve { response: "..." }
    } catch (error) {
        console.error('Error al obtener la respuesta:', error);
        return "Lo siento, hubo un error al procesar tu solicitud.";
    }
}

// Evento para el botón que expande el chat
document.getElementById('chat-toggle-button').addEventListener('click', openChat);

// Evento para el botón que minimiza el chat
document.getElementById('minimize-button').addEventListener('click', closeChat);

// Función para abrir el chat y ocultar el botón pequeño
function openChat() {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.style.display = 'flex';
    document.getElementById('chat-toggle-button').style.display = 'none';
}

// Función para cerrar el chat y mostrar el botón pequeño
function closeChat() {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.style.display = 'none';
    document.getElementById('chat-toggle-button').style.display = 'flex';
}

// Función para limpiar el chat
function clearChat() {
    const chatWindow = document.getElementById('chat-window');
    chatWindow.innerHTML = '';
}