let user_name = "";
let nickname = "";

function handleKeyPress(event) {
  if (event.key === "Enter") {
    sendMessage();
  }
}

function appendMessage(message, isUser) {
  const chatMessages = document.getElementById("chat-messages");
  const messageDiv = document.createElement("div");
  messageDiv.className =
    "chat-message flex items-center justify-" +
    (isUser ? "end" : "start") +
    " space-x-2";
  const iconDiv = document.createElement("div");
  iconDiv.className =
    "h-6 w-6 rounded-full " +
    (isUser ? "bg-green-500" : "bg-blue-500") +
    " flex items-center justify-center";
  const iconSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  iconSvg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  iconSvg.setAttribute("viewBox", "0 0 20 20");
  iconSvg.setAttribute("fill", "currentColor");
  const iconPath = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "path"
  );
  iconPath.setAttribute("fill-rule", "evenodd");
  iconPath.setAttribute(
    "d",
    "M10 2a8 8 0 1 0 0 16A8 8 0 0 0 10 2zM9 7a1 1 0 0 1 2 0v5a1 1 0 1 1-2 0V7zm1 9a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"
  );
  iconPath.setAttribute("clip-rule", "evenodd");
  iconSvg.appendChild(iconPath);
  iconDiv.appendChild(iconSvg);
  const textDiv = document.createElement("div");
  textDiv.className =
    "chat-text " +
    (isUser ? "bg-green-100" : "bg-blue-100") +
    " py-2 px-4 rounded-lg";
  textDiv.textContent = message;
  messageDiv.appendChild(iconDiv);
  messageDiv.appendChild(textDiv);
  chatMessages.appendChild(messageDiv);
}

function updateSentimentPlot(plotPath) {
  const plotContainer = document.getElementById("sentiment-plot-container");
  const plotImg = document.getElementById("sentiment-plot");
  plotImg.src = plotPath + "?t=" + new Date().getTime(); // Add cache-busting query parameter
  plotContainer.style.display = "block";
}

function generateRandomNickname() {
  const nicknames = ["Buddy", "Champ", "Pal", "BFF", "Bae"];
  const randomIndex = Math.floor(Math.random() * nicknames.length);
  return nicknames[randomIndex];
}

function generateRandomQuestion() {
  const questions = [
    "What's your favorite hobby?",
    "Do you have any pets?",
    "What's the last book you read?",
    "What's your favorite movie?",
    "Do you enjoy cooking?",
    "What's your dream travel destination?",
  ];
  const randomIndex = Math.floor(Math.random() * questions.length);
  return questions[randomIndex];
}

function sendMessage() {
  const userInput = document.getElementById("user-input");
  const message = userInput.value.trim();
  if (user_name === "") {
    user_name = message;
    appendMessage(`Hello, ${user_name}! What's your nickname?`, false);
  } else if (nickname === "") {
    nickname = message !== "" ? message : generateRandomNickname();
    appendMessage(
      `Nice to meet you, ${nickname}! How are you doing today?`,
      false
    );
  } else if (message !== "") {
    appendMessage(message, true);

    fetch("/chatbot", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: "text=" + encodeURIComponent(message),
    })
      .then((response) => response.json())
      .then((data) => {
        appendMessage(data.reply, false);
        updateSentimentPlot(data.plot_path);
        setTimeout(() => {
          appendMessage(generateRandomQuestion(), false);
        }, 500);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }
  userInput.value = "";
}
