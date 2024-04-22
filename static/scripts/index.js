async function analyzeSentiment() {
  const text = document.getElementById("sentiment").value;
  const selectedModel = document.getElementById("model").value;
  const sentimentResult = document.getElementById("sentimentResult");

  data = {
    model: selectedModel,
    text: text,
  };

  try {
    const response = await fetch("http://localhost:5000/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    const result = await response.json();
    console.log("Success:", result);
    sentimentResult.innerText = `Sentiment: ${result.score}`;
    sentimentResult.classList.remove("hidden");
  } catch (error) {
    console.error("Error:", error.message);
  }
}

const generate = async () => {
  const textbox = document.querySelector("#sentiment");
  const response = await fetch(
    `https://dummyjson.com/comments/1${Math.floor(Math.random() * 30) + 1}`
  );
  const text = await response.json();
  console.log(text.body);
  textbox.value = text.body;
};
