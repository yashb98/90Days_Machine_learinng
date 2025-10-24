import React, { useState } from "react";
import axios from "axios";

const SentimentForm: React.FC = () => {
  const [text, setText] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", { text });
      setResult(res.data.predicted_sentiment);
    } catch (err) {
      console.error(err);
      setResult("⚠️ API connection error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-screen h-screen flex flex-col justify-center items-center bg-gradient-to-br from-gray-100 via-gray-200 to-gray-300">
      {/* Card Container */}
      <div className="bg-white rounded-3xl shadow-2xl p-10 w-full max-w-3xl flex flex-col items-center">
        <h1 className="text-5xl font-extrabold text-gray-900 mb-4 text-center">
          Sentiment Analyser
        </h1>
        <p className="text-gray-600 text-lg mb-8 text-center">
          Analyse emotions in text using Machine Learning — right from your browser!
        </p>

        {/* Form */}
        <form
          onSubmit={handleSubmit}
          className="flex flex-col items-center w-full space-y-6"
        >
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type your thoughts here..."
            rows={5}
            className="w-full border border-gray-300 p-4 text-lg rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          />

          <button
            type="submit"
            disabled={loading}
            className="w-full !bg-black text-white text-lg font-semibold py-3 rounded-xl hover:bg-gray-800 transition-all duration-300"
          >
            {loading ? "Analysing..." : "Predict Sentiment"}
          </button>
        </form>

        {/* Result */}
        {result && (
          <div className="mt-10 text-center transition-all duration-300">
            <h2
              className={`text-3xl font-bold ${
                result === "positive"
                  ? "text-green-600"
                  : result === "negative"
                  ? "text-red-600"
                  : "text-yellow-600"
              }`}
            >
              {result === "positive" && "😊 Positive Sentiment!"}
              {result === "negative" && "😔 Negative Sentiment"}
              {result === "⚠️ API connection error" && "⚠️ API Error"}
            </h2>
            <p className="mt-3 text-gray-600">
              {result === "positive"
                ? "Your text radiates good vibes!"
                : result === "negative"
                ? "Seems like a downbeat tone detected."
                : ""}
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="mt-8 text-gray-600 text-sm">
        Built by{" "}
        <a
          href="https://github.com/yashb98"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline"
        >
          Yash Bishnoi
        </a>{" "} | 
        {" "}
        <a
          href="https://www.linkedin.com/in/yash-bishnoi-2ab36a1a5/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline"
        >
           LinkedIn
        </a>{" "} | {" "}
        <a
          href="https://hub.docker.com/r/yashbishnoi98/sentiment-flask-app"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline"
        >
           DockerHub
        </a>
      </footer>
    </div>
  );
};

export default SentimentForm;