"use client";
import { useState } from "react";

type Recommendation = {
  title: string;
  score: number;
};

export default function RecommendPage() {
  const [userId, setUserId] = useState("");
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchRecommendations = async () => {
    if (!userId) return;

    setLoading(true);
    setError("");
    try {
      const res = await fetch(`http://localhost:8000/recommend/${userId}`);
      if (!res.ok) throw new Error("User not found or server error");

      const data = await res.json();
      setRecommendations(data.recommendations ?? []);
    } catch (err) {
      setError("Failed to fetch recommendations.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="max-w-xl mx-auto px-4 py-16">
      <h1 className="text-3xl font-bold mb-6 text-center">
        🎬 Movie Recommendations
      </h1>

      <div className="flex gap-2 items-center mb-6">
        <input
          type="number"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          placeholder="Enter User ID"
          className="flex-1 border px-4 py-2 rounded shadow text-black"
        />
        <button
          onClick={fetchRecommendations}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
        >
          Get Recs
        </button>
      </div>

      {loading && <p className="text-center text-sm">Fetching...</p>}
      {error && <p className="text-red-600 text-center">{error}</p>}

      <ul className="space-y-4 mt-6">
        {recommendations.map((rec, index) => (
          <li
            key={index}
            className="border rounded p-4 shadow-sm hover:shadow-md transition"
          >
            <div className="text-lg font-semibold">{rec.title}</div>
            <div className="text-sm text-gray-600">Score: {rec.score.toFixed(2)}</div>
          </li>
        ))}
      </ul>
    </main>
  );
}
