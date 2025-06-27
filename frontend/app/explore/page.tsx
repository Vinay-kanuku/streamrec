'use client';
import { useEffect, useState } from 'react';

type Movie = {
  movie_id: number;
  title: string;
  genres: string;
};

export default function ExplorePage() {
  const [queue, setQueue] = useState<Movie[]>([]);
  const [index, setIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const userId = 1; // You can make this dynamic later

  const current = queue[index];

  useEffect(() => {
    // Ideally, fetch from backend. For now, mock it
    const mockMovies: Movie[] = [
      { movie_id: 1, title: 'Inception', genres: 'Action|Sci-Fi' },
      { movie_id: 2, title: 'Interstellar', genres: 'Adventure|Drama|Sci-Fi' },
      { movie_id: 3, title: 'The Matrix', genres: 'Action|Sci-Fi' },
    ];
    setQueue(mockMovies);
    setLoading(false);
  }, []);

  const handleAction = async (action: 'like' | 'dislike' | 'skip') => {
    if (!current) return;

    await fetch('http://localhost:8000/event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        movie_id: current.movie_id,
        action: action,
        timestamp: Date.now(),
      }),
    });

    setIndex((prev) => prev + 1);
  };

  return (
    <main className="min-h-screen bg-zinc-950 text-white flex flex-col items-center justify-center p-6">
      <h1 className="text-3xl font-bold mb-6 text-center">🔥 Discover Movies</h1>

      {loading ? (
        <p>Loading movies...</p>
      ) : !current ? (
        <p className="text-xl text-center">🎉 No more movies in queue</p>
      ) : (
        <div className="bg-zinc-800 rounded-xl shadow-lg p-6 w-full max-w-md text-center transition-all">
          <h2 className="text-2xl font-semibold mb-2">{current.title}</h2>
          <p className="text-sm text-zinc-400 mb-6">{current.genres}</p>

          <div className="flex justify-center gap-4">
            <button
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded shadow"
              onClick={() => handleAction('dislike')}
            >
              👎 Dislike
            </button>
            <button
              className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded shadow"
              onClick={() => handleAction('skip')}
            >
              ⏭️ Skip
            </button>
            <button
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded shadow"
              onClick={() => handleAction('like')}
            >
              👍 Like
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
