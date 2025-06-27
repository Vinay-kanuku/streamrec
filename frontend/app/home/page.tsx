'use client';
import { useEffect, useState } from 'react';

type Movie = {
  movie_id: number;
  title: string;
  poster_url: string;
};

export default function HomePage() {
  const [movies, setMovies] = useState<Movie[]>([]);
  const [watching, setWatching] = useState<Movie | null>(null);
  const [watchStart, setWatchStart] = useState<number>(0);

  const userId = 1; // later make dynamic

  useEffect(() => {
    // MOCK: replace with real backend call
    setMovies([
      { movie_id: 101, title: "Inception", poster_url: "/image.png" },
      { movie_id: 102, title: "The Matrix", poster_url: "/matrix.jpg" },
      { movie_id: 103, title: "Interstellar", poster_url: "/interstellar.jpg" },
    ]);
  }, []);

  const logEvent = async (movie_id: number, event: string, extra = {}) => {
    await fetch('http://localhost:8000/event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        movie_id,
        event,
        timestamp: Date.now(),
        ...extra,
      }),
    });
  };

  const handleHover = (movie_id: number) => {
    logEvent(movie_id, 'hover');
  };

  const handleWatch = (movie: Movie) => {
    logEvent(movie.movie_id, 'watch_start');
    setWatching(movie);
    setWatchStart(Date.now());
  };

  const handleCloseWatch = () => {
    if (watching) {
      const duration = (Date.now() - watchStart) / 1000;
      logEvent(watching.movie_id, 'watch_time', { duration });
      setWatching(null);
    }
  };

  return (
    <main className="min-h-screen bg-black text-white p-6">
      <h1 className="text-3xl font-bold mb-6">🎬 Explore Movies</h1>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
        {movies.map((movie) => (
          <div
            key={movie.movie_id}
            onMouseEnter={() => handleHover(movie.movie_id)}
            onClick={() => handleWatch(movie)}
            className="cursor-pointer relative group hover:scale-105 transition-transform duration-300"
          >
            <img
              src={movie.poster_url}
              alt={movie.title}
              className="rounded-lg w-full h-[250px] object-cover"
            />
            <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/80 to-transparent text-sm font-medium">
              {movie.title}
            </div>
          </div>
        ))}
      </div>

      {/* WATCH MODAL */}
      {watching && (
        <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center">
          <div className="bg-zinc-800 p-6 rounded-lg shadow-lg w-[90%] max-w-xl">
            <h2 className="text-2xl font-bold mb-4">{watching.title}</h2>
            <p className="mb-4 text-zinc-300">🎥 Simulated player...</p>
            <button
              onClick={handleCloseWatch}
              className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded"
            >
              ❌ Close
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
