import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*", // Proxy to FastAPI backend
      },
      {
        source: "/history-images/:path*",
        destination: "http://localhost:8000/history-images/:path*", // Proxy static images
      },
    ];
  },
};

export default nextConfig;
