/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#1a1a2e",
        panel: "#16213e",
        accent: "#0f3460",
        hi: "#e94560",
        text: "#e0e0e0",
        muted: "#888899",
        border: "#2a2a4a",
        card: "#1e1e3a",
        success: "#4ecca3",
        warning: "#f5a623",
        canvas: "#0a0a1a",
      },
    },
  },
  plugins: [],
};
