/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg:        '#1a1a2e',
        panel:     '#16213e',
        accent:    '#0f3460',
        highlight: '#e94560',
        muted:     '#888888',
      },
    },
  },
  plugins: [],
}
