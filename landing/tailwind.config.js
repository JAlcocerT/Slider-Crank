/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        paper:    '#E8E4DD',
        signal:   '#E63B2E',
        offwhite: '#F5F3EE',
        ink:      '#111111',
      },
      fontFamily: {
        grotesk:      ['"Space Grotesk"', 'sans-serif'],
        'serif-drama': ['"DM Serif Display"', 'serif'],
        'mono-data':  ['"Space Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
}
