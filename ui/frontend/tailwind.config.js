/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        brand: {
          50:  "#f0f4ff",
          100: "#dde8ff",
          200: "#c3d4ff",
          300: "#9db4ff",
          400: "#748aff",
          500: "#4f5fff",
          600: "#3a3ff5",
          700: "#2f2fd8",
          800: "#2929ae",
          900: "#272a89",
          950: "#171850",
        },
        risk: {
          low:      "#22c55e",
          moderate: "#f59e0b",
          high:     "#ef4444",
          critical: "#7c3aed",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
};
