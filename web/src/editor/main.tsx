// Editor entry (P2): placeholder replaced by the full editor in phase 2.
import React from "react";
import { createRoot } from "react-dom/client";

function App() {
  return (
    <div style={{ padding: 24, fontFamily: "sans-serif" }}>
      <h1>arrow-toy editor</h1>
      <p>Editor UI ships in phase P2. The render page and geometry contract are
      already available.</p>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
