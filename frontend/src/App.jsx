import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import styled from "styled-components";

// Import your LandingPage component (make sure it's named in PascalCase)
import LandingPage from "./components/landingpage";  // Adjust the path if necessary

const AppContainer = styled.div`
  height: 100vh;
  width: 100vw;
  overflow-y: auto; /* Enable vertical scrolling */
  overflow-x: hidden; /* Disable horizontal scrolling */
`;

const App = () => {
  return (
    <Router>
      <AppContainer>
        <Routes>
          {/* Define the route for the landing page */}
          <Route path="/" element={<LandingPage />} />
          <Route path="/" element={<LandingPage />} />
          
          {/* Add more routes here as needed */}
        </Routes>
      </AppContainer>
    </Router>
  );
};

export default App;
