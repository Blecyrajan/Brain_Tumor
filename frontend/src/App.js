import React, { useState } from 'react';
import UploadImage from './components/UploadImage';
import PredictionResult from './components/PredictionResult';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [result, setResult] = useState(null);

  const handlePrediction = async (imageData) => {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: imageData,
    });
    const data = await response.json();
    setResult(data);
  };

  return (
    <div className="container">
      <h1 className="text-center mt-5">Brain Tumor Detection</h1>
      <UploadImage onPredict={handlePrediction} />
      {result && <PredictionResult result={result} />}
    </div>
  );
}

export default App;