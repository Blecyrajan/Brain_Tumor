import React from 'react';

const PredictionResult = ({ result }) => (
  <div className="mt-4">
    <h4>Prediction: {result.prediction}</h4>
    <p>Confidence: {result.confidence.toFixed(2)}%</p>

    {result.reason && (
      <p><strong>Reason:</strong> {result.reason}</p>
    )}

    <h5>Explanation:</h5>
    <img
      src={`data:image/png;base64,${result.explanation}`}
      alt="Explanation"
      className="img-fluid"
    />
  </div>
);

export default PredictionResult;
