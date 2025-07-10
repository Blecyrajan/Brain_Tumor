import React, { useState } from 'react';

const UploadImage = ({ onPredict }) => {
  const [image, setImage] = useState(null);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', image);
    onPredict(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="mt-3">
      <input type="file" className="form-control" onChange={handleImageChange} required />
      <button type="submit" className="btn btn-primary mt-2">Upload & Predict</button>
    </form>
  );
};

export default UploadImage;