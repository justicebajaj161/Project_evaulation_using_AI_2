export const submitProject = async (formData) => {
    const response = await fetch('http://localhost:8000/analyze-project', {
      method: 'POST',
      body: formData,
    });
  
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Submission failed');
    }
  
    return await response.json();
  };