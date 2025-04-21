export const submitProject = async (formData) => {
  try {
    const response = await fetch('http://localhost:8000/analyze-project', {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header - let browser set it automatically
      // with the correct boundary for multipart/form-data
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error in submitProject:', error);
    throw error;
  }
};