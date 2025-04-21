'use client'
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { submitProject } from '@/lib/api';

export default function SubmitPage() {
  const [file, setFile] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    setIsSubmitting(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('zip_file', file);

      // Get criteria from localStorage
      const criteria = JSON.parse(localStorage.getItem('evaluationCriteria'));
      
      // Add criteria to formData with exact backend field names
      formData.append('project_about', criteria.project_about);
      formData.append('technology', criteria.technology);
      formData.append('problem_statement', criteria.problem_statement);
      formData.append('scoring_pattern', JSON.stringify(criteria.scoring_pattern));

      // Debug: Log formData entries
      for (let [key, value] of formData.entries()) {
        console.log(key, value);
      }

      const result = await submitProject(formData);
      localStorage.setItem('evaluationResult', JSON.stringify(result));
      router.push('/results');
    } catch (err) {
      setError(err.message || 'Submission failed');
      console.error('Submission error:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Submit Your Project</h1>
      
      <form onSubmit={handleSubmit} className="space-y-6 max-w-lg">
        <div>
          <label className="block mb-2">Upload Project ZIP File</label>
          <input
            type="file"
            accept=".zip"
            onChange={(e) => setFile(e.target.files[0])}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        
        {error && <p className="text-red-500">{error}</p>}
        
        <button
          type="submit"
          disabled={isSubmitting}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-blue-300"
        >
          {isSubmitting ? 'Submitting...' : 'Submit Project'}
        </button>
      </form>
    </div>
  );
}