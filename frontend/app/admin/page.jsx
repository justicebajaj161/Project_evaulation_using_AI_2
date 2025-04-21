'use client'
import { useState, useEffect } from 'react';
import DynamicScoring from '@/components/DynamicScoring';
import { useRouter } from 'next/navigation';

export default function AdminPage() {
  const [projectAbout, setProjectAbout] = useState('');
  const [technology, setTechnology] = useState('');
  const [problemStatement, setProblemStatement] = useState('');
  const router = useRouter();

  const handleSubmit = (e) => {
    e.preventDefault();
    // Store criteria in localStorage or context
    const evaluationCriteria = {
      projectAbout,
      technology,
      problemStatement,
      scoringPattern: JSON.parse(localStorage.getItem('scoringPattern'))
    };
    console.log(evaluationCriteria)
    localStorage.setItem('evaluationCriteria', JSON.stringify(evaluationCriteria));
    router.push('/submit');
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Set Evaluation Criteria</h1>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="block mb-2">Project About</label>
          <input
            type="text"
            value={projectAbout}
            onChange={(e) => setProjectAbout(e.target.value)}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        
        <div>
          <label className="block mb-2">Technology</label>
          <input
            type="text"
            value={technology}
            onChange={(e) => setTechnology(e.target.value)}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        
        <div>
          <label className="block mb-2">Problem Statement</label>
          <textarea
            value={problemStatement}
            onChange={(e) => setProblemStatement(e.target.value)}
            className="w-full p-2 border rounded"
            rows={4}
            required
          />
        </div>
        
        <DynamicScoring />
        
        <button
          type="submit"
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Set Criteria
        </button>
      </form>
    </div>
  );
}