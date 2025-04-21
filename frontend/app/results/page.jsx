'use client'
import { useEffect, useState } from 'react';
import ResultCard from '@/components/ResultCard';

export default function ResultsPage() {
  const [result, setResult] = useState(null);
  const [criteria, setCriteria] = useState(null);

  useEffect(() => {
    const storedResult = localStorage.getItem('evaluationResult');
    const storedCriteria = localStorage.getItem('evaluationCriteria');
    
    if (storedResult) setResult(JSON.parse(storedResult));
    if (storedCriteria) setCriteria(JSON.parse(storedCriteria));
  }, []);

  if (!result || !criteria) {
    return (
      <div className="container mx-auto p-4">
        <p>No evaluation results found. Please submit a project first.</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Evaluation Results</h1>
      
      <div className="mb-8 p-4 bg-gray-100 rounded">
        <h2 className="text-xl font-semibold mb-2">Project Criteria</h2>
        <p><strong>About:</strong> {criteria.project_about}</p>
        <p><strong>Technology:</strong> {criteria.technology}</p>
        <p><strong>Problem Statement:</strong> {criteria.problem_statement}</p>
      </div>
      
      <ResultCard result={result} />
    </div>
  );
}