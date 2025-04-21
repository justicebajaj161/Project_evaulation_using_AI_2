'use client'
import { useState, useEffect } from 'react';

export default function DynamicScoring() {
  const [rows, setRows] = useState([{ component: '', max_score: 0 }]);
  const [totalScore, setTotalScore] = useState(0);

  useEffect(() => {
    const sum = rows.reduce((acc, row) => acc + (Number(row.max_score) || 0), 0);
    setTotalScore(sum);
  }, [rows]);

  const addRow = () => {
    if (totalScore < 100) {
      setRows([...rows, { component: '', max_score: 0 }]);
    }
  };

  const removeRow = (index) => {
    if (rows.length > 1) {
      const newRows = [...rows];
      newRows.splice(index, 1);
      setRows(newRows);
    }
  };

  const handleChange = (index, field, value) => {
    const newRows = [...rows];
    newRows[index][field] = value;
    setRows(newRows);
  };

  useEffect(() => {
    localStorage.setItem('scoringPattern', JSON.stringify(rows));
  }, [rows]);

  return (
    <div className="mb-6">
      <h2 className="text-xl font-semibold mb-4">Scoring Pattern (Total: {totalScore}/100)</h2>
      
      {rows.map((row, index) => (
        <div key={index} className="flex gap-4 mb-3 items-end">
          <div className="flex-1">
            <label className="block mb-1">Component</label>
            <input
              type="text"
              value={row.component}
              onChange={(e) => handleChange(index, 'component', e.target.value)}
              className="w-full p-2 border rounded"
              required
            />
          </div>
          
          <div className="w-24">
            <label className="block mb-1">Max Score</label>
            <input
              type="number"
              value={row.max_score}
              onChange={(e) => handleChange(index, 'max_score', e.target.value)}
              className="w-full p-2 border rounded"
              min="0"
              max={100 - totalScore + (Number(row.max_score) || 0)}
              required
            />
          </div>
          
          <button
            type="button"
            onClick={() => removeRow(index)}
            className="bg-red-500 text-white px-3 py-2 rounded hover:bg-red-600"
            disabled={rows.length <= 1}
          >
            Remove
          </button>
        </div>
      ))}
      
      <button
        type="button"
        onClick={addRow}
        className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
        disabled={totalScore >= 100}
      >
        Add Component
      </button>
      
      {totalScore !== 100 && (
        <p className="text-red-500 mt-2">
          Total score must equal 100 (currently {totalScore})
        </p>
      )}
    </div>
  );
}