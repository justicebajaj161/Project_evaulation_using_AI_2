export default function ResultCard({ result }) {
    if (!result) return null;
  
    return (
      <div className="space-y-6">
        <div className="p-6 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Overall Evaluation</h2>
          
          <div className="flex items-center mb-4">
            <div className="text-4xl font-bold mr-4">
              {result.analysis.score}/100
            </div>
            <div>
              <div className="w-full bg-gray-200 rounded-full h-4">
                <div
                  className="bg-blue-600 h-4 rounded-full"
                  style={{ width: `${result.analysis.score}%` }}
                ></div>
              </div>
              <p className="mt-1 text-sm text-gray-600">
                {result.analysis.overall_feedback}
              </p>
            </div>
          </div>
        </div>
  
        <div className="p-6 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Component Breakdown</h2>
          
          {result.analysis.component_evaluations?.map((component, index) => (
            <div key={index} className="mb-4 pb-4 border-b last:border-b-0">
              <div className="flex justify-between items-center mb-2">
                <h3 className="font-medium">{component.component}</h3>
                <span className="font-bold">
                  {component.score}/
                  {result.evaluation_criteria.scoring_pattern.find(
                    (c) => c.component === component.component
                  )?.max_score || '?'}
                </span>
              </div>
              <p className="text-sm text-gray-600 mb-1">
                <strong>Feedback:</strong> {component.feedback}
              </p>
              {component.suggestions && (
                <p className="text-sm text-gray-600">
                  <strong>Suggestions:</strong> {component.suggestions}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  }