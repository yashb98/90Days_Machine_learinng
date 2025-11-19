import React, { useState } from 'react';
import { Minus, ThumbsUp, Loader, Zap, Cpu, CheckCircle, Pencil, Save, X } from 'lucide-react';

// --- Configuration ---
// NOTE: In a real environment, this should point to the AWS ALB/CloudFront URL of the API Gateway.
const API_URL = '/api/rag_query'; 
const SUBMIT_URL = '/api/submit_correction';

// Define the shape of the data returned by the backend
interface RAGResult {
  mode: 'A' | 'B';
  answer: string;
  context: string;
  model_name: string;
  latency_ms: string; // Added latency
  raw_context: string 
  safety_guardrail?: {
    status: string;
    hallucinated_concepts: string[];
  };
}

// Define the shape of the feedback object (Phase 5)
interface Feedback {
  query: string;
  mode: 'A' | 'B';
  is_positive: boolean;
  timestamp: number;
}

// Component to handle modal messaging (replacing alert())
const Modal: React.FC<{ title: string; message: string; onClose: () => void }> = ({ title, message, onClose }) => (
  <div className="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 p-4">
    <div className="bg-white p-6 rounded-xl shadow-2xl max-w-sm w-full">
      <h3 className="text-xl font-bold text-gray-800 mb-3">{title}</h3>
      <p className="text-gray-700 mb-4 whitespace-pre-wrap">{message}</p>
      <button onClick={onClose} 
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
        OK
      </button>
    </div>
  </div>
);

const ABtest: React.FC = () => {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'A' | 'B'>('A');
  const [result, setResult] = useState<RAGResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [modal, setModal] = useState<{ title: string; message: string } | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedAnswer, setEditedAnswer] = useState("");


  
  // --- FIX ---
  // Renamed 'feedbackLog' to '_feedbackLog' to satisfy the TypeScript
  // "unused variable" warning, as we are only writing to it.
  const [_feedbackLog, setFeedbackLog] = useState<Feedback[]>([]);

  // Determine card styling based on the active mode
  const getCardStyle = (mode: 'A' | 'B') => {
    if (mode === 'A') {
      return "border-green-500 bg-green-50 text-green-800";
    }
    return "border-purple-500 bg-purple-50 text-purple-800";
  };

  const showMessage = (title: string, message: string) => {
    setModal({ title, message });
  };

  const handleQuery = async () => {
    if (!query.trim()) {
      showMessage("Input Required", "Please enter a query in the text box before submitting.");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      // API call to the Python backend (API Gateway)
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), mode })
      });

      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }

      const data: RAGResult = await response.json();
      setResult(data);

    } catch (error) {
      console.error("Fetch Error:", error);
      showMessage("API Connection Error", `Could not connect to the backend server or process response. Details: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const handelSubmitCorrection = async() => {
    if (!result) return;
    try{
      const payload= {
        query: query,
        context: result.raw_context,
        original_answer: result.answer,
        corrected_answer: editedAnswer,
        model_name: result.model_name,
        mode: result.mode,
      };

      const response = await fetch(SUBMIT_URL, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        showMessage("Success", "Correctoon Saved to Dataset!")
        setIsEditing(false);
        setResult({...result, answer: editedAnswer});
      }
      else {
        showMessage("Error", "Failed to save Correction.");
      }
    }
    catch(error){
      console.error("Error submitting correction: ", error)
    }
  };
  const recordFeedback = (isPositive: boolean) => {
    if (!result) return;
    
    const newFeedback: Feedback = {
      query: query,
      mode: result.mode,
      is_positive: isPositive,
      timestamp: Date.now()
    };

    // Phase 5: Log feedback data (simulated Firestore/MLflow logging)
    setFeedbackLog(prev => [...prev, newFeedback]);
    
    let status = isPositive ? "Positive (Thumbs Up)" : "Negative (Inaccurate/Hallucinated)";
    showMessage("Feedback Recorded", 
                `Feedback for **${result.model_name}** recorded.\nStatus: ${status}\n\nThis data is logged for model evaluation (Phase 4) and continual learning (Phase 5).`);
  };

  return (
    <div className="min-h-screen p-4 md:p-8 flex items-center justify-center bg-gray-50">
      {modal && <Modal title={modal.title} message={modal.message} onClose={() => setModal(null)} />}

      <div className="w-full max-w-4xl bg-white shadow-xl rounded-2xl p-6 md:p-8">

        <header className="mb-6 border-b pb-4">
          <h1 className="text-3xl font-bold text-gray-800">RAG Production Arena</h1>
          <p className="text-sm text-gray-500 mt-1">A/B Test interface for MLOps validation</p>
          
          {/* A/B Selector */}
          <div className="mt-4 flex flex-col md:flex-row items-stretch md:items-center space-y-2 md:space-y-0 md:space-x-4 p-3 bg-indigo-50 rounded-xl border border-indigo-200">
            <label htmlFor="ab_test_mode" className="text-sm font-semibold text-indigo-700 whitespace-nowrap flex-shrink-0">
                <Zap className="inline w-4 h-4 mr-1"/> Active Pipeline:
            </label>
            <select id="ab_test_mode" value={mode} onChange={(e) => setMode(e.target.value as 'A' | 'B')}
                    className="flex-grow p-2 border border-indigo-300 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-sm bg-white">
                <option value="A">Mode A (Mistral-7B - Baseline/Stable)</option>
                <option value="B">Mode B (Gemini 2.5 Pro - Reasoning/Structure)</option>
            </select>
          </div>
        </header>

        {/* Query Input */}
        <div className="mb-6">
          <label htmlFor="query_input" className="block text-lg font-medium text-gray-700 mb-2">Query the EHR Data:</label>
          <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3">
            <input type="text" id="query_input" value={query} onChange={(e) => setQuery(e.target.value)}
                   placeholder="e.g., What are the risk factors for this patient?" 
                   className="flex-grow p-3 border border-gray-300 rounded-xl shadow-inner focus:ring-blue-600 focus:border-blue-600 text-gray-800 transition duration-150"
                   onKeyDown={(e) => { if (e.key === 'Enter') handleQuery(); }} disabled={loading}
            />
            <button onClick={handleQuery} disabled={loading}
                    className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-xl shadow-md hover:bg-blue-700 transition duration-150 transform hover:scale-[1.02] disabled:bg-gray-400 disabled:cursor-not-allowed">
              {loading ? <Loader className="animate-spin w-5 h-5 mx-auto"/> : 'Get RAG Answer'}
            </button>
          </div>
        </div>

        {/* Response Area */}
        <div className="space-y-6">
          
          {/* Loading Indicator */}
          {loading && (
            <div className="text-center p-4 bg-gray-100 rounded-xl text-gray-600">
              <Loader className="animate-spin inline-block w-6 h-6 border-4 border-t-blue-600 border-gray-200 rounded-full text-blue-600"/>
              <p className="mt-2">Retrieving context and generating answer for <span className="font-semibold">{mode === 'A' ? 'Mistral-7B' : 'Gemini 2.5 Pro'}</span>...</p>
            </div>
          )}

          {/* RAG Answer Card */}
          {result && (
            <div className={`border-l-4 ${getCardStyle(result.mode).split(' ')[0]} ${getCardStyle(result.mode).split(' ')[1]} ${getCardStyle(result.mode).split(' ')[2]} p-4 rounded-xl shadow-lg transition duration-300`}>
                
                {/* Header with Edit Toggle */}
                <div className="flex justify-between items-start mb-2">
                  <h3 className={`text-xl font-bold flex items-center ${getCardStyle(result.mode).split(' ')[2]}`}>
                      <CheckCircle className="w-6 h-6 mr-2"/>
                      RAG Final Answer (<span id="active_mode" className="ml-1 font-mono text-base">{result.model_name}</span>)
                  </h3>
                  
                  {/* Only show Edit button if not currently editing */}
                  {!isEditing && (
                    <button 
                      onClick={() => { 
                        setIsEditing(true); 
                        // Pre-fill the editor with the current answer
                        setEditedAnswer(result.answer); 
                      }}
                      className="flex items-center text-xs md:text-sm bg-white border border-gray-300 text-gray-600 px-3 py-1 rounded-lg hover:bg-gray-50 transition shadow-sm"
                    >
                      <Pencil className="w-3 h-3 md:w-4 md:h-4 mr-1"/> Edit Answer
                    </button>
                  )}
                </div>

                {/* Conditional Rendering: Edit Mode vs Read Mode */}
                {isEditing ? (
                  <div className="mt-2 animate-in fade-in duration-200">
                    <textarea
                      value={editedAnswer}
                      onChange={(e) => setEditedAnswer(e.target.value)}
                      className="w-full p-3 border border-blue-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-800 font-mono text-sm bg-white shadow-inner"
                      rows={8}
                      placeholder="Correct the answer here..."
                    />
                    <div className="flex space-x-3 mt-3 justify-end">
                       <button 
                        onClick={() => setIsEditing(false)}
                        className="flex items-center px-4 py-2 text-gray-600 bg-gray-200 rounded-lg hover:bg-gray-300 transition"
                      >
                        <X className="w-4 h-4 mr-1"/> Cancel
                      </button>
                      <button 
                        onClick={handelSubmitCorrection}
                        className="flex items-center px-4 py-2 text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition shadow-md"
                      >
                        <Save className="w-4 h-4 mr-1"/> Save to Dataset
                      </button>
                    </div>
                  </div>
                ) : (
                  // Normal Read Mode (renders HTML)
                  <div id="rag_answer" className="text-gray-700 text-base" dangerouslySetInnerHTML={{ __html: result.answer }} />
                )}
                
                {/* Display Guardrail Warning if it exists (From Day 43) */}
                {result.safety_guardrail?.status === 'FLAGGED' && !isEditing && (
                   <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-800 text-sm flex items-start">
                     <span className="mr-2">⚠️</span>
                     <div>
                       <strong>Potential Hallucination Detected:</strong> The model mentioned concepts not found in the source text: 
                       <span className="font-mono ml-1">{result.safety_guardrail.hallucinated_concepts.join(", ")}</span>
                     </div>
                   </div>
                )}

                {/* Feedback Loop (Phase 5) */}
                <div className="mt-4 flex items-center space-x-4 text-sm text-gray-500 border-t pt-3 mt-3">
                    <span className="font-medium text-gray-700">Was this response accurate and helpful?</span>
                    <button title="Accurate and Helpful" className="flex items-center text-green-500 hover:text-green-700 transition p-2 bg-white rounded-full shadow hover:shadow-lg" onClick={() => recordFeedback(true)}>
                        <ThumbsUp className="w-5 h-5"/>
                    </button>
                    <button title="Inaccurate or Hallucinated" className="flex items-center text-red-500 hover:text-red-700 transition p-2 bg-white rounded-full shadow hover:shadow-lg" onClick={() => recordFeedback(false)}>
                        <Minus className="w-5 h-5"/>
                    </button>
                </div>
            </div>
          )}

          {/* Retrieved Context Card */}
          {result && (
            <div className="p-4 bg-gray-100 rounded-xl shadow-inner border border-gray-200">
                <h3 className="text-lg font-bold text-gray-700 mb-2 flex items-center">
                    <Cpu className="w-5 h-5 mr-2"/>
                    Retrieved Context (Source Passages)
                </h3>
                <div id="retrieved_context" className="scrollable-content text-sm text-gray-600 whitespace-pre-wrap p-2 border-t border-gray-300">
                    {result.context}
                </div>
            </div>
          )}
          
        </div>
      </div>
    </div>
  );
};

export default ABtest;