import { useState } from 'react';
import SystemStatusPanel from './components/SystemStatusPanel';
import ECGAnalysisPanel from './components/ECGAnalysisPanel';

function App() {
  const [aiEngineReady, setAiEngineReady] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/* Logo with ECG wave */}
              <div className="relative">
                <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                  </svg>
                </div>
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  DeepECGAnalyser <span className="text-sm font-semibold text-gray-500">- ECG Data Analysis</span>
                </h1>
                <p className="text-xs text-gray-500">AI-powered cardiac analysis with multiple diagnostic models</p>
              </div>
            </div>

          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-4">
        <div className="space-y-4">
            {/* Main Layout: Analysis Panel (large) + System Status (sidebar) */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              {/* Main Analysis Panel - Takes 3 columns */}
              <div className="lg:col-span-3">
                <ECGAnalysisPanel aiEngineReady={aiEngineReady} />
              </div>

              {/* System Status Sidebar - Takes 1 column */}
              <div className="lg:col-span-1">
                <SystemStatusPanel onEngineStatusChange={setAiEngineReady} />
              </div>
            </div>

          </div>
      </main>

      {/* Footer */}
      <footer className="bg-white/80 border-t border-gray-200 mt-8">
        <div className="max-w-7xl mx-auto px-6 py-5">
          {/* About & Citation */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
            {/* Description */}
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2">About DeepECGAnalyser</h4>
              <p className="text-xs text-gray-500 leading-relaxed">
                This application uses the <span className="font-semibold text-gray-700">HeartWise AI engine</span> for
                ECG interpretation. HeartWise provides foundation models for generalizable electrocardiogram analysis,
                comparing supervised and self-supervised approaches (EfficientNet, WCR) across 77 diagnostic classes.
                All processing is performed locally via Docker with GPU acceleration.
              </p>
            </div>

            {/* Citation inline in About text */}
            <div className="flex items-end">
              <p className="text-xs text-gray-400 leading-relaxed">
                Nolin-Lapalme, A., Sowa, A., Delfrate, J., et al. (2025). <em>medRxiv</em>.{' '}
                <a href="https://doi.org/10.1101/2025.03.02.25322575"
                   target="_blank" rel="noopener noreferrer"
                   className="text-blue-500 hover:text-blue-600">
                  DOI
                </a>
              </p>
            </div>
          </div>

          {/* Bottom bar */}
          <div className="border-t border-gray-100 pt-3 flex items-center justify-between text-xs text-gray-400">
            <p>DeepECGAnalyser - Benoit LEQUEUX (Cercle IA SFC)</p>
            <p className="flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full"></span>
              100% Local Processing
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
