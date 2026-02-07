import React, { useState, useRef, useEffect } from 'react';
import {
  getAvailableECGModels,
  runFullECGAnalysis,
  runBatchECGAnalysis,
  BatchECGAnalysisResponse,
} from '../api';
import ModelSelector, { ECGModel } from './ModelSelector';
import ECGVisualResults, { FullECGAnalysisResult } from './ECGVisualResults';
import ECGViewer from './ECGViewer';

interface ECGAnalysisPanelProps {
  aiEngineReady: boolean;
}

const ECGAnalysisPanel: React.FC<ECGAnalysisPanelProps> = ({ aiEngineReady }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [batchMode, setBatchMode] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [isLoadingModels, setIsLoadingModels] = useState(true);

  // Available models for full analysis
  const [availableModels, setAvailableModels] = useState<ECGModel[]>([]);
  const [selectedFullModels, setSelectedFullModels] = useState<string[]>([]);

  // Results
  const [fullAnalysisResult, setFullAnalysisResult] = useState<FullECGAnalysisResult | null>(null);

  // Batch results
  const [batchResults, setBatchResults] = useState<BatchECGAnalysisResponse | null>(null);
  const [currentBatchIndex, setCurrentBatchIndex] = useState(0);

  // Pipeline options
  const [useGpu, setUseGpu] = useState(true);

  // ECG Viewer
  const [showECGViewer, setShowECGViewer] = useState(false);

  const ALLOWED_EXTENSIONS = ['.csv', '.parquet', '.xml', '.npy'];

  // Load models on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        // Load available models for full analysis
        const availableData = await getAvailableECGModels();
        const models: ECGModel[] = availableData.models.map(m => ({
          id: m.id,
          name: m.name,
          architecture: m.architecture,
          type: m.type,
          description: m.description,
        }));
        setAvailableModels(models);
        setSelectedFullModels(availableData.default_selection);
      } catch (err) {
        console.error('Failed to load models:', err);
        setError('Échec du chargement des modèles');
      } finally {
        setIsLoadingModels(false);
      }
    };
    loadModels();
  }, []);

  // Auto-clear errors
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 10000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  const validateFile = (file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError(`Type de fichier invalide. Autorisés: ${ALLOWED_EXTENSIONS.join(', ')}`);
      return false;
    }
    return true;
  };

  const handleFileSelect = (file: File) => {
    if (validateFile(file)) {
      setSelectedFile(file);
      setSelectedFiles([file]);
      setError(null);
      clearAllResults();
    }
  };

  const handleMultipleFilesSelect = (files: File[]) => {
    const validFiles = files.filter(validateFile);
    if (validFiles.length > 0) {
      setSelectedFiles(validFiles);
      setSelectedFile(validFiles[0]);
      setError(null);
      clearAllResults();
    }
  };

  const clearAllResults = () => {
    setFullAnalysisResult(null);
    setBatchResults(null);
    setCurrentBatchIndex(0);
  };

  // Export batch results to CSV for research
  const exportBatchResultsToCSV = () => {
    if (!batchResults || batchResults.results.length === 0) return;

    // Collect all unique diagnosis names (from multi_label), multi-label model archs, and binary model IDs
    const allDiagnoses = new Set<string>();
    const allMultiLabelArchs = new Set<string>(); // architecture names (EFFICIENTNET, WCR)
    const allBinaryModels = new Map<string, string>(); // id -> name

    batchResults.results.forEach(r => {
      if (r.success && r.result) {
        Object.values(r.result.results).forEach(modelResult => {
          // Multi-label models (77 classes) have many diagnoses (>10)
          if (modelResult.model_type === 'multi_label' || modelResult.diagnoses.length > 10) {
            modelResult.diagnoses.forEach(d => allDiagnoses.add(d.name));
            allMultiLabelArchs.add(modelResult.architecture.toUpperCase());
          } else if (modelResult.model_type === 'binary') {
            // Binary models - use model_id as identifier
            allBinaryModels.set(modelResult.model_id, modelResult.model_name);
          }
        });
      }
    });

    const diagnosisNames = Array.from(allDiagnoses).sort();
    const multiLabelArchs = Array.from(allMultiLabelArchs).sort();
    const binaryModelIds = Array.from(allBinaryModels.keys()).sort();
    const hasMultipleMultiLabel = multiLabelArchs.length > 1;

    // Build CSV header
    const headers = ['filename'];

    // Add columns for each diagnosis - include model arch if multiple multi-label models
    diagnosisNames.forEach(name => {
      if (hasMultipleMultiLabel) {
        // Add columns for each architecture
        multiLabelArchs.forEach(arch => {
          headers.push(`${name}_${arch}_prob`);
          headers.push(`${name}_${arch}_status`);
        });
        // Add difference column
        headers.push(`${name}_diff`);
      } else {
        headers.push(`${name}_prob`);
        headers.push(`${name}_status`);
      }
    });

    // Add columns for binary models using their model_id
    binaryModelIds.forEach(id => {
      headers.push(`${id}_prob`);
      headers.push(`${id}_status`);
    });

    // Build CSV rows
    const rows: string[][] = [];

    batchResults.results.forEach(r => {
      if (!r.success || !r.result) return;

      const row: string[] = [r.filename];

      // Create maps for quick lookup - keyed by "diagnosis_arch" for multi-label
      const diagnosisByArch = new Map<string, Map<string, { prob: number; status: string }>>();
      const binaryMap = new Map<string, { prob: number; status: string }>();

      Object.values(r.result.results).forEach(modelResult => {
        if (modelResult.model_type === 'multi_label' || modelResult.diagnoses.length > 10) {
          const arch = modelResult.architecture.toUpperCase();
          if (!diagnosisByArch.has(arch)) {
            diagnosisByArch.set(arch, new Map());
          }
          modelResult.diagnoses.forEach(d => {
            diagnosisByArch.get(arch)!.set(d.name, { prob: d.probability, status: d.status });
          });
        } else if (modelResult.model_type === 'binary') {
          // Binary model - get the first diagnosis which is the main result
          if (modelResult.diagnoses.length > 0) {
            const d = modelResult.diagnoses[0];
            binaryMap.set(modelResult.model_id, { prob: d.probability, status: d.status });
          }
        }
      });

      // Add diagnosis values
      diagnosisNames.forEach(name => {
        if (hasMultipleMultiLabel) {
          const probs: number[] = [];
          multiLabelArchs.forEach(arch => {
            const archMap = diagnosisByArch.get(arch);
            const result = archMap?.get(name);
            row.push(result ? result.prob.toFixed(2) : '');
            row.push(result ? result.status : '');
            if (result) probs.push(result.prob);
          });
          // Calculate difference between models
          if (probs.length >= 2) {
            const diff = Math.abs(probs[0] - probs[1]);
            row.push(diff.toFixed(2));
          } else {
            row.push('');
          }
        } else {
          // Single model - get from first architecture
          const archMap = diagnosisByArch.get(multiLabelArchs[0]);
          const result = archMap?.get(name);
          row.push(result ? result.prob.toFixed(2) : '');
          row.push(result ? result.status : '');
        }
      });

      // Add binary model values
      binaryModelIds.forEach(id => {
        const result = binaryMap.get(id);
        row.push(result ? result.prob.toFixed(2) : '');
        row.push(result ? result.status : '');
      });

      rows.push(row);
    });

    // Generate CSV content
    const csvContent = [
      headers.join(';'),
      ...rows.map(row => row.join(';'))
    ].join('\n');

    // Download CSV
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `ecg_analysis_results_${new Date().toISOString().slice(0, 10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export single analysis result to CSV for research
  const exportSingleResultToCSV = () => {
    if (!fullAnalysisResult || !selectedFile) return;

    // Collect multi-label models and binary models separately
    const multiLabelModels: { id: string; arch: string; diagnoses: Map<string, { prob: number; status: string }> }[] = [];
    const binaryModels: { id: string; name: string }[] = [];
    const allDiagnosisNames = new Set<string>();

    Object.entries(fullAnalysisResult.results).forEach(([modelId, modelResult]) => {
      // Multi-label models (77 classes) have many diagnoses (>10)
      if (modelResult.model_type === 'multi_label' || modelResult.diagnoses.length > 10) {
        const diagMap = new Map<string, { prob: number; status: string }>();
        modelResult.diagnoses.forEach(d => {
          diagMap.set(d.name, { prob: d.probability, status: d.status });
          allDiagnosisNames.add(d.name);
        });
        multiLabelModels.push({
          id: modelId,
          arch: modelResult.architecture.toUpperCase(),
          diagnoses: diagMap
        });
      } else if (modelResult.model_type === 'binary') {
        // Binary models - use model_id as identifier (lvef_40, lvef_50, afib_5y, etc.)
        binaryModels.push({ id: modelResult.model_id, name: modelResult.model_name });
      }
    });

    const diagnosisNames = Array.from(allDiagnosisNames).sort();
    binaryModels.sort((a, b) => a.id.localeCompare(b.id));
    const hasMultipleMultiLabel = multiLabelModels.length > 1;

    // Build CSV header
    const headers = ['filename'];

    // Add columns for each diagnosis - include model arch if multiple multi-label models
    diagnosisNames.forEach(name => {
      if (hasMultipleMultiLabel) {
        // Add columns for each model
        multiLabelModels.forEach(model => {
          headers.push(`${name}_${model.arch}_prob`);
          headers.push(`${name}_${model.arch}_status`);
        });
        // Add difference column
        headers.push(`${name}_diff`);
      } else {
        headers.push(`${name}_prob`);
        headers.push(`${name}_status`);
      }
    });

    // Add columns for binary models using their model_id
    binaryModels.forEach(model => {
      headers.push(`${model.id}_prob`);
      headers.push(`${model.id}_status`);
    });

    // Build row
    const row: string[] = [selectedFile.name];

    // Create binary map for quick lookup
    const binaryMap = new Map<string, { prob: number; status: string }>();
    Object.values(fullAnalysisResult.results).forEach(modelResult => {
      if (modelResult.model_type === 'binary') {
        if (modelResult.diagnoses.length > 0) {
          const d = modelResult.diagnoses[0];
          binaryMap.set(modelResult.model_id, { prob: d.probability, status: d.status });
        }
      }
    });

    // Add diagnosis values
    diagnosisNames.forEach(name => {
      if (hasMultipleMultiLabel) {
        const probs: number[] = [];
        multiLabelModels.forEach(model => {
          const result = model.diagnoses.get(name);
          row.push(result ? result.prob.toFixed(2) : '');
          row.push(result ? result.status : '');
          if (result) probs.push(result.prob);
        });
        // Calculate difference between models
        if (probs.length >= 2) {
          const diff = Math.abs(probs[0] - probs[1]);
          row.push(diff.toFixed(2));
        } else {
          row.push('');
        }
      } else {
        const result = multiLabelModels[0]?.diagnoses.get(name);
        row.push(result ? result.prob.toFixed(2) : '');
        row.push(result ? result.status : '');
      }
    });

    // Add binary model values
    binaryModels.forEach(model => {
      const result = binaryMap.get(model.id);
      row.push(result ? result.prob.toFixed(2) : '');
      row.push(result ? result.status : '');
    });

    // Generate CSV content
    const csvContent = [
      headers.join(';'),
      row.join(';')
    ].join('\n');

    // Download CSV
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `ecg_analysis_${selectedFile.name.replace(/\.[^/.]+$/, '')}_${new Date().toISOString().slice(0, 10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (batchMode && files.length > 1) {
      handleMultipleFilesSelect(files);
    } else if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      if (batchMode && files.length > 1) {
        handleMultipleFilesSelect(Array.from(files));
      } else {
        handleFileSelect(files[0]);
      }
    }
  };

  const handleAnalyze = async () => {
    // Batch mode
    if (batchMode && selectedFiles.length > 0) {
      setIsAnalyzing(true);
      setError(null);
      clearAllResults();

      try {
        const modelsToRun = selectedFullModels.length > 0 ? selectedFullModels : ['all'];
        const response = await runBatchECGAnalysis(
          selectedFiles,
          modelsToRun,
          useGpu
        );
        setBatchResults(response);
        setCurrentBatchIndex(0);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Échec de l'analyse par lot");
      } finally {
        setIsAnalyzing(false);
      }
      return;
    }

    // Single file mode
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    clearAllResults();

    try {
      const modelsToRun = selectedFullModels.length > 0 ? selectedFullModels : ['all'];
      const response = await runFullECGAnalysis(
        selectedFile,
        modelsToRun,
        useGpu
      );
      setFullAnalysisResult(response as FullECGAnalysisResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Échec de l'analyse");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setSelectedFiles([]);
    clearAllResults();
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Navigation for batch results
  const goToPreviousECG = () => {
    if (currentBatchIndex > 0) {
      setCurrentBatchIndex(currentBatchIndex - 1);
    }
  };

  const goToNextECG = () => {
    if (batchResults && currentBatchIndex < batchResults.results.length - 1) {
      setCurrentBatchIndex(currentBatchIndex + 1);
    }
  };

  const goToECGIndex = (index: number) => {
    if (batchResults && index >= 0 && index < batchResults.results.length) {
      setCurrentBatchIndex(index);
    }
  };

  // Get current batch result for display
  const currentBatchResult = batchResults?.results[currentBatchIndex];

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const hasResult = fullAnalysisResult || batchResults;
  const isXmlFile = selectedFile && ['.xml', '.npy'].includes('.' + selectedFile.name.split('.').pop()?.toLowerCase());

  return (
    <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-red-500 to-pink-600 p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">Analyse ECG Complète</h2>
            <p className="text-red-100 text-sm mt-1">
              Diagnostic cardiaque par IA avec affichage visuel
            </p>
          </div>
          {aiEngineReady && (
            <div className="flex items-center gap-2 bg-white/20 px-3 py-1.5 rounded-full">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-sm font-medium">Moteur IA Prêt</span>
            </div>
          )}
        </div>
      </div>

      <div className="p-6">
        {/* Not Ready Warning */}
        {!aiEngineReady && !hasResult && (
          <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-xl">
            <div className="flex items-center gap-3">
              <svg className="w-6 h-6 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <div>
                <p className="font-medium text-yellow-800">Moteur IA non prêt</p>
                <p className="text-sm text-yellow-600">Démarrez le conteneur Docker depuis la barre latérale</p>
              </div>
            </div>
          </div>
        )}

        {/* Description */}
        {!hasResult && (
          <p className="text-center text-sm text-gray-500 mb-6">
            Analyse complète avec tous les modèles et affichage visuel des résultats
          </p>
        )}

        {/* Model Selection for Full Analysis */}
        {!hasResult && !isLoadingModels && (
          <div className="mb-6">
            <ModelSelector
              availableModels={availableModels}
              selectedModels={selectedFullModels}
              onSelectionChange={setSelectedFullModels}
              disabled={!aiEngineReady}
            />

            {/* Options Row */}
            <div className="mt-4 grid grid-cols-2 gap-4">
              {/* GPU Option */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                <div className="flex-1 mr-4">
                  <span className="font-medium text-gray-800">GPU (CUDA)</span>
                  <p className="text-xs text-gray-500">Accélération</p>
                </div>
                <button
                  onClick={() => setUseGpu(!useGpu)}
                  className={`relative flex-shrink-0 w-12 h-6 rounded-full transition-colors ${
                    useGpu ? 'bg-green-500' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`absolute left-0.5 top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                      useGpu ? 'translate-x-6' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>

              {/* Batch Mode Option */}
              <div className="flex items-center justify-between p-4 bg-purple-50 rounded-xl border border-purple-200">
                <div className="flex-1 mr-4">
                  <span className="font-medium text-purple-800">Mode Lot</span>
                  <p className="text-xs text-purple-600">Multi-fichiers</p>
                </div>
                <button
                  onClick={() => setBatchMode(!batchMode)}
                  className={`relative flex-shrink-0 w-12 h-6 rounded-full transition-colors ${
                    batchMode ? 'bg-purple-500' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`absolute left-0.5 top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                      batchMode ? 'translate-x-6' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        )}


        {/* Loading Models */}
        {isLoadingModels && (
          <div className="flex items-center justify-center py-12">
            <svg className="animate-spin h-8 w-8 text-red-500" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span className="ml-3 text-gray-600">Chargement des modèles...</span>
          </div>
        )}

        {/* File Upload Area */}
        {selectedFiles.length === 0 && !hasResult && !isLoadingModels && (
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`
              border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer
              transition-all duration-200
              ${!aiEngineReady ? 'opacity-50 pointer-events-none' : ''}
              ${isDragging
                ? batchMode ? 'border-purple-500 bg-purple-50' : 'border-red-500 bg-red-50'
                : 'border-gray-300 hover:border-red-400 hover:bg-gray-50'
              }
            `}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.parquet,.xml,.npy"
              multiple={batchMode}
              onChange={handleInputChange}
              className="hidden"
            />
            <div className={`w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center ${batchMode ? 'bg-purple-100' : 'bg-red-100'}`}>
              <svg className={`w-8 h-8 ${batchMode ? 'text-purple-500' : 'text-red-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {batchMode ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                )}
              </svg>
            </div>
            <p className="text-gray-700 font-semibold text-lg">
              {batchMode ? 'Déposez vos fichiers ECG ici' : 'Déposez votre fichier ECG ici'}
            </p>
            <p className="text-gray-500 text-sm mt-1">ou cliquez pour parcourir</p>
            <p className="text-xs text-gray-400 mt-3">
              {batchMode ? 'Sélectionnez plusieurs fichiers XML ou NPY' : 'Formats: XML, CSV, Parquet, NPY (max 100MB)'}
            </p>
            {batchMode && (
              <div className="mt-3 inline-flex items-center gap-1 px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                Mode traitement par lot actif
              </div>
            )}
          </div>
        )}

        {/* Selected Files + Analyze Button */}
        {selectedFiles.length > 0 && !hasResult && (
          <div className="space-y-4">
            {/* Single file display */}
            {!batchMode && selectedFile && (
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center">
                    <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <div>
                    <p className="font-semibold text-gray-800">{selectedFile.name}</p>
                    <p className="text-sm text-gray-500">{formatFileSize(selectedFile.size)}</p>
                  </div>
                </div>
                <button
                  onClick={handleReset}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            )}

            {/* Batch files display */}
            {batchMode && selectedFiles.length > 0 && (
              <div className="p-4 bg-purple-50 border border-purple-200 rounded-xl">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                      <span className="text-purple-700 font-bold">{selectedFiles.length}</span>
                    </div>
                    <div>
                      <p className="font-semibold text-purple-800">Fichiers sélectionnés</p>
                      <p className="text-xs text-purple-600">Traitement par lot</p>
                    </div>
                  </div>
                  <button
                    onClick={handleReset}
                    className="p-2 text-purple-400 hover:text-purple-600 hover:bg-purple-100 rounded-lg transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <div className="max-h-32 overflow-y-auto space-y-1">
                  {selectedFiles.map((file, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-sm text-purple-700 bg-white/50 px-2 py-1 rounded">
                      <span className="w-5 h-5 flex items-center justify-center bg-purple-200 rounded text-xs font-medium">
                        {idx + 1}
                      </span>
                      <span className="truncate">{file.name}</span>
                      <span className="text-purple-500 text-xs">({formatFileSize(file.size)})</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Info for XML files + supported formats */}
            {isXmlFile && (
              <div className="p-3 bg-blue-50 border border-blue-200 rounded-xl">
                <div className="flex items-center gap-2 text-blue-700">
                  <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-sm font-medium">
                    Fichier XML - Conversion automatique vers GE MUSE si nécessaire
                  </span>
                </div>
                <div className="mt-2 ml-7">
                  <table className="text-xs text-gray-600 w-full">
                    <thead>
                      <tr className="text-left text-gray-500">
                        <th className="pr-3 pb-1 font-medium">Format</th>
                        <th className="pr-3 pb-1 font-medium">Constructeur</th>
                        <th className="pb-1 font-medium">Conversion</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr><td className="pr-3 py-0.5 font-semibold text-green-700">GE MUSE RestingECG</td><td className="pr-3">GE Healthcare</td><td><span className="text-green-600">Natif</span></td></tr>
                      <tr><td className="pr-3 py-0.5">Philips PageWriter / TC</td><td className="pr-3">Philips</td><td>Repbeats + resampling</td></tr>
                      <tr><td className="pr-3 py-0.5">HL7 aECG (Annotated ECG)</td><td className="pr-3">Standard FDA</td><td>Digits + scale uV</td></tr>
                      <tr><td className="pr-3 py-0.5">CardiologyXML</td><td className="pr-3">Générique</td><td>Base64 int16</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Analysis summary */}
            <div className="p-3 bg-purple-50 border border-purple-200 rounded-xl">
              <div className="flex items-center gap-2 text-purple-700">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                </svg>
                <span className="text-sm font-medium">
                  {selectedFullModels.length} modèle{selectedFullModels.length > 1 ? 's' : ''} sélectionné{selectedFullModels.length > 1 ? 's' : ''} • GPU: {useGpu ? 'Activé' : 'Désactivé'}
                </span>
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing || !aiEngineReady || selectedFullModels.length === 0 || (batchMode && selectedFiles.length === 0)}
              className={`
                w-full py-4 rounded-xl font-semibold text-lg transition-all duration-200
                ${isAnalyzing || !aiEngineReady || selectedFullModels.length === 0 || (batchMode && selectedFiles.length === 0)
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : batchMode
                  ? 'bg-gradient-to-r from-purple-500 to-indigo-600 text-white hover:shadow-lg hover:scale-[1.02]'
                  : 'bg-gradient-to-r from-red-500 to-pink-600 text-white hover:shadow-lg hover:scale-[1.02]'
                }
              `}
            >
              {isAnalyzing ? (
                <div className="flex items-center justify-center gap-3">
                  <svg className="w-6 h-6 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  {batchMode ? `Traitement en cours (${selectedFiles.length} fichiers)...` : 'Analyse en cours...'}
                </div>
              ) : batchMode ? (
                `Lancer l'Analyse par Lot (${selectedFiles.length} fichiers × ${selectedFullModels.length} modèles)`
              ) : (
                `Lancer l'Analyse Complète (${selectedFullModels.length} modèles)`
              )}
            </button>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl">
            <div className="flex items-center gap-3 text-red-700">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="font-medium">{error}</span>
            </div>
          </div>
        )}

        {/* Results */}
        {/* ECG Viewer Toggle Button */}
        {fullAnalysisResult && (
          <div className="mb-4">
            <button
              onClick={() => setShowECGViewer(!showECGViewer)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                showECGViewer
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <span className="text-xl">💓</span>
              {showECGViewer ? 'Masquer le tracé ECG' : 'Afficher le tracé ECG'}
            </button>
          </div>
        )}

        {/* ECG Viewer */}
        {showECGViewer && fullAnalysisResult && (
          <div className="mb-6">
            <ECGViewer
              filename={fullAnalysisResult.ecg_filename}
              onClose={() => setShowECGViewer(false)}
            />
          </div>
        )}

        {/* Batch Results with Navigation */}
        {batchResults && (
          <div className="space-y-4">
            {/* Batch Summary Header */}
            <div className="p-4 bg-purple-50 border border-purple-200 rounded-xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
                    <span className="text-2xl">📊</span>
                  </div>
                  <div>
                    <h3 className="font-bold text-purple-800">Traitement par Lot Terminé</h3>
                    <p className="text-sm text-purple-600">
                      {batchResults.successful} réussi{batchResults.successful > 1 ? 's' : ''} / {batchResults.total_files} fichier{batchResults.total_files > 1 ? 's' : ''} • {(batchResults.total_processing_time_ms / 1000).toFixed(1)}s
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {batchResults.failed > 0 && (
                    <span className="px-2 py-1 bg-red-100 text-red-700 rounded-full text-sm font-medium">
                      {batchResults.failed} échec{batchResults.failed > 1 ? 's' : ''}
                    </span>
                  )}
                  {/* Export CSV Button */}
                  <button
                    onClick={exportBatchResultsToCSV}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition-colors shadow-sm"
                    title="Exporter tous les résultats en CSV pour la recherche"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Exporter CSV
                  </button>
                </div>
              </div>
            </div>

            {/* ECG Navigation */}
            <div className="flex items-center justify-between p-3 bg-gray-100 rounded-xl">
              <button
                onClick={goToPreviousECG}
                disabled={currentBatchIndex === 0}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  currentBatchIndex === 0
                    ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    : 'bg-white text-gray-700 hover:bg-gray-50 shadow-sm'
                }`}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Précédent
              </button>

              <div className="flex items-center gap-2">
                {batchResults.results.map((_, idx) => (
                  <button
                    key={idx}
                    onClick={() => goToECGIndex(idx)}
                    className={`w-8 h-8 rounded-full font-medium text-sm transition-colors ${
                      idx === currentBatchIndex
                        ? 'bg-purple-500 text-white'
                        : batchResults.results[idx].success
                        ? 'bg-white text-gray-700 hover:bg-gray-50'
                        : 'bg-red-100 text-red-700 hover:bg-red-200'
                    }`}
                  >
                    {idx + 1}
                  </button>
                ))}
              </div>

              <button
                onClick={goToNextECG}
                disabled={currentBatchIndex >= batchResults.results.length - 1}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  currentBatchIndex >= batchResults.results.length - 1
                    ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    : 'bg-white text-gray-700 hover:bg-gray-50 shadow-sm'
                }`}
              >
                Suivant
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>

            {/* Current ECG Info */}
            {currentBatchResult && (
              <div className={`p-3 rounded-xl ${currentBatchResult.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className={`text-2xl ${currentBatchResult.success ? '' : ''}`}>
                      {currentBatchResult.success ? '✅' : '❌'}
                    </span>
                    <div>
                      <p className="font-semibold text-gray-800">
                        ECG {currentBatchIndex + 1}/{batchResults.total_files}: {currentBatchResult.filename}
                      </p>
                      <p className={`text-sm ${currentBatchResult.success ? 'text-green-600' : 'text-red-600'}`}>
                        {currentBatchResult.success ? `Patient: ${currentBatchResult.patient_id}` : currentBatchResult.error}
                      </p>
                    </div>
                  </div>

                  {/* ECG Viewer Toggle for Batch */}
                  {currentBatchResult.success && currentBatchResult.result && (
                    <button
                      onClick={() => setShowECGViewer(!showECGViewer)}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                        showECGViewer
                          ? 'bg-green-600 text-white'
                          : 'bg-gray-700 text-white hover:bg-gray-600'
                      }`}
                    >
                      <span className="text-lg">💓</span>
                      {showECGViewer ? 'Masquer ECG' : 'Voir ECG'}
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* ECG Viewer for Batch Results */}
            {showECGViewer && currentBatchResult?.success && currentBatchResult.result && (
              <div className="mb-4">
                <ECGViewer
                  filename={currentBatchResult.result.ecg_filename}
                  onClose={() => setShowECGViewer(false)}
                />
              </div>
            )}

            {/* Display current ECG result */}
            {currentBatchResult?.success && currentBatchResult.result && (
              <ECGVisualResults
                result={currentBatchResult.result as FullECGAnalysisResult}
                onReset={handleReset}
              />
            )}

            {/* Error display for failed ECG */}
            {currentBatchResult && !currentBatchResult.success && (
              <div className="p-6 bg-red-50 border border-red-200 rounded-xl text-center">
                <div className="w-16 h-16 mx-auto mb-4 bg-red-100 rounded-full flex items-center justify-center">
                  <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h4 className="font-semibold text-red-800 mb-2">Échec de l'analyse</h4>
                <p className="text-red-600">{currentBatchResult.error}</p>
              </div>
            )}
          </div>
        )}

        {fullAnalysisResult && (
          <div className="space-y-4">
            {/* Export CSV Button for Single Analysis */}
            <div className="flex justify-end">
              <button
                onClick={exportSingleResultToCSV}
                className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition-colors shadow-sm"
                title="Exporter les résultats en CSV pour la recherche"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Exporter CSV
              </button>
            </div>
            <ECGVisualResults result={fullAnalysisResult} onReset={handleReset} />
          </div>
        )}
      </div>
    </div>
  );
};

export default ECGAnalysisPanel;
