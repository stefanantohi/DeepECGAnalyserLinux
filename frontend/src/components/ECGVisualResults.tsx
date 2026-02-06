import React, { useState } from 'react';

// Types
export interface ECGDiagnosis {
  name: string;
  probability: number;
  threshold: number;
  status: 'normal' | 'borderline' | 'abnormal';
  category: string;
}

export interface ECGModelResult {
  model_id: string;
  model_name: string;
  model_type: string;
  architecture: string;
  success: boolean;
  error?: string;
  diagnoses: ECGDiagnosis[];
  by_category: Record<string, ECGDiagnosis[]>;
}

export interface ECGCriticalFinding {
  diagnosis: string;
  probability: number;
  model: string;
  category?: string;
}

export interface ECGAnalysisSummary {
  overall_status: 'normal' | 'borderline' | 'abnormal';
  total_abnormal: number;
  total_borderline: number;
  critical_findings: ECGCriticalFinding[];
}

export interface ECGFileFormatInfo {
  original_format: string;
  original_encoding: string;
  conversions_applied: string[];
  conversion_notes?: string;
}

export interface FullECGAnalysisResult {
  success: boolean;
  patient_id: string;
  ecg_filename: string;
  file_format_info?: ECGFileFormatInfo;
  models_executed: string[];
  results: Record<string, ECGModelResult>;
  summary: ECGAnalysisSummary;
  warnings: string[];
  processing_time_ms: number;
  error?: string;
}

interface ECGVisualResultsProps {
  result: FullECGAnalysisResult;
  onReset: () => void;
}

// Status colors and icons
const getStatusConfig = (status: 'normal' | 'borderline' | 'abnormal') => {
  switch (status) {
    case 'normal':
      return {
        color: 'green',
        bgColor: 'bg-green-500',
        bgLight: 'bg-green-50',
        borderColor: 'border-green-200',
        textColor: 'text-green-700',
        icon: '✓',
        label: 'Normal',
      };
    case 'borderline':
      return {
        color: 'yellow',
        bgColor: 'bg-yellow-500',
        bgLight: 'bg-yellow-50',
        borderColor: 'border-yellow-200',
        textColor: 'text-yellow-700',
        icon: '⚠',
        label: 'Borderline',
      };
    case 'abnormal':
      return {
        color: 'red',
        bgColor: 'bg-red-500',
        bgLight: 'bg-red-50',
        borderColor: 'border-red-200',
        textColor: 'text-red-700',
        icon: '✗',
        label: 'Anormal',
      };
  }
};

// Progress bar component
const ProgressBar: React.FC<{
  value: number;
  threshold: number;
  status: 'normal' | 'borderline' | 'abnormal';
}> = ({ value, threshold, status }) => {
  const config = getStatusConfig(status);

  return (
    <div className="relative w-full h-3 bg-gray-200 rounded-full overflow-hidden">
      {/* Value bar */}
      <div
        className={`absolute h-full ${config.bgColor} transition-all duration-500`}
        style={{ width: `${Math.min(value, 100)}%` }}
      />
      {/* Threshold marker */}
      <div
        className="absolute h-full w-0.5 bg-gray-600"
        style={{ left: `${threshold}%` }}
        title={`Seuil: ${threshold}%`}
      />
    </div>
  );
};

// Diagnosis card component
const DiagnosisCard: React.FC<{ diagnosis: ECGDiagnosis }> = ({ diagnosis }) => {
  const config = getStatusConfig(diagnosis.status);

  return (
    <div className={`p-3 rounded-lg ${config.bgLight} ${config.borderColor} border`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`text-lg ${config.textColor}`}>{config.icon}</span>
          <span className="font-medium text-gray-800 text-sm">{diagnosis.name}</span>
        </div>
        <span className={`font-bold ${config.textColor}`}>
          {diagnosis.probability.toFixed(1)}%
        </span>
      </div>
      <ProgressBar
        value={diagnosis.probability}
        threshold={diagnosis.threshold}
        status={diagnosis.status}
      />
      <div className="flex justify-between mt-1 text-xs text-gray-500">
        <span>Seuil: {diagnosis.threshold}%</span>
        <span className={config.textColor}>{config.label}</span>
      </div>
    </div>
  );
};

// Category accordion component
const CategoryAccordion: React.FC<{
  category: string;
  diagnoses: ECGDiagnosis[];
  defaultExpanded?: boolean;
}> = ({ category, diagnoses, defaultExpanded = false }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const abnormalCount = diagnoses.filter(d => d.status === 'abnormal').length;
  const borderlineCount = diagnoses.filter(d => d.status === 'borderline').length;
  const hasIssues = abnormalCount > 0 || borderlineCount > 0;

  return (
    <div className={`border rounded-xl overflow-hidden ${hasIssues ? 'border-yellow-300' : 'border-gray-200'}`}>
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`w-full flex items-center justify-between p-4 text-left transition-colors ${
          hasIssues ? 'bg-yellow-50 hover:bg-yellow-100' : 'bg-gray-50 hover:bg-gray-100'
        }`}
      >
        <div className="flex items-center gap-3">
          <span className="font-semibold text-gray-800">{category}</span>
          <span className="text-sm text-gray-500">({diagnoses.length})</span>
          {abnormalCount > 0 && (
            <span className="px-2 py-0.5 text-xs font-medium bg-red-100 text-red-700 rounded-full">
              {abnormalCount} anormal{abnormalCount > 1 ? 's' : ''}
            </span>
          )}
          {borderlineCount > 0 && (
            <span className="px-2 py-0.5 text-xs font-medium bg-yellow-100 text-yellow-700 rounded-full">
              {borderlineCount} limite{borderlineCount > 1 ? 's' : ''}
            </span>
          )}
        </div>
        <svg
          className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          {diagnoses
            .sort((a, b) => {
              // Sort by status (abnormal first) then by probability
              const statusOrder = { abnormal: 0, borderline: 1, normal: 2 };
              if (statusOrder[a.status] !== statusOrder[b.status]) {
                return statusOrder[a.status] - statusOrder[b.status];
              }
              return b.probability - a.probability;
            })
            .map((diag, idx) => (
              <DiagnosisCard key={idx} diagnosis={diag} />
            ))}
        </div>
      )}
    </div>
  );
};

// Binary result card (for LVEF, AF models)
const BinaryResultCard: React.FC<{
  modelId: string;
  modelName: string;
  result: ECGModelResult;
}> = ({ modelId, modelName, result }) => {
  // For binary models, get the primary diagnosis
  const primaryDiag = result.diagnoses[0];
  if (!primaryDiag) return null;

  const config = getStatusConfig(primaryDiag.status);

  const getModelIcon = (id: string) => {
    if (id.includes('lvef')) return '❤️';
    if (id.includes('afib')) return '⚡';
    return '📊';
  };

  return (
    <div className={`p-5 rounded-xl ${config.bgLight} ${config.borderColor} border-2`}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{getModelIcon(modelId)}</span>
          <div>
            <h4 className="font-bold text-gray-800">{modelName}</h4>
            <p className="text-sm text-gray-500">{result.architecture.toUpperCase()}</p>
          </div>
        </div>
        <div className={`flex items-center gap-1 px-3 py-1 rounded-full ${config.bgColor} text-white`}>
          <span>{config.icon}</span>
          <span className="font-medium">{config.label}</span>
        </div>
      </div>

      <div className="mb-3">
        <div className="flex justify-between mb-1">
          <span className="text-sm text-gray-600">Probabilité</span>
          <span className={`font-bold text-xl ${config.textColor}`}>
            {primaryDiag.probability.toFixed(1)}%
          </span>
        </div>
        <ProgressBar
          value={primaryDiag.probability}
          threshold={primaryDiag.threshold}
          status={primaryDiag.status}
        />
        <div className="text-xs text-gray-500 mt-1">
          Seuil de détection: {primaryDiag.threshold}%
        </div>
      </div>
    </div>
  );
};

// Multi-label model comparison card
const MultiLabelModelCard: React.FC<{
  modelId: string;
  modelResult: ECGModelResult;
  compact?: boolean;
}> = ({ modelId, modelResult, compact = false }) => {
  const [expanded, setExpanded] = useState(!compact);

  const abnormalCount = modelResult.diagnoses.filter(d => d.status === 'abnormal').length;
  const borderlineCount = modelResult.diagnoses.filter(d => d.status === 'borderline').length;
  const hasIssues = abnormalCount > 0 || borderlineCount > 0;

  const overallStatus = abnormalCount > 0 ? 'abnormal' : borderlineCount > 0 ? 'borderline' : 'normal';
  const config = getStatusConfig(overallStatus);

  return (
    <div className={`rounded-xl border-2 ${config.borderColor} overflow-hidden`}>
      <div
        className={`p-4 ${config.bgLight} cursor-pointer`}
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg ${config.bgColor} flex items-center justify-center`}>
              <span className="text-white font-bold text-sm">77</span>
            </div>
            <div>
              <h4 className="font-bold text-gray-800">{modelResult.model_name}</h4>
              <p className="text-xs text-gray-500">{modelResult.architecture.toUpperCase()}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {abnormalCount > 0 && (
              <span className="px-2 py-1 bg-red-100 text-red-700 rounded-full text-xs font-medium">
                {abnormalCount} anormal{abnormalCount > 1 ? 's' : ''}
              </span>
            )}
            {borderlineCount > 0 && (
              <span className="px-2 py-1 bg-yellow-100 text-yellow-700 rounded-full text-xs font-medium">
                {borderlineCount} limite{borderlineCount > 1 ? 's' : ''}
              </span>
            )}
            <svg
              className={`w-5 h-5 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </div>

      {expanded && (
        <div className="p-4 bg-white max-h-96 overflow-y-auto">
          <div className="grid grid-cols-1 gap-2">
            {modelResult.diagnoses
              .filter(d => d.status !== 'normal')
              .sort((a, b) => {
                const statusOrder = { abnormal: 0, borderline: 1, normal: 2 };
                if (statusOrder[a.status] !== statusOrder[b.status]) {
                  return statusOrder[a.status] - statusOrder[b.status];
                }
                return b.probability - a.probability;
              })
              .slice(0, 10)
              .map((diag, idx) => (
                <DiagnosisCard key={idx} diagnosis={diag} />
              ))}
            {modelResult.diagnoses.filter(d => d.status !== 'normal').length === 0 && (
              <p className="text-center text-gray-500 py-4">Aucune anomalie détectée</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Comparison table for multi-label models
const ModelComparisonTable: React.FC<{
  models: [string, ECGModelResult][];
}> = ({ models }) => {
  // Get all unique diagnosis names
  const allDiagnosisNames = new Set<string>();
  models.forEach(([_, model]) => {
    model.diagnoses.forEach(d => allDiagnosisNames.add(d.name));
  });

  // Create comparison data
  const comparisonData = Array.from(allDiagnosisNames).map(name => {
    const row: Record<string, any> = { name };
    models.forEach(([modelId, model]) => {
      const diag = model.diagnoses.find(d => d.name === name);
      row[modelId] = diag || null;
    });
    return row;
  });

  // Sort by max probability across models
  comparisonData.sort((a, b) => {
    const maxA = Math.max(...models.map(([id]) => a[id]?.probability || 0));
    const maxB = Math.max(...models.map(([id]) => b[id]?.probability || 0));
    return maxB - maxA;
  });

  // Filter to show only items with differences or abnormal status
  const significantData = comparisonData.filter(row => {
    const values = models.map(([id]) => row[id]?.probability || 0);
    const maxDiff = Math.max(...values) - Math.min(...values);
    const hasAbnormal = models.some(([id]) => row[id]?.status === 'abnormal' || row[id]?.status === 'borderline');
    return maxDiff > 10 || hasAbnormal;
  }).slice(0, 20);

  if (significantData.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        Résultats similaires entre les modèles - aucune différence significative
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-gray-100">
            <th className="p-3 text-left font-semibold text-gray-700">Diagnostic</th>
            {models.map(([modelId, model]) => (
              <th key={modelId} className="p-3 text-center font-semibold text-gray-700">
                <div>{model.architecture.toUpperCase()}</div>
                <div className="text-xs font-normal text-gray-500">{model.model_name}</div>
              </th>
            ))}
            <th className="p-3 text-center font-semibold text-gray-700">Diff</th>
          </tr>
        </thead>
        <tbody>
          {significantData.map((row, idx) => {
            const values = models.map(([id]) => row[id]?.probability || 0);
            const diff = Math.max(...values) - Math.min(...values);

            return (
              <tr key={idx} className="border-b hover:bg-gray-50">
                <td className="p-3 font-medium text-gray-800">{row.name}</td>
                {models.map(([modelId]) => {
                  const diag = row[modelId];
                  if (!diag) {
                    return <td key={modelId} className="p-3 text-center text-gray-400">-</td>;
                  }
                  const config = getStatusConfig(diag.status);
                  return (
                    <td key={modelId} className="p-3 text-center">
                      <span className={`font-bold ${config.textColor}`}>
                        {diag.probability.toFixed(1)}%
                      </span>
                      <span className={`ml-2 text-xs ${config.textColor}`}>
                        {config.icon}
                      </span>
                    </td>
                  );
                })}
                <td className="p-3 text-center">
                  <span className={`font-medium ${diff > 20 ? 'text-orange-600' : 'text-gray-500'}`}>
                    {diff > 0 ? `±${diff.toFixed(0)}%` : '-'}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

// Main component
const ECGVisualResults: React.FC<ECGVisualResultsProps> = ({ result, onReset }) => {
  const [activeTab, setActiveTab] = useState<'summary' | 'details' | 'comparison' | 'all'>('summary');

  const summaryConfig = getStatusConfig(result.summary.overall_status);

  // Separate binary and multi-label results
  const binaryResults = Object.entries(result.results).filter(
    ([_, r]) => r.model_type === 'binary'
  );
  const multiLabelResults = Object.entries(result.results).filter(
    ([_, r]) => r.model_type === 'multi_label'
  );

  // Check if we have multiple multi-label models for comparison
  const hasMultipleMultiLabel = multiLabelResults.length > 1;

  // Aggregate all diagnoses by category from multi-label models
  const allDiagnosesByCategory: Record<string, ECGDiagnosis[]> = {};
  multiLabelResults.forEach(([modelId, modelResult]) => {
    Object.entries(modelResult.by_category).forEach(([category, diagnoses]) => {
      if (!allDiagnosesByCategory[category]) {
        allDiagnosesByCategory[category] = [];
      }
      // Add model info to diagnosis for comparison
      diagnoses.forEach(diag => {
        const existing = allDiagnosesByCategory[category].find(d => d.name === diag.name);
        if (!existing) {
          allDiagnosesByCategory[category].push({ ...diag, category: `${diag.category} (${modelResult.architecture})` });
        }
      });
    });
  });

  return (
    <div className="space-y-6">
      {/* Overall Status Banner */}
      <div className={`p-6 rounded-2xl ${summaryConfig.bgLight} ${summaryConfig.borderColor} border-2`}>
        <div className="flex items-center gap-4">
          <div className={`w-16 h-16 rounded-full ${summaryConfig.bgColor} flex items-center justify-center`}>
            <span className="text-3xl text-white">{summaryConfig.icon}</span>
          </div>
          <div className="flex-1">
            <h2 className={`text-2xl font-bold ${summaryConfig.textColor}`}>
              {result.summary.overall_status === 'normal'
                ? 'ECG Normal'
                : result.summary.overall_status === 'borderline'
                ? 'Résultats à surveiller'
                : 'Anomalies détectées'}
            </h2>
            <p className="text-gray-600 mt-1">
              {result.models_executed.length} modèle{result.models_executed.length > 1 ? 's' : ''} exécuté{result.models_executed.length > 1 ? 's' : ''} en {(result.processing_time_ms / 1000).toFixed(1)}s
            </p>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500">Patient</div>
            <div className="font-semibold text-gray-800">{result.patient_id}</div>
          </div>
        </div>

        {/* Critical findings */}
        {result.summary.critical_findings.length > 0 && (
          <div className="mt-4 pt-4 border-t border-red-200">
            <h4 className="font-semibold text-red-700 mb-2">Résultats critiques:</h4>
            <div className="flex flex-wrap gap-2">
              {result.summary.critical_findings.map((finding, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium"
                >
                  {finding.diagnosis}: {finding.probability.toFixed(1)}%
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <div className="text-3xl font-bold text-gray-800">{result.models_executed.length}</div>
          <div className="text-sm text-gray-500">Modèles</div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <div className="text-3xl font-bold text-green-600">
            {Object.values(result.results).reduce(
              (acc, r) => acc + r.diagnoses.filter(d => d.status === 'normal').length,
              0
            )}
          </div>
          <div className="text-sm text-gray-500">Normaux</div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <div className="text-3xl font-bold text-yellow-600">{result.summary.total_borderline}</div>
          <div className="text-sm text-gray-500">Limites</div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <div className="text-3xl font-bold text-red-600">{result.summary.total_abnormal}</div>
          <div className="text-sm text-gray-500">Anormaux</div>
        </div>
      </div>

      {/* File Format Info */}
      {result.file_format_info && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="text-xl">📄</span>
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-semibold text-blue-800">Format détecté:</span>
                <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-sm font-medium">
                  {result.file_format_info.original_format === 'philips_pagewriter' ? 'Philips PageWriter TC' :
                   result.file_format_info.original_format === 'mhi' ? 'MHI (Montreal Heart Institute)' :
                   result.file_format_info.original_format === 'ge_muse' ? 'GE MUSE' :
                   result.file_format_info.original_format === 'numpy' ? 'NumPy Array' :
                   result.file_format_info.original_format}
                </span>
                {result.file_format_info.original_encoding !== 'utf-8' && (
                  <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-sm">
                    {result.file_format_info.original_encoding.toUpperCase()}
                  </span>
                )}
              </div>
              {result.file_format_info.conversions_applied.length > 0 && (
                <div className="mt-2 flex items-center gap-2 flex-wrap">
                  <span className="text-sm text-blue-600">Conversions:</span>
                  {result.file_format_info.conversions_applied.map((conv, idx) => (
                    <span key={idx} className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs">
                      ✓ {conv}
                    </span>
                  ))}
                </div>
              )}
              {result.file_format_info.conversion_notes && (
                <p className="mt-1 text-xs text-blue-600 italic">
                  {result.file_format_info.conversion_notes}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex bg-gray-100 rounded-xl p-1">
        {[
          { id: 'summary', label: 'Résumé' },
          ...(hasMultipleMultiLabel ? [{ id: 'comparison', label: 'Comparaison' }] : []),
          { id: 'details', label: 'Par catégorie' },
          { id: 'all', label: 'Tous les résultats' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-white shadow-md text-red-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'summary' && (
        <div className="space-y-6">
          {/* Binary Results (LVEF, AF) */}
          {binaryResults.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-3">Modèles de Prédiction</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {binaryResults.map(([modelId, modelResult]) => (
                  <BinaryResultCard
                    key={modelId}
                    modelId={modelId}
                    modelName={modelResult.model_name}
                    result={modelResult}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Multi-label model cards (side by side when multiple) */}
          {multiLabelResults.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-3">
                Classification 77 Classes
                {hasMultipleMultiLabel && (
                  <span className="ml-2 text-sm font-normal text-gray-500">
                    ({multiLabelResults.length} modèles)
                  </span>
                )}
              </h3>
              <div className={`grid gap-4 ${hasMultipleMultiLabel ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
                {multiLabelResults.map(([modelId, modelResult]) => (
                  <MultiLabelModelCard
                    key={modelId}
                    modelId={modelId}
                    modelResult={modelResult}
                    compact={hasMultipleMultiLabel}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Top abnormal findings */}
          {result.summary.total_abnormal > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-3">
                Principaux résultats anormaux
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {multiLabelResults
                  .flatMap(([_, r]) => r.diagnoses)
                  .filter(d => d.status === 'abnormal')
                  .sort((a, b) => b.probability - a.probability)
                  .slice(0, 6)
                  .map((diag, idx) => (
                    <DiagnosisCard key={idx} diagnosis={diag} />
                  ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Comparison Tab - only when multiple multi-label models */}
      {activeTab === 'comparison' && hasMultipleMultiLabel && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
              <span>📊</span>
              Comparaison des Modèles 77 Classes
            </h3>
            <p className="text-sm text-gray-500 mb-4">
              Différences significatives entre EfficientNet et WCR Transformer (écart &gt;10% ou anomalies)
            </p>
            <ModelComparisonTable models={multiLabelResults} />
          </div>

          {/* Side by side model summaries */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {multiLabelResults.map(([modelId, modelResult]) => {
              const abnormalCount = modelResult.diagnoses.filter(d => d.status === 'abnormal').length;
              const borderlineCount = modelResult.diagnoses.filter(d => d.status === 'borderline').length;
              const normalCount = modelResult.diagnoses.filter(d => d.status === 'normal').length;

              return (
                <div key={modelId} className="bg-white rounded-xl border border-gray-200 p-5">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center">
                      <span className="text-white font-bold">77</span>
                    </div>
                    <div>
                      <h4 className="font-bold text-gray-800">{modelResult.model_name}</h4>
                      <p className="text-sm text-gray-500">{modelResult.architecture.toUpperCase()}</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="bg-green-50 rounded-lg p-3">
                      <div className="text-2xl font-bold text-green-600">{normalCount}</div>
                      <div className="text-xs text-green-700">Normaux</div>
                    </div>
                    <div className="bg-yellow-50 rounded-lg p-3">
                      <div className="text-2xl font-bold text-yellow-600">{borderlineCount}</div>
                      <div className="text-xs text-yellow-700">Limites</div>
                    </div>
                    <div className="bg-red-50 rounded-lg p-3">
                      <div className="text-2xl font-bold text-red-600">{abnormalCount}</div>
                      <div className="text-xs text-red-700">Anormaux</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {activeTab === 'details' && (
        <div className="space-y-4">
          {Object.entries(allDiagnosesByCategory)
            .sort(([_, a], [__, b]) => {
              // Sort categories by number of issues
              const aIssues = a.filter(d => d.status !== 'normal').length;
              const bIssues = b.filter(d => d.status !== 'normal').length;
              return bIssues - aIssues;
            })
            .map(([category, diagnoses]) => (
              <CategoryAccordion
                key={category}
                category={category}
                diagnoses={diagnoses}
                defaultExpanded={diagnoses.some(d => d.status !== 'normal')}
              />
            ))}
        </div>
      )}

      {activeTab === 'all' && (
        <div className="space-y-4">
          {/* All diagnoses in a table - with comparison if multiple models */}
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="bg-gray-100">
                  <th className="p-3 text-left font-semibold text-gray-700">Diagnostic</th>
                  <th className="p-3 text-left font-semibold text-gray-700">Catégorie</th>
                  {hasMultipleMultiLabel ? (
                    <>
                      {multiLabelResults.map(([modelId, model]) => (
                        <th key={modelId} className="p-3 text-center font-semibold text-gray-700">
                          <div>{model.architecture.toUpperCase()}</div>
                          <div className="text-xs font-normal text-gray-500">Prob / Status</div>
                        </th>
                      ))}
                      <th className="p-3 text-center font-semibold text-gray-700">Diff</th>
                    </>
                  ) : (
                    <>
                      <th className="p-3 text-center font-semibold text-gray-700">Probabilité</th>
                      <th className="p-3 text-center font-semibold text-gray-700">Seuil</th>
                      <th className="p-3 text-center font-semibold text-gray-700">Status</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {(() => {
                  // Get all unique diagnosis names
                  const allDiagNames = new Set<string>();
                  multiLabelResults.forEach(([_, r]) => r.diagnoses.forEach(d => allDiagNames.add(d.name)));

                  // Create lookup maps for each model
                  const modelMaps = multiLabelResults.map(([modelId, r]) => {
                    const map = new Map<string, ECGDiagnosis>();
                    r.diagnoses.forEach(d => map.set(d.name, d));
                    return { modelId, map, arch: r.architecture };
                  });

                  // Build rows with all diagnoses
                  return Array.from(allDiagNames)
                    .map(name => {
                      const firstDiag = modelMaps[0]?.map.get(name);
                      const diagsForName = modelMaps.map(m => m.map.get(name));
                      const maxProb = Math.max(...diagsForName.map(d => d?.probability || 0));
                      const worstStatus = diagsForName.reduce((worst, d) => {
                        if (!d) return worst;
                        if (d.status === 'abnormal') return 'abnormal';
                        if (d.status === 'borderline' && worst !== 'abnormal') return 'borderline';
                        return worst;
                      }, 'normal' as 'normal' | 'borderline' | 'abnormal');

                      return { name, firstDiag, diagsForName, maxProb, worstStatus };
                    })
                    .sort((a, b) => {
                      const statusOrder = { abnormal: 0, borderline: 1, normal: 2 };
                      if (statusOrder[a.worstStatus] !== statusOrder[b.worstStatus]) {
                        return statusOrder[a.worstStatus] - statusOrder[b.worstStatus];
                      }
                      return b.maxProb - a.maxProb;
                    })
                    .map((row, idx) => {
                      const rowConfig = getStatusConfig(row.worstStatus);

                      if (hasMultipleMultiLabel) {
                        const probs = row.diagsForName.map(d => d?.probability || 0);
                        const diff = probs.length >= 2 ? Math.abs(probs[0] - probs[1]) : 0;

                        return (
                          <tr key={idx} className={`border-b ${rowConfig.bgLight} hover:bg-opacity-75`}>
                            <td className="p-3 font-medium text-gray-800">{row.name}</td>
                            <td className="p-3 text-gray-600">{row.firstDiag?.category || '-'}</td>
                            {row.diagsForName.map((diag, i) => {
                              if (!diag) {
                                return <td key={i} className="p-3 text-center text-gray-400">-</td>;
                              }
                              const config = getStatusConfig(diag.status);
                              return (
                                <td key={i} className="p-3 text-center">
                                  <span className={`font-bold ${config.textColor}`}>
                                    {diag.probability.toFixed(1)}%
                                  </span>
                                  <span className={`ml-1 text-xs ${config.textColor}`}>
                                    {config.icon}
                                  </span>
                                </td>
                              );
                            })}
                            <td className="p-3 text-center">
                              <span className={`font-medium ${diff > 20 ? 'text-orange-600' : diff > 10 ? 'text-yellow-600' : 'text-gray-500'}`}>
                                {diff > 0 ? `±${diff.toFixed(1)}%` : '-'}
                              </span>
                            </td>
                          </tr>
                        );
                      } else {
                        const diag = row.firstDiag;
                        if (!diag) return null;
                        const config = getStatusConfig(diag.status);
                        return (
                          <tr key={idx} className={`border-b ${config.bgLight} hover:bg-opacity-75`}>
                            <td className="p-3 font-medium text-gray-800">{diag.name}</td>
                            <td className="p-3 text-gray-600">{diag.category}</td>
                            <td className="p-3 text-center">
                              <span className={`font-bold ${config.textColor}`}>
                                {diag.probability.toFixed(1)}%
                              </span>
                            </td>
                            <td className="p-3 text-center text-gray-500">{diag.threshold}%</td>
                            <td className="p-3 text-center">
                              <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-sm ${config.bgColor} text-white`}>
                                {config.icon} {config.label}
                              </span>
                            </td>
                          </tr>
                        );
                      }
                    });
                })()}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Warnings with special styling for ECG reconstruction */}
      {result.warnings.length > 0 && (
        <div className="space-y-3">
          {result.warnings.map((warning, idx) => {
            const isReconstruction = warning.includes('8 dérivations') || warning.includes('12 dérivations');

            if (isReconstruction) {
              return (
                <div key={idx} className="p-4 bg-blue-50 border border-blue-300 rounded-xl flex items-start gap-3">
                  <div className="flex-shrink-0 w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-blue-800">ECG Reconstruit</h4>
                    <p className="text-sm text-blue-700 mt-1">{warning}</p>
                    <p className="text-xs text-blue-600 mt-2">
                      Les dérivations III, aVR, aVL et aVF ont été calculées à partir de I et II
                    </p>
                  </div>
                </div>
              );
            }

            return (
              <div key={idx} className="p-4 bg-yellow-50 border border-yellow-200 rounded-xl flex items-start gap-3">
                <svg className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <p className="text-sm text-yellow-700">{warning}</p>
              </div>
            );
          })}
        </div>
      )}

      {/* Error */}
      {result.error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
          <h4 className="font-semibold text-red-800 mb-2">Erreur:</h4>
          <p className="text-sm text-red-700">{result.error}</p>
        </div>
      )}

      {/* Reset button */}
      <button
        onClick={onReset}
        className="w-full py-4 border-2 border-gray-300 rounded-xl text-gray-700 font-semibold hover:bg-gray-50 transition-colors"
      >
        Nouvelle Analyse
      </button>
    </div>
  );
};

export default ECGVisualResults;
