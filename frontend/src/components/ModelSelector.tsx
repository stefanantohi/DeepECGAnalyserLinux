import React, { useState, useEffect } from 'react';

export interface ECGModel {
  id: string;
  name: string;
  architecture: 'efficientnet' | 'wcr';
  type: 'multi_label' | 'binary';
  description?: string;
}

interface ModelSelectorProps {
  availableModels: ECGModel[];
  selectedModels: string[];
  onSelectionChange: (models: string[]) => void;
  disabled?: boolean;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  availableModels,
  selectedModels,
  onSelectionChange,
  disabled = false,
}) => {
  const [selectAll, setSelectAll] = useState(true);
  const [expandedArchitecture, setExpandedArchitecture] = useState<string | null>('efficientnet');

  // Group models by architecture
  const modelsByArchitecture = availableModels.reduce((acc, model) => {
    if (!acc[model.architecture]) {
      acc[model.architecture] = [];
    }
    acc[model.architecture].push(model);
    return acc;
  }, {} as Record<string, ECGModel[]>);

  // Check if all models are selected
  useEffect(() => {
    setSelectAll(selectedModels.length === availableModels.length);
  }, [selectedModels, availableModels]);

  const handleSelectAll = () => {
    if (selectAll) {
      onSelectionChange([]);
    } else {
      onSelectionChange(availableModels.map(m => m.id));
    }
    setSelectAll(!selectAll);
  };

  const handleSelectArchitecture = (arch: string) => {
    const archModels = modelsByArchitecture[arch]?.map(m => m.id) || [];
    const otherSelected = selectedModels.filter(id => !archModels.includes(id));
    const allArchSelected = archModels.every(id => selectedModels.includes(id));

    if (allArchSelected) {
      onSelectionChange(otherSelected);
    } else {
      onSelectionChange([...new Set([...otherSelected, ...archModels])]);
    }
  };

  const handleToggleModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      onSelectionChange(selectedModels.filter(id => id !== modelId));
    } else {
      onSelectionChange([...selectedModels, modelId]);
    }
  };

  const isArchitectureSelected = (arch: string) => {
    const archModels = modelsByArchitecture[arch]?.map(m => m.id) || [];
    return archModels.every(id => selectedModels.includes(id));
  };

  const isArchitecturePartiallySelected = (arch: string) => {
    const archModels = modelsByArchitecture[arch]?.map(m => m.id) || [];
    const selectedCount = archModels.filter(id => selectedModels.includes(id)).length;
    return selectedCount > 0 && selectedCount < archModels.length;
  };

  const getModelTypeIcon = (type: string) => {
    return type === 'multi_label' ? '77' : '01';
  };

  const getModelTypeLabel = (type: string) => {
    return type === 'multi_label' ? '77 Classes' : 'Binaire';
  };

  return (
    <div className={`space-y-4 ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
      {/* Header with Select All */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800">Sélection des Modèles</h3>
        <button
          onClick={handleSelectAll}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            selectAll
              ? 'bg-red-500 text-white hover:bg-red-600'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          {selectAll ? 'Tout désélectionner' : 'Tout sélectionner'}
        </button>
      </div>

      {/* Selection count */}
      <div className="text-sm text-gray-500">
        {selectedModels.length} modèle{selectedModels.length !== 1 ? 's' : ''} sélectionné{selectedModels.length !== 1 ? 's' : ''} sur {availableModels.length}
      </div>

      {/* Models by Architecture */}
      <div className="space-y-3">
        {Object.entries(modelsByArchitecture).map(([architecture, models]) => (
          <div
            key={architecture}
            className="border border-gray-200 rounded-xl overflow-hidden"
          >
            {/* Architecture Header */}
            <div
              className={`flex items-center justify-between p-4 cursor-pointer transition-colors ${
                isArchitectureSelected(architecture)
                  ? 'bg-red-50'
                  : 'bg-gray-50 hover:bg-gray-100'
              }`}
              onClick={() => setExpandedArchitecture(
                expandedArchitecture === architecture ? null : architecture
              )}
            >
              <div className="flex items-center gap-3">
                {/* Checkbox */}
                <div
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSelectArchitecture(architecture);
                  }}
                  className={`w-5 h-5 rounded border-2 flex items-center justify-center cursor-pointer transition-colors ${
                    isArchitectureSelected(architecture)
                      ? 'bg-red-500 border-red-500'
                      : isArchitecturePartiallySelected(architecture)
                      ? 'bg-red-200 border-red-400'
                      : 'border-gray-300 hover:border-red-400'
                  }`}
                >
                  {(isArchitectureSelected(architecture) || isArchitecturePartiallySelected(architecture)) && (
                    <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  )}
                </div>

                {/* Architecture name */}
                <div>
                  <span className="font-semibold text-gray-800 uppercase">
                    {architecture === 'efficientnet' ? 'EfficientNet V2' : 'WCR Transformer'}
                  </span>
                  <span className="ml-2 text-sm text-gray-500">
                    ({models.length} modèles)
                  </span>
                </div>
              </div>

              {/* Expand/Collapse icon */}
              <svg
                className={`w-5 h-5 text-gray-400 transition-transform ${
                  expandedArchitecture === architecture ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>

            {/* Models List */}
            {expandedArchitecture === architecture && (
              <div className="divide-y divide-gray-100">
                {models.map((model) => (
                  <label
                    key={model.id}
                    className={`flex items-center gap-4 p-4 cursor-pointer transition-colors ${
                      selectedModels.includes(model.id)
                        ? 'bg-red-50/50'
                        : 'hover:bg-gray-50'
                    }`}
                  >
                    {/* Checkbox */}
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(model.id)}
                      onChange={() => handleToggleModel(model.id)}
                      className="w-4 h-4 rounded border-gray-300 text-red-500 focus:ring-red-500"
                    />

                    {/* Model Type Badge */}
                    <div
                      className={`w-10 h-10 rounded-lg flex items-center justify-center text-xs font-bold ${
                        model.type === 'multi_label'
                          ? 'bg-purple-100 text-purple-700'
                          : 'bg-blue-100 text-blue-700'
                      }`}
                    >
                      {getModelTypeIcon(model.type)}
                    </div>

                    {/* Model Info */}
                    <div className="flex-1">
                      <div className="font-medium text-gray-800">{model.name}</div>
                      <div className="text-sm text-gray-500">
                        {getModelTypeLabel(model.type)}
                        {model.description && ` - ${model.description}`}
                      </div>
                    </div>

                    {/* Status indicator */}
                    {selectedModels.includes(model.id) && (
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                    )}
                  </label>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Quick selection buttons */}
      <div className="flex gap-2 pt-2">
        <button
          onClick={() => {
            const efficientnetModels = modelsByArchitecture['efficientnet']?.map(m => m.id) || [];
            onSelectionChange(efficientnetModels);
          }}
          className="flex-1 py-2 px-4 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
        >
          EfficientNet uniquement
        </button>
        <button
          onClick={() => {
            const wcrModels = modelsByArchitecture['wcr']?.map(m => m.id) || [];
            onSelectionChange(wcrModels);
          }}
          className="flex-1 py-2 px-4 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
        >
          WCR uniquement
        </button>
      </div>
    </div>
  );
};

export default ModelSelector;
