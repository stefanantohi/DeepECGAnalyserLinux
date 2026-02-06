import React, { useState, useEffect } from 'react';
import {
  getDockerStatus,
  startDockerEngine,
  stopDockerEngine,
  getConfig,
  updateConfig,
  DockerStatusResponse,
  ConfigResponse,
} from '../api';

interface SystemStatusPanelProps {
  onEngineStatusChange: (ready: boolean) => void;
}

const SystemStatusPanel: React.FC<SystemStatusPanelProps> = ({ onEngineStatusChange }) => {
  const [dockerStatus, setDockerStatus] = useState<DockerStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [aiHealthy, setAiHealthy] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [workspacePath, setWorkspacePath] = useState('');
  const [workspaceConfig, setWorkspaceConfig] = useState<ConfigResponse | null>(null);
  const [isEditingPath, setIsEditingPath] = useState(false);
  const [editPath, setEditPath] = useState('');
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const status = await getDockerStatus();
      setDockerStatus(status);
      setError(null); // Clear error on success

      // Container is ready when it's running - no API health check needed
      // This container uses docker exec commands, not an HTTP API
      if (status.container_running) {
        setAiHealthy(true);
        onEngineStatusChange(true);
      } else {
        setAiHealthy(false);
        onEngineStatusChange(false);
      }
    } catch (err) {
      setError('Failed to get status');
      onEngineStatusChange(false);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchConfig = async () => {
    try {
      const config = await getConfig();
      setWorkspaceConfig(config);
      setWorkspacePath(config.workspace_path);
      setConfigError(null);
    } catch (err) {
      setConfigError('Failed to load config');
    }
  };

  const handleSaveConfig = async () => {
    setIsSavingConfig(true);
    setConfigError(null);
    try {
      const config = await updateConfig(editPath);
      setWorkspaceConfig(config);
      setWorkspacePath(config.workspace_path);
      setIsEditingPath(false);
    } catch (err: any) {
      setConfigError(err?.response?.data?.detail || 'Failed to update config');
    } finally {
      setIsSavingConfig(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchConfig();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    setIsStarting(true);
    setError(null);
    try {
      // Pass workspace path to mount as /data
      const result = await startDockerEngine(undefined, workspacePath || undefined);
      if (result.success) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        await fetchStatus();
      } else {
        setError(result.error || 'Failed to start');
      }
    } catch (err) {
      setError('Failed to start engine');
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    setIsStopping(true);
    setError(null);
    try {
      await stopDockerEngine();
      await fetchStatus();
    } catch (err) {
      setError('Failed to stop engine');
    } finally {
      setIsStopping(false);
    }
  };

  const getOverallStatus = () => {
    if (isLoading) return 'loading';
    if (!dockerStatus?.docker_running) return 'docker-off';
    if (!dockerStatus?.container_running) return 'stopped';
    if (aiHealthy) return 'ready';
    return 'starting';
  };

  const status = getOverallStatus();

  const statusConfig = {
    'loading': { color: 'gray', text: 'Checking...', bg: 'bg-gray-100' },
    'docker-off': { color: 'red', text: 'Docker Off', bg: 'bg-red-50' },
    'stopped': { color: 'yellow', text: 'Stopped', bg: 'bg-yellow-50' },
    'starting': { color: 'blue', text: 'Starting...', bg: 'bg-blue-50' },
    'ready': { color: 'green', text: 'Ready', bg: 'bg-green-50' },
  };

  const config = statusConfig[status];

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      {/* Compact Header - Always visible */}
      <div
        className={`p-4 cursor-pointer transition-colors ${config.bg}`}
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${
              status === 'ready' ? 'bg-green-500' :
              status === 'starting' ? 'bg-blue-500 animate-pulse' :
              status === 'stopped' ? 'bg-yellow-500' :
              'bg-red-500'
            }`} />
            <div>
              <h3 className="font-semibold text-gray-800 text-sm">AI Engine</h3>
              <p className={`text-xs ${
                status === 'ready' ? 'text-green-600' :
                status === 'starting' ? 'text-blue-600' :
                status === 'stopped' ? 'text-yellow-600' :
                'text-red-600'
              }`}>
                {config.text}
              </p>
            </div>
          </div>
          <svg
            className={`w-5 h-5 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>

        {/* Quick action button when not expanded */}
        {!expanded && status === 'stopped' && (
          <button
            onClick={(e) => { e.stopPropagation(); handleStart(); }}
            disabled={isStarting}
            className="mt-3 w-full py-2 bg-green-500 hover:bg-green-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
          >
            {isStarting ? 'Starting...' : 'Start Engine'}
          </button>
        )}
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="p-4 border-t border-gray-100 space-y-4">
          {/* Status Items */}
          <div className="space-y-2">
            <StatusItem
              label="Docker"
              status={dockerStatus?.docker_running ? 'ok' : 'error'}
              detail={dockerStatus?.docker_running ? 'Running' : 'Not running'}
            />
            <StatusItem
              label="GPU"
              status={dockerStatus?.gpu_available ? 'ok' : 'warning'}
              detail={dockerStatus?.gpu_available ? 'Available' : 'Not detected'}
            />
            <StatusItem
              label="Container"
              status={dockerStatus?.container_running ? 'ok' : 'off'}
              detail={dockerStatus?.container_running ? 'Running' : 'Stopped'}
            />
            <StatusItem
              label="Engine"
              status={aiHealthy ? 'ok' : 'off'}
              detail={aiHealthy ? 'Ready' : 'Not ready'}
            />
          </div>

          {/* Error Display */}
          {error && (
            <div className="p-2 bg-red-50 border border-red-200 rounded-lg text-xs text-red-600">
              {error}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-2">
            {!dockerStatus?.container_running ? (
              <button
                onClick={handleStart}
                disabled={isStarting || !dockerStatus?.docker_running}
                className="flex-1 py-2 bg-green-500 hover:bg-green-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isStarting ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Starting...
                  </span>
                ) : 'Start Engine'}
              </button>
            ) : (
              <button
                onClick={handleStop}
                disabled={isStopping}
                className="flex-1 py-2 bg-red-500 hover:bg-red-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
              >
                {isStopping ? 'Stopping...' : 'Stop Engine'}
              </button>
            )}
            <button
              onClick={fetchStatus}
              className="px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-lg transition-colors"
              title="Refresh status"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>

          {/* Workspace Configuration */}
          <div className="border-t border-gray-100 pt-3">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide">Workspace</h4>
              {!isEditingPath && !dockerStatus?.container_running && (
                <button
                  onClick={() => { setEditPath(workspacePath); setIsEditingPath(true); }}
                  className="text-xs text-blue-600 hover:text-blue-700"
                >
                  Modifier
                </button>
              )}
            </div>

            {isEditingPath ? (
              <div className="space-y-2">
                <input
                  type="text"
                  value={editPath}
                  onChange={(e) => setEditPath(e.target.value)}
                  placeholder="C:/path/to/workspace"
                  className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded-lg focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleSaveConfig}
                    disabled={isSavingConfig || !editPath.trim()}
                    className="flex-1 py-1.5 bg-blue-500 hover:bg-blue-600 text-white text-xs font-medium rounded-lg transition-colors disabled:opacity-50"
                  >
                    {isSavingConfig ? 'Saving...' : 'Enregistrer'}
                  </button>
                  <button
                    onClick={() => { setIsEditingPath(false); setConfigError(null); }}
                    className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-600 text-xs rounded-lg transition-colors"
                  >
                    Annuler
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-xs text-gray-600 break-all bg-gray-50 px-2 py-1.5 rounded-lg">
                {workspacePath || '...'}
              </div>
            )}

            {configError && (
              <div className="mt-1 text-xs text-red-600">{configError}</div>
            )}

            {/* Subdirectories status */}
            {workspaceConfig && (
              <div className="mt-2 space-y-1">
                {Object.entries(workspaceConfig.subdirectories).map(([name, exists]) => (
                  <div key={name} className="flex items-center justify-between text-xs">
                    <span className="text-gray-500 font-mono">{name}/</span>
                    <div className={`w-2 h-2 rounded-full ${exists ? 'bg-green-500' : 'bg-red-500'}`} />
                  </div>
                ))}
              </div>
            )}

            {workspaceConfig && (!workspaceConfig.workspace_exists || Object.values(workspaceConfig.subdirectories).some(v => !v)) && (
              <button
                onClick={async () => {
                  setIsSavingConfig(true);
                  try {
                    const config = await updateConfig(workspacePath);
                    setWorkspaceConfig(config);
                  } catch (err: any) {
                    setConfigError(err?.response?.data?.detail || 'Failed to create directories');
                  } finally {
                    setIsSavingConfig(false);
                  }
                }}
                disabled={isSavingConfig}
                className="mt-2 w-full py-1.5 bg-amber-500 hover:bg-amber-600 text-white text-xs font-medium rounded-lg transition-colors disabled:opacity-50"
              >
                {isSavingConfig ? 'Création...' : 'Créer les répertoires manquants'}
              </button>
            )}

            {dockerStatus?.container_running && (
              <p className="text-xs text-gray-400 mt-1 italic">
                Arrêtez le container pour modifier le chemin
              </p>
            )}
          </div>

          {/* Help text */}
          {!dockerStatus?.docker_running && (
            <p className="text-xs text-gray-500 text-center">
              Please start Docker Desktop first
            </p>
          )}
        </div>
      )}
    </div>
  );
};

const StatusItem: React.FC<{
  label: string;
  status: 'ok' | 'warning' | 'error' | 'off';
  detail: string;
}> = ({ label, status, detail }) => (
  <div className="flex items-center justify-between py-1">
    <span className="text-sm text-gray-600">{label}</span>
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-500">{detail}</span>
      <div className={`w-2 h-2 rounded-full ${
        status === 'ok' ? 'bg-green-500' :
        status === 'warning' ? 'bg-yellow-500' :
        status === 'error' ? 'bg-red-500' :
        'bg-gray-300'
      }`} />
    </div>
  </div>
);

export default SystemStatusPanel;
